"""Chess Vision — Video Analysis GUI

Analyzes chess game videos by:
1. Detecting the chess board region using a GroundingDINO (GLIP) model
2. Batch-processing ALL frames through ChessNet (with caching)
3. Rendering a live chess board with annotations alongside the video

Usage:
    python video_gui.py
    python video_gui.py --video path/to/video.mp4
    python video_gui.py --model model.pth --device cuda
"""

import argparse
import bisect
import collections
import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time

import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps, ImageTk
import chess

from model import ChessNet
from generate_data import ChessGenerator
import augmentations
from board_detection import BoardDetector

try:
    # Ensure libmpv-2.dll next to this script is found
    os.environ["PATH"] = (
        os.path.dirname(os.path.abspath(__file__))
        + os.pathsep + os.environ.get("PATH", "")
    )
    import mpv as _mpv_module
except (ImportError, OSError):
    _mpv_module = None

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

IDX_TO_PIECE = {
    0: None,
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
}

PIECE_THEME = "classic"
BOARD_THEME = "green"

SAMPLE_INTERVAL = 0.1       # seconds between sampled frames
BATCH_SIZE = 64              # frames per GPU batch
STABILITY_FRAMES = 3
FORCE_INIT_SECONDS = 30.0
BOARD_SYNC_MS = 100          # how often to poll mpv and update the board
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing Utilities
# ═══════════════════════════════════════════════════════════════════════

if hasattr(Image, "Resampling"):
    _BICUBIC = Image.Resampling.BICUBIC
    _LANCZOS = Image.Resampling.LANCZOS
else:
    _BICUBIC = Image.BICUBIC
    _LANCZOS = Image.LANCZOS


class ResizeMax:
    def __init__(self, max_size=256):
        self.max_size = max_size

    def __call__(self, img):
        w, h = img.size
        s = self.max_size / max(w, h)
        return img.resize((int(w * s), int(h * s)), _BICUBIC)


class PadToSquare:
    def __init__(self, size=256, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        out = Image.new("RGB", (self.size, self.size), self.fill)
        out.paste(img, ((self.size - w) // 2, (self.size - h) // 2))
        return out


def visual_to_logical(v, black_pov):
    r, c = divmod(v, 8)
    if black_pov:
        return r * 8 + (7 - c)
    return (7 - r) * 8 + c


def indices_to_fen(idx):
    rows = []
    for r in range(7, -1, -1):
        s, empty = "", 0
        for f in range(8):
            p = IDX_TO_PIECE[int(idx[r * 8 + f])]
            if p is None:
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += p
        if empty:
            s += str(empty)
        rows.append(s)
    return "/".join(rows) + " w - - 0 1"


def fen_diff(a, b):
    return a.split()[0] != b.split()[0]


def is_starting_position(fen):
    p = fen.split()[0].split("/")
    return (len(p) == 8
            and p[0] == "rnbqkbnr" and p[1] == "pppppppp"
            and p[6] == "PPPPPPPP" and p[7] == "RNBQKBNR")


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def find_downloaded_video(directory):
    candidates = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if lower.endswith(".part"):
            continue
        if lower.endswith((".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# ═══════════════════════════════════════════════════════════════════════
# Board Detector — GroundingDINO (GLIP)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Game-State Tracker
# ═══════════════════════════════════════════════════════════════════════

class GameStateTracker:
    """
    Maintains an authoritative chess.Board by matching ChessNet predictions
    to legal moves, with debouncing to reject transient noise.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = None
        self.last_fen = None
        self.moves = []
        self._pending = None
        self._pending_n = 0
        self._first_ts = None
        self._game_start_ts = -999.0

    def update(self, predicted_fen, ts=0.0):
        if self._first_ts is None:
            self._first_ts = ts

        # --- debounce ---
        if self._pending:
            if not fen_diff(predicted_fen, self._pending):
                self._pending_n += 1
            else:
                self._pending = predicted_fen
                self._pending_n = 1
                return None
            if self._pending_n < STABILITY_FRAMES:
                return None
        else:
            if self.last_fen is not None and fen_diff(predicted_fen, self.last_fen):
                self._pending = predicted_fen
                self._pending_n = 1
                return None
            if self.last_fen is not None:
                return None

        target = self._pending or predicted_fen
        self._pending = None
        self._pending_n = 0

        # --- initialise ---
        if self.board is None:
            if is_starting_position(target):
                self.board = chess.Board()
                self.last_fen = target
                return {"type": "init", "fen": self.board.fen()}
            if self._first_ts is not None and ts - self._first_ts > FORCE_INIT_SECONDS:
                try:
                    self.board = chess.Board(target)
                    self.last_fen = target
                    return {"type": "init", "fen": self.board.fen(), "forced": True}
                except ValueError:
                    return None
            return None

        if not fen_diff(target, self.last_fen):
            return None

        # --- new game ---
        if is_starting_position(target) and ts - self._game_start_ts > 60:
            self.board = chess.Board()
            self.last_fen = target
            self.moves = []
            self._game_start_ts = ts
            return {"type": "init", "fen": self.board.fen(), "msg": "New game"}

        # --- exact legal-move match ---
        try:
            target_bf = chess.Board(target).board_fen()
        except ValueError:
            return None
        tmp = self.board.copy()
        for m in tmp.legal_moves:
            tmp.push(m)
            if tmp.board_fen() == target_bf:
                san = self.board.san(m)
                self.board.push(m)
                self.last_fen = target
                self.moves.append(san)
                return {"type": "move", "san": san, "fen": self.board.fen()}
            tmp.pop()

        # --- relaxed match (hamming <= 2) ---
        target_board = chess.Board(target)
        best_move, best_dist = None, 999
        for m in tmp.legal_moves:
            tmp.push(m)
            d = sum(
                1 for sq in chess.SQUARES
                if tmp.piece_at(sq) != target_board.piece_at(sq)
            )
            if d < best_dist:
                best_dist, best_move = d, m
            tmp.pop()

        if best_move and best_dist <= 2:
            san = self.board.san(best_move)
            self.board.push(best_move)
            self.last_fen = self.board.fen()
            self.moves.append(san)
            return {"type": "move", "san": san, "fen": self.board.fen(),
                    "relaxed": True}

        # --- multi-step BFS (depth <= 3) ---
        path = self._bfs(target_bf)
        if path:
            sans = []
            for fm in path:
                sans.append(self.board.san(fm))
                self.board.push(fm)
            self.last_fen = self.board.fen()
            self.moves.extend(sans)
            return {"type": "move", "san": " ".join(sans),
                    "fen": self.board.fen(), "multi": True}

        # --- correction (fallback) ---
        try:
            nb = chess.Board(target)
            nb.castling_rights = self.board.castling_rights
            self.board = nb
            self.last_fen = target
            return {"type": "correction", "fen": self.board.fen()}
        except ValueError:
            return None

    def _bfs(self, target_bf, depth=3):
        q = collections.deque([(self.board.copy(), [])])
        seen = {self.board.board_fen()}
        while q:
            cur, path = q.popleft()
            if len(path) >= depth:
                continue
            for m in cur.legal_moves:
                cur.push(m)
                bf = cur.board_fen()
                if bf == target_bf:
                    return path + [m]
                if bf not in seen and len(path) + 1 < depth:
                    seen.add(bf)
                    q.append((cur.copy(), path + [m]))
                cur.pop()
        return None

    @property
    def move_text(self):
        parts = []
        for i, s in enumerate(self.moves):
            if i % 2 == 0:
                parts.append(f"{i // 2 + 1}.")
            parts.append(s)
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Video Processor — batched GPU inference over entire video
# ═══════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """
    Processes an entire video through ChessNet using a pipelined
    frame-reader thread + batched GPU inference.
    """

    def __init__(self, model, transform, device, bbox):
        self.model = model
        self.transform = transform
        self.device = device
        self.bbox = bbox

    def process(self, video_path, interval=SAMPLE_INTERVAL,
                batch_size=BATCH_SIZE, progress_cb=None, cancel=None):
        """
        Process all frames.  Returns list of per-sample dicts.

        progress_cb(done, total) — called after each batch.
        cancel — threading.Event; set it to abort early.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Build target-frame list
        targets = []
        t = 0.0
        while t < duration:
            idx = int(t * fps)
            if idx < total_frames:
                targets.append((idx, t))
            t += interval

        total_samples = len(targets)
        if total_samples == 0:
            cap.release()
            return []

        # Pipeline: reader thread fills queue, main thread runs GPU inference
        batch_queue = queue.Queue(maxsize=4)
        reader = threading.Thread(
            target=self._read_frames,
            args=(cap, targets, batch_size, batch_queue, cancel),
            daemon=True,
        )
        reader.start()

        results = []
        while True:
            if cancel and cancel.is_set():
                break
            try:
                item = batch_queue.get(timeout=1.0)
            except queue.Empty:
                if not reader.is_alive():
                    break
                continue

            if item is None:   # sentinel
                break

            batch_results = self._infer_batch(*item)
            results.extend(batch_results)

            if progress_cb:
                progress_cb(len(results), total_samples)

        reader.join(timeout=5)
        return results

    # ── frame reader (runs in its own thread) ─────────────────────

    def _read_frames(self, cap, targets, batch_size, out_queue, cancel):
        frame_idx = 0
        target_ptr = 0
        batch_tensors = []
        batch_meta = []

        x1, y1, x2, y2 = self.bbox

        while target_ptr < len(targets):
            if cancel and cancel.is_set():
                break

            target_frame, target_ts = targets[target_ptr]

            # Skip to target (sequential grab is much faster than seek)
            while frame_idx < target_frame:
                cap.grab()
                frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            target_ptr += 1

            # Crop board region
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = self.transform(pil)

            batch_tensors.append(tensor)
            batch_meta.append((target_ts, target_frame))

            if len(batch_tensors) >= batch_size or target_ptr >= len(targets):
                out_queue.put((list(batch_tensors), list(batch_meta)))
                batch_tensors.clear()
                batch_meta.clear()

        cap.release()
        out_queue.put(None)

    # ── batched GPU inference ─────────────────────────────────────

    def _infer_batch(self, tensors, meta):
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            out_p, out_h, out_a, out_f = self.model(batch)

            flips = (torch.sigmoid(out_f).squeeze(-1) > 0.5).cpu().numpy()
            pieces_vis = torch.argmax(out_p, dim=2).cpu().numpy()
            h_labels = torch.argmax(
                torch.softmax(out_h, dim=2), dim=2,
            ).cpu().numpy()
            a_detect = (torch.sigmoid(out_a) > 0.5).cpu().numpy()

        results = []
        for i in range(len(tensors)):
            ts, frame_idx = meta[i]
            flip = bool(flips[i])

            # Pieces: visual -> logical
            log_pieces = np.zeros(64, dtype=np.int64)
            for v in range(64):
                log_pieces[visual_to_logical(v, flip)] = pieces_vis[i, v]
            fen = indices_to_fen(log_pieces)

            # Highlights: visual -> logical
            highlights = []
            for vis_sq in range(64):
                label = int(h_labels[i, vis_sq])
                if label > 0:
                    log_sq = visual_to_logical(vis_sq, flip)
                    highlights.append({
                        "sq": log_sq,
                        "color": "yellow" if label == 1 else "red",
                    })
            if len(highlights) > 16:
                highlights = []  # noise filter

            # Arrows: visual -> logical
            arrows = []
            for fr_vis, to_vis in np.argwhere(a_detect[i]):
                arrows.append([
                    visual_to_logical(int(fr_vis), flip),
                    visual_to_logical(int(to_vis), flip),
                ])
            if len(arrows) > 10:
                arrows = []

            results.append({
                "ts": round(ts, 3),
                "frame": frame_idx,
                "fen": fen,
                "flip": flip,
                "highlights": highlights,
                "arrows": arrows,
            })

        return results


# ═══════════════════════════════════════════════════════════════════════
# Timeline Building + Caching
# ═══════════════════════════════════════════════════════════════════════

def build_timeline(raw_results):
    """
    Run GameStateTracker over raw inference results.
    Each timeline entry includes raw predictions + accepted board state.
    """
    tracker = GameStateTracker()
    timeline = []

    for r in raw_results:
        tracker.update(r["fen"], r["ts"])

        entry = {
            "ts": r["ts"],
            "frame": r["frame"],
            "fen": r["fen"],
            "flip": r["flip"],
            "highlights": r["highlights"],
            "arrows": r["arrows"],
            "accepted_fen": tracker.board.fen() if tracker.board else None,
            "moves": list(tracker.moves),
        }
        timeline.append(entry)

    return timeline


def _cache_key(video_path, bbox):
    stat = os.stat(video_path)
    key = f"{os.path.basename(video_path)}_{stat.st_size}_{tuple(bbox)}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_cache_path(video_path, bbox):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_cache_key(video_path, bbox)}.json")


def save_cache(path, timeline, video_path, bbox, fps, duration):
    data = {
        "video": video_path,
        "bbox": list(bbox),
        "fps": fps,
        "duration": duration,
        "timeline": timeline,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_cache(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_timeline(timeline, frame_list, frame_idx):
    """Binary-search for the timeline entry at or just before *frame_idx*."""
    i = bisect.bisect_right(frame_list, frame_idx) - 1
    return timeline[max(0, i)]


# ═══════════════════════════════════════════════════════════════════════
# Main GUI  —  uses embedded mpv for hardware-accelerated video playback
# ═══════════════════════════════════════════════════════════════════════

class VideoAnalysisGUI:
    BRD_MAX = (480, 480)

    def __init__(self, model_path="model.pth", device="cuda"):
        self.root = tk.Tk()
        self.root.title("Chess Vision \u2014 Video Analysis")
        self.root.geometry("1200x820")
        self.root.minsize(1100, 750)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        # Video state
        self.video_path = None
        self.video_fps = 30.0
        self.video_duration = 0.0
        self.total_frames = 0
        self.board_bbox = None
        self._closed = False

        # mpv player (created when timeline is ready)
        self.player = None
        self._slider_lock = False
        self._resume_after_seek = False

        # Pre-processed timeline
        self.timeline = None
        self.timeline_frames = None
        self._last_rendered_entry = None
        self._last_board_photo = None
        self._cancel_processing = threading.Event()

        # Models
        self.model = None
        self.transform = None
        self.generator = None
        self.piece_assets = None
        self.board_asset = None
        self.detector = None

        self._build_ui()
        self._load_model(model_path)

    # ── Model loading ──────────────────────────────────────────────

    def _load_model(self, path):
        if not os.path.exists(path):
            self._status(f"{path} not found.")
            return

        self.model = ChessNet(num_classes=13, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            ResizeMax(256),
            PadToSquare(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.generator = ChessGenerator()
        pt = (PIECE_THEME if PIECE_THEME in self.generator.piece_images
              else sorted(self.generator.piece_images)[0])
        bt = (BOARD_THEME if BOARD_THEME in self.generator.board_images
              else sorted(self.generator.board_images)[0])
        self.piece_assets = self.generator.piece_images[pt]
        self.board_asset = self.generator.board_images[bt]

        mpv_ok = "mpv ready" if _mpv_module else "mpv NOT found"
        self._status(
            f"Model loaded ({self.device}, {mpv_ok}). Provide a video."
        )

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        c = ttk.Frame(self.root, padding=10)
        c.pack(fill="both", expand=True)

        ttk.Label(
            c, text="Chess Vision \u2014 Video Analysis",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            c,
            text=(
                "Paste a YouTube URL or open a local video file. "
                "The board is detected with GLIP, then all frames are "
                "batch-processed through ChessNet."
            ),
            wraplength=1000,
        ).pack(anchor="w", pady=(2, 10))

        # ── Input row ──
        inp = ttk.Frame(c)
        inp.pack(fill="x", pady=(0, 8))

        ttk.Label(inp, text="URL:").pack(side="left")
        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(inp, textvariable=self.url_var, width=55)
        url_entry.pack(side="left", padx=4)
        url_entry.bind("<Return>", lambda _: self._load_url())

        self.btn_go = ttk.Button(inp, text="Analyze", command=self._load_url)
        self.btn_go.pack(side="left", padx=2)

        self.btn_open = ttk.Button(
            inp, text="Open File", command=self._open_file,
        )
        self.btn_open.pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="Loading model\u2026")
        ttk.Label(inp, textvariable=self.status_var,
                  foreground="#555").pack(side="right")

        # ── Progress bar (hidden by default) ──
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            c, variable=self.progress_var, maximum=100,
        )

        # ── Display row ──
        disp = ttk.Frame(c)
        disp.pack(fill="both", expand=True)

        vf = ttk.LabelFrame(disp, text="Video", padding=6)
        vf.pack(side="left", fill="both", expand=True, padx=(0, 4))

        # Label for pre-playback state (detection preview, messages)
        self.vid_lbl = ttk.Label(vf, text="No video loaded", anchor="center")
        self.vid_lbl.pack(fill="both", expand=True)

        # Frame for mpv embedding (shown when playback starts)
        self._vid_frame = tk.Frame(vf, bg="black")

        bf = ttk.LabelFrame(disp, text="Detected Board State", padding=6)
        bf.pack(side="left", fill="y", padx=(4, 0))
        self.brd_lbl = ttk.Label(
            bf, text="Waiting for analysis\u2026", anchor="center",
        )
        self.brd_lbl.pack(fill="both", expand=True)

        # ── Transport controls ──
        ctrl = ttk.Frame(c)
        ctrl.pack(fill="x", pady=(8, 4))

        self.btn_play = ttk.Button(
            ctrl, text="\u25B6  Play",
            command=self._toggle_play, state="disabled",
        )
        self.btn_play.pack(side="left", padx=(0, 8))

        self.time_var = tk.StringVar(value="0:00 / 0:00")
        ttk.Label(ctrl, textvariable=self.time_var).pack(
            side="left", padx=(0, 8),
        )

        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(
            ctrl, from_=0, to=100, orient="horizontal",
            variable=self.slider_var, command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True)
        self.slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)

        # ── Game-state info ──
        info = ttk.LabelFrame(c, text="Game State", padding=8)
        info.pack(fill="x", pady=(4, 0))

        self.info_text = tk.Text(
            info, height=5, wrap="word", font=("Consolas", 10),
        )
        self.info_text.pack(fill="x")
        self.info_text.insert(
            "1.0", "Game state will appear here during analysis.",
        )
        self.info_text.config(state="disabled")

    # ── UI helpers ─────────────────────────────────────────────────

    def _status(self, msg):
        self.status_var.set(msg)

    def _set_img(self, label, pil_img, max_size):
        """One-off image display (detection preview)."""
        img = ImageOps.contain(pil_img, max_size, _LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo, text="")
        label.image = photo

    def _set_buttons(self, enabled):
        st = "normal" if enabled else "disabled"
        self.btn_go.config(state=st)
        self.btn_open.config(state=st)

    # ── mpv player management ──────────────────────────────────────

    def _create_player(self):
        """Embed an mpv player inside the video frame."""
        if _mpv_module is None:
            self._status(
                "python-mpv not installed \u2014 "
                "pip install python-mpv  and install mpv"
            )
            return False

        self._destroy_player()

        # Swap the label for the mpv host frame
        self.vid_lbl.pack_forget()
        self._vid_frame.pack(fill="both", expand=True)
        self.root.update_idletasks()

        wid = str(int(self._vid_frame.winfo_id()))
        self.player = _mpv_module.MPV(
            wid=wid,
            keep_open="yes",
            osc=False,
            input_default_bindings=False,
            input_vo_keyboard=False,
            cursor_autohide="no",
            # Let mpv pick the best VO (gpu / d3d11 / etc.)
        )
        self.player.play(self.video_path)
        self.player.wait_until_playing()
        self.player.pause = True
        return True

    def _destroy_player(self):
        if self.player is not None:
            try:
                self.player.terminate()
            except Exception:
                pass
            self.player = None
        # Show the label again
        self._vid_frame.pack_forget()
        self.vid_lbl.pack(fill="both", expand=True)

    # ── Board rendering from timeline entry ────────────────────────

    def _show_board_at_frame(self, frame_idx):
        if not self.timeline:
            return
        entry = lookup_timeline(
            self.timeline, self.timeline_frames, frame_idx,
        )
        if entry is self._last_rendered_entry:
            return
        self._last_rendered_entry = entry
        self._render_board_entry(entry)
        self._update_info_entry(entry)

    def _render_board_entry(self, entry):
        flip = entry["flip"]

        if entry["accepted_fen"]:
            board = chess.Board(entry["accepted_fen"])
        else:
            board = chess.Board(entry["fen"])

        highlight_specs = []
        for hl in entry["highlights"]:
            color = (255, 255, 0) if hl["color"] == "yellow" else (255, 0, 0)
            highlight_specs.append((hl["sq"], color))

        img = self.generator.render_board(
            board,
            self.piece_assets,
            self.board_asset,
            flipped=flip,
            highlights=highlight_specs,
        )
        img = augmentations.draw_coordinates(img, flipped=flip)

        for start, end in entry["arrows"]:
            img = augmentations.draw_arrow(
                img, start, end, color=(255, 170, 0), flipped=flip,
            )

        contained = ImageOps.contain(img, self.BRD_MAX, _LANCZOS)
        photo = ImageTk.PhotoImage(contained)
        self.brd_lbl.configure(image=photo, text="")
        self.brd_lbl.image = photo
        self._last_board_photo = photo

    def _update_info_entry(self, entry):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", tk.END)

        if entry["accepted_fen"]:
            pov = "Black at bottom" if entry["flip"] else "White at bottom"
            moves = entry["moves"]
            parts = []
            for i, san in enumerate(moves):
                if i % 2 == 0:
                    parts.append(f"{i // 2 + 1}.")
                parts.append(san)
            move_text = " ".join(parts)

            txt = f"FEN:  {entry['accepted_fen']}\n"
            txt += f"POV:  {pov}\n"
            txt += f"Moves: {move_text}" if move_text else "Moves: (waiting)"
        else:
            txt = "Waiting for starting position\u2026"

        self.info_text.insert("1.0", txt)
        self.info_text.config(state="disabled")

    # ── Board sync loop (polls mpv at ~10 Hz) ─────────────────────

    def _sync_board(self):
        """Periodically read mpv's position and update board + transport."""
        if self._closed:
            return

        if self.player and self.timeline:
            try:
                pos = self.player.time_pos
                paused = self.player.pause
                eof = self.player.eof_reached
            except Exception:
                pos, paused, eof = None, True, False

            if eof and not paused:
                self.player.pause = True
                self.btn_play.config(text="\u25B6  Play")

            if pos is not None and pos >= 0:
                frame_idx = int(pos * self.video_fps)
                self._show_board_at_frame(frame_idx)

                # Time display
                self.time_var.set(
                    f"{fmt_time(pos)} / {fmt_time(self.video_duration)}"
                )
                # Slider
                if not self._slider_lock:
                    self._slider_lock = True
                    self.slider_var.set(
                        pos / max(self.video_duration, 0.01) * 100,
                    )
                    self._slider_lock = False

        self.root.after(BOARD_SYNC_MS, self._sync_board)

    # ── Video input ────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.webm *.avi *.mov"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._begin_analysis(path)

    def _load_url(self):
        url = self.url_var.get().strip()
        if not url:
            return
        self._status("Downloading video\u2026")
        self._set_buttons(False)
        threading.Thread(
            target=self._download_video, args=(url,), daemon=True,
        ).start()

    def _download_video(self, url):
        try:
            tmp = tempfile.mkdtemp(prefix="chessvision_")
            out_tpl = os.path.join(tmp, "video.%(ext)s")
            fmt = (
                "bestvideo[height<=720][ext=mp4]"
                "+bestaudio[ext=m4a]/best[height<=720]"
            )
            result = subprocess.run(
                ["yt-dlp", "-f", fmt, "--merge-output-format", "mp4",
                 "-o", out_tpl, url],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                result = subprocess.run(
                    ["yt-dlp", "-f",
                     "best[height<=720][ext=mp4]/best[height<=720]",
                     "-o", out_tpl, url],
                    capture_output=True, text=True, timeout=600,
                )
            if result.returncode != 0:
                err = result.stderr[:200]
                self.root.after(
                    0, lambda: self._status(f"Download failed: {err}"),
                )
                self.root.after(0, lambda: self._set_buttons(True))
                return

            video_path = find_downloaded_video(tmp)
            if video_path is None:
                self.root.after(
                    0, lambda: self._status("Download produced no video file."),
                )
                self.root.after(0, lambda: self._set_buttons(True))
                return

            self.root.after(
                0, lambda p=video_path: self._begin_analysis(p),
            )

        except FileNotFoundError:
            self.root.after(
                0, lambda: self._status(
                    "yt-dlp not found \u2014 pip install yt-dlp"
                ),
            )
            self.root.after(0, lambda: self._set_buttons(True))
        except Exception as exc:
            self.root.after(
                0, lambda: self._status(f"Download error: {exc}"),
            )
            self.root.after(0, lambda: self._set_buttons(True))

    # ── Analysis pipeline ──────────────────────────────────────────

    def _begin_analysis(self, video_path):
        self._cancel_processing.set()
        self._destroy_player()

        self.video_path = video_path
        self.timeline = None
        self.timeline_frames = None
        self._last_rendered_entry = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._status(f"Failed to open: {os.path.basename(video_path)}")
            self._set_buttons(True)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 0 else 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.video_fps
        cap.release()

        self._status(
            f"Video loaded ({fmt_time(self.video_duration)}). "
            "Detecting board\u2026"
        )
        self._set_buttons(False)
        self.btn_play.config(state="disabled")

        threading.Thread(target=self._detect_board, daemon=True).start()

    def _detect_board(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid = int(total * 0.4)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.root.after(
                    0, lambda: self._status("Could not read video frame."),
                )
                self.root.after(0, lambda: self._set_buttons(True))
                return

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if self.detector is None:
                self.detector = BoardDetector(device=str(self.device))

            self.root.after(
                0, lambda: self._status("Running GLIP board detection\u2026"),
            )
            bbox = self.detector.detect(pil)
            self.detector.unload()

            if bbox is None:
                self.root.after(
                    0, lambda: self._status("No chess board detected."),
                )
                self.root.after(0, lambda: self._set_buttons(True))
                return

            self.board_bbox = bbox

            draw = ImageDraw.Draw(pil)
            draw.rectangle(bbox, outline="lime", width=3)
            self.root.after(
                0, lambda img=pil: self._set_img(
                    self.vid_lbl, img, (640, 480),
                ),
            )

            self.root.after(0, self._on_detection_done)

        except ImportError:
            self.root.after(
                0, lambda: self._status(
                    "pip install transformers  (needed for board detection)"
                ),
            )
            self.root.after(0, lambda: self._set_buttons(True))
        except Exception as exc:
            self.root.after(
                0, lambda: self._status(f"Detection error: {exc}"),
            )
            self.root.after(0, lambda: self._set_buttons(True))

    def _on_detection_done(self):
        # Check cache
        try:
            cache_path = get_cache_path(self.video_path, self.board_bbox)
            if os.path.exists(cache_path):
                cached = load_cache(cache_path)
                self._finish_with_timeline(cached["timeline"])
                self._status(
                    f"Loaded {len(self.timeline)} cached frames. Press Play."
                )
                return
        except Exception:
            pass

        self._status("Processing video\u2026")
        self.progress_var.set(0)
        self.progress_bar.pack(fill="x", pady=(4, 0))
        self._cancel_processing.clear()

        threading.Thread(
            target=self._process_video, daemon=True,
        ).start()

    def _process_video(self):
        t0 = time.perf_counter()

        processor = VideoProcessor(
            self.model, self.transform, self.device, self.board_bbox,
        )

        def on_progress(done, total):
            pct = done / max(total, 1) * 100
            self.root.after(0, lambda: self._update_progress(done, total, pct))

        raw = processor.process(
            self.video_path,
            interval=SAMPLE_INTERVAL,
            batch_size=BATCH_SIZE,
            progress_cb=on_progress,
            cancel=self._cancel_processing,
        )

        if self._cancel_processing.is_set():
            return

        timeline = build_timeline(raw)
        elapsed = time.perf_counter() - t0

        try:
            cache_path = get_cache_path(self.video_path, self.board_bbox)
            save_cache(
                cache_path, timeline, self.video_path,
                self.board_bbox, self.video_fps, self.video_duration,
            )
        except Exception:
            pass

        self.root.after(
            0, lambda: self._on_processing_done(timeline, elapsed),
        )

    def _update_progress(self, done, total, pct):
        self.progress_var.set(pct)
        self._status(f"Processing: {done}/{total} frames ({pct:.0f}%)")

    def _on_processing_done(self, timeline, elapsed):
        self.progress_bar.pack_forget()
        self._finish_with_timeline(timeline)
        self._status(
            f"Processed {len(timeline)} frames in {elapsed:.1f}s. Press Play."
        )

    def _finish_with_timeline(self, timeline):
        self.timeline = timeline
        self.timeline_frames = [e["frame"] for e in timeline]
        self._last_rendered_entry = None

        if not self._create_player():
            return

        # Show initial board
        self._show_board_at_frame(0)
        self._sync_board()  # start polling loop

        self.btn_play.config(state="normal")
        self._set_buttons(True)

    # ── Playback (delegates to mpv) ────────────────────────────────

    def _toggle_play(self):
        if not self.player:
            return
        try:
            self.player.pause = not self.player.pause
        except Exception:
            return
        if self.player.pause:
            self.btn_play.config(text="\u25B6  Play")
        else:
            self.btn_play.config(text="\u23F8  Pause")

    # ── Seek (delegates to mpv) ────────────────────────────────────

    def _on_slider(self, _val):
        if self._slider_lock or not self.player:
            return
        frac = self.slider_var.get() / 100
        target_sec = frac * self.video_duration
        try:
            self.player.seek(target_sec, "absolute")
        except Exception:
            pass

    def _on_slider_press(self, _event):
        if self.player:
            try:
                self._resume_after_seek = not self.player.pause
                self.player.pause = True
            except Exception:
                self._resume_after_seek = False

    def _on_slider_release(self, _event):
        if not self.player:
            return
        frac = self.slider_var.get() / 100
        target_sec = frac * self.video_duration
        try:
            self.player.seek(target_sec, "absolute")
        except Exception:
            pass
        if self._resume_after_seek:
            self._resume_after_seek = False
            try:
                self.player.pause = False
            except Exception:
                pass

    # ── Lifecycle ──────────────────────────────────────────────────

    def _on_close(self):
        self._closed = True
        self._cancel_processing.set()
        self._destroy_player()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Chess Vision \u2014 Video Analysis GUI",
    )
    parser.add_argument(
        "--model", default="model.pth",
        help="Path to ChessNet model checkpoint",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device (cuda / cpu)",
    )
    parser.add_argument(
        "--video", default=None,
        help="Auto-load a video file on startup",
    )
    args = parser.parse_args()

    gui = VideoAnalysisGUI(model_path=args.model, device=args.device)

    if args.video:
        gui.root.after(100, lambda: gui._begin_analysis(args.video))

    gui.run()


if __name__ == "__main__":
    main()
