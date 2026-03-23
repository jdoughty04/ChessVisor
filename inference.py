import argparse
import csv
import json
import os
import threading
from pathlib import Path

import chess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None

import augmentations
from board_detection import BoardDetector, crop_board
from generate_data import ChessGenerator
from model import ChessNet


DEFAULT_PIECE_THEME = "classic"
DEFAULT_BOARD_THEME = "green"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

if hasattr(Image, "Resampling"):
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_LANCZOS = Image.LANCZOS


class ResizeMax:
    """Resize the image so the largest side is max_size, preserving aspect ratio."""

    def __init__(self, max_size=256):
        self.max_size = max_size

    def __call__(self, img):
        w, h = img.size
        scale = self.max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), RESAMPLE_BICUBIC)


class PadToSquare:
    """Pad the image to be a square of side `size` with `fill` color."""

    def __init__(self, size=256, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        new_img = Image.new("RGB", (self.size, self.size), self.fill)
        new_img.paste(img, ((self.size - w) // 2, (self.size - h) // 2))
        return new_img


class ChessInference:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = ChessNet(num_classes=13, pretrained=False).to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        except Exception as exc:
            print(f"Error loading model: {exc}")
            raise

        self.model.eval()

        self.transform = transforms.Compose(
            [
                ResizeMax(256),
                PadToSquare(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.idx_to_piece = {
            0: None,
            1: "P",
            2: "N",
            3: "B",
            4: "R",
            5: "Q",
            6: "K",
            7: "p",
            8: "n",
            9: "b",
            10: "r",
            11: "q",
            12: "k",
        }

        self.generator = None
        self.default_piece_theme = None
        self.default_board_theme = None
        self.board_detector = BoardDetector(device=str(self.device))

    def _ensure_generator(self):
        if self.generator is not None:
            return

        self.generator = ChessGenerator()
        self.default_piece_theme = self._resolve_theme(
            DEFAULT_PIECE_THEME, self.generator.piece_images
        )
        self.default_board_theme = self._resolve_theme(
            DEFAULT_BOARD_THEME, self.generator.board_images
        )

    @staticmethod
    def _resolve_theme(preferred_theme, collection):
        if preferred_theme in collection:
            return preferred_theme
        if not collection:
            raise RuntimeError("No rendering assets are available.")
        return sorted(collection.keys())[0]

    def _get_render_assets(self, piece_theme=None, board_theme=None):
        self._ensure_generator()

        piece_theme = self._resolve_theme(
            piece_theme or self.default_piece_theme, self.generator.piece_images
        )
        board_theme = self._resolve_theme(
            board_theme or self.default_board_theme, self.generator.board_images
        )

        return (
            self.generator.piece_images[piece_theme],
            self.generator.board_images[board_theme],
        )

    def prepare_image(self, image, detect_board=True):
        prepared = image.convert("RGB")
        metadata = {
            "board_detection_requested": bool(detect_board),
            "board_detection_status": "disabled",
            "board_bbox": None,
            "board_detection_error": None,
        }

        if not detect_board:
            return prepared, metadata

        metadata["board_detection_status"] = "not_found"

        try:
            bbox = self.board_detector.detect(prepared)
        except ImportError:
            metadata["board_detection_status"] = "unavailable"
            metadata["board_detection_error"] = (
                "GroundingDINO board detection is unavailable; install transformers "
                "and allow the model download on first use."
            )
        except Exception as exc:
            metadata["board_detection_status"] = "error"
            metadata["board_detection_error"] = str(exc)
        else:
            if bbox is not None:
                metadata["board_bbox"] = bbox
                metadata["board_detection_status"] = "detected"
                prepared = crop_board(prepared, bbox)
        finally:
            self.board_detector.unload()

        return prepared, metadata

    @staticmethod
    def _to_logical_index(visual_idx, flipped):
        r_vis = visual_idx // 8
        c_vis = visual_idx % 8

        if flipped:
            rank = r_vis
            file = 7 - c_vis
        else:
            rank = 7 - r_vis
            file = c_vis

        return rank * 8 + file

    def predict(self, image_path, detect_board=True):
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            print(f"Could not load image {image_path}: {exc}")
            return None

        return self.predict_image(image, detect_board=detect_board)

    def predict_image(self, image, detect_board=True):
        prepared_image, prep_metadata = self.prepare_image(
            image,
            detect_board=detect_board,
        )
        input_tensor = self.transform(prepared_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_p, out_h, out_a, out_f = self.model(input_tensor)

            prob_f = torch.sigmoid(out_f).item()
            is_flipped = prob_f > 0.5

            probs_p_visual = torch.softmax(out_p, dim=2).squeeze(0).cpu().numpy()
            preds_p_visual = np.argmax(probs_p_visual, axis=1)
            conf_p_visual = np.max(probs_p_visual, axis=1)

            preds_p = np.zeros(64, dtype=np.int64)
            conf_p = np.zeros(64, dtype=np.float32)
            for vis_sq in range(64):
                log_sq = self._to_logical_index(vis_sq, is_flipped)
                preds_p[log_sq] = preds_p_visual[vis_sq]
                conf_p[log_sq] = conf_p_visual[vis_sq]

            probs_h_visual = torch.softmax(out_h, dim=2).squeeze(0).cpu().numpy()
            preds_h_visual = np.argmax(probs_h_visual, axis=1)
            conf_h_visual = np.max(probs_h_visual, axis=1)

            preds_h = np.zeros(64, dtype=np.int64)
            conf_h = np.zeros(64, dtype=np.float32)
            for vis_sq in range(64):
                log_sq = self._to_logical_index(vis_sq, is_flipped)
                preds_h[log_sq] = preds_h_visual[vis_sq]
                conf_h[log_sq] = conf_h_visual[vis_sq]

            probs_a = torch.sigmoid(out_a).squeeze(0).cpu().numpy()
            threshold = 0.5
            rows, cols = np.where(probs_a > threshold)

            arrows = []
            for start_vis, end_vis in zip(rows, cols):
                start_log = self._to_logical_index(start_vis, is_flipped)
                end_log = self._to_logical_index(end_vis, is_flipped)
                arrows.append((int(start_log), int(end_log)))

            result = {
                "pieces": preds_p,
                "pieces_confidence": conf_p,
                "highlights": preds_h,
                "highlights_confidence": conf_h,
                "arrows": arrows,
                "arrows_probs": probs_a,
                "flipped": is_flipped,
                "confidence_flipped": prob_f,
            }
            result.update(prep_metadata)
            result["fen"] = self.pieces_to_fen(result["pieces"])
            return result

    @staticmethod
    def format_board_detection(result):
        status = result.get("board_detection_status", "disabled")
        bbox = result.get("board_bbox")
        error = result.get("board_detection_error")

        if status == "detected" and bbox is not None:
            return f"Board detection: detected board at {bbox}"
        if status == "not_found":
            return "Board detection: no board found, used the full image"
        if status == "unavailable":
            return "Board detection: unavailable, used the full image"
        if status == "error":
            suffix = f" ({error})" if error else ""
            return f"Board detection: failed, used the full image{suffix}"
        return "Board detection: disabled"

    @staticmethod
    def annotate_source_image(image, result):
        preview = image.convert("RGB").copy()
        bbox = result.get("board_bbox")
        if bbox is None:
            return preview

        draw = ImageDraw.Draw(preview)
        line_width = max(2, int(round(max(preview.size) / 220)))
        draw.rectangle(bbox, outline=(64, 220, 96), width=line_width)
        return preview

    def pieces_to_fen(self, pieces):
        ranks = []
        for rank in range(7, -1, -1):
            empties = 0
            rank_tokens = []
            for file in range(8):
                piece_symbol = self.idx_to_piece.get(int(pieces[rank * 8 + file]))
                if piece_symbol is None:
                    empties += 1
                    continue

                if empties:
                    rank_tokens.append(str(empties))
                    empties = 0
                rank_tokens.append(piece_symbol)

            if empties:
                rank_tokens.append(str(empties))

            ranks.append("".join(rank_tokens) or "8")

        return "/".join(ranks)

    def summarize_result(self, result):
        highlight_labels = {1: "generic", 2: "red"}
        highlighted = [
            f"{chess.square_name(idx)} ({highlight_labels.get(int(label), 'unknown')})"
            for idx, label in enumerate(result["highlights"])
            if int(label) > 0
        ]
        arrows = [
            f"{chess.square_name(start)} -> {chess.square_name(end)}"
            for start, end in result["arrows"]
        ]

        perspective = "black at bottom" if result["flipped"] else "white at bottom"

        lines = [
            self.format_board_detection(result),
            f"FEN: {result['fen']}",
            (
                "Perspective: "
                f"{perspective} ({result['confidence_flipped']:.2%} black-perspective confidence)"
            ),
            (
                "Piece confidence: "
                f"mean {result['pieces_confidence'].mean():.3f}, "
                f"min {result['pieces_confidence'].min():.3f}, "
                f"max {result['pieces_confidence'].max():.3f}"
            ),
            (
                f"Highlights ({len(highlighted)}): "
                + (", ".join(highlighted[:8]) if highlighted else "none")
            ),
            (
                f"Arrows ({len(arrows)}): "
                + (", ".join(arrows[:6]) if arrows else "none")
            ),
        ]

        if len(highlighted) > 8:
            lines.append(f"More highlights: {len(highlighted) - 8}")
        if len(arrows) > 6:
            lines.append(f"More arrows: {len(arrows) - 6}")

        return "\n".join(lines)

    def visualize_to_image(self, result, piece_theme=None, board_theme=None):
        pieces_assets, board_asset = self._get_render_assets(
            piece_theme=piece_theme,
            board_theme=board_theme,
        )

        board = chess.Board(None)
        for idx, p_idx in enumerate(result["pieces"]):
            symbol = self.idx_to_piece.get(int(p_idx))
            if symbol:
                board.set_piece_at(idx, chess.Piece.from_symbol(symbol))

        flipped = result["flipped"]
        print(
            "Predicted Perspective: "
            f"{'Black' if flipped else 'White'} "
            f"(Prob: {result['confidence_flipped']:.2f})"
        )

        highlight_specs = []
        for sq_idx, label in enumerate(result["highlights"]):
            if label == 1:
                highlight_specs.append((sq_idx, (255, 255, 0)))
            elif label == 2:
                highlight_specs.append((sq_idx, (255, 0, 0)))

        img = self.generator.render_board(
            board,
            pieces_assets,
            board_asset,
            flipped=flipped,
            highlights=highlight_specs,
        )
        img = augmentations.draw_coordinates(img, flipped=flipped)

        for start, end in result["arrows"]:
            img = augmentations.draw_arrow(
                img,
                start,
                end,
                color=(255, 170, 0),
                flipped=flipped,
            )

        return img

    def visualize(self, result, output_path, piece_theme=None, board_theme=None):
        output_image = self.visualize_to_image(
            result,
            piece_theme=piece_theme,
            board_theme=board_theme,
        )
        output_image.save(output_path)
        print(f"Prediction visualization saved to {output_path}")


class ChessInferenceGUI:
    PREVIEW_SIZE = (460, 460)

    def __init__(self, inference, piece_theme=None, board_theme=None):
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
            from PIL import ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "Tkinter is required for GUI mode but is not available in this Python build."
            ) from exc

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ImageTk = ImageTk

        self.inference = inference
        self.piece_theme = piece_theme
        self.board_theme = board_theme
        self.drag_and_drop_enabled = TkinterDnD is not None and DND_FILES is not None

        self.root = TkinterDnD.Tk() if self.drag_and_drop_enabled else tk.Tk()
        self.root.title("Chess Vision Inference")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 720)

        self.current_source_image = None
        self.current_prediction_image = None
        self.current_result = None
        self.current_source_name = None
        self.is_busy = False

        self.status_var = tk.StringVar(value="Load a screenshot or board image to begin.")
        self._build_layout()

    def _build_layout(self):
        style = self.ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        container = self.ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        title = self.ttk.Label(
            container,
            text="Chess Vision Demo",
            font=("Segoe UI", 20, "bold"),
        )
        title.pack(anchor="w")

        subtitle_text = (
            "Drop a screenshot anywhere in this window, paste from the clipboard with "
            "Ctrl+V, or browse for a file. Full screenshots are auto-cropped to the "
            "detected board before prediction."
        )
        if not self.drag_and_drop_enabled:
            subtitle_text += " Drag-and-drop activates automatically after installing tkinterdnd2."

        subtitle = self.ttk.Label(
            container,
            text=subtitle_text,
            wraplength=1000,
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(6, 16))

        controls = self.ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 12))

        self.open_button = self.ttk.Button(
            controls,
            text="Open Image",
            command=self.open_image_dialog,
        )
        self.open_button.pack(side="left", padx=(0, 8))

        self.paste_button = self.ttk.Button(
            controls,
            text="Paste Clipboard",
            command=self.paste_from_clipboard,
        )
        self.paste_button.pack(side="left", padx=(0, 8))

        self.save_button = self.ttk.Button(
            controls,
            text="Save Prediction",
            command=self.save_prediction,
            state="disabled",
        )
        self.save_button.pack(side="left", padx=(0, 8))

        self.clear_button = self.ttk.Button(
            controls,
            text="Clear",
            command=self.clear_images,
        )
        self.clear_button.pack(side="left")

        status = self.ttk.Label(
            controls,
            textvariable=self.status_var,
            foreground="#444444",
        )
        status.pack(side="right")

        image_row = self.ttk.Frame(container)
        image_row.pack(fill="both", expand=True)

        input_frame = self.ttk.LabelFrame(image_row, text="Input Image", padding=12)
        input_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        output_frame = self.ttk.LabelFrame(image_row, text="Predicted Board", padding=12)
        output_frame.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.input_preview = self.ttk.Label(
            input_frame,
            text="Input preview will appear here",
            anchor="center",
            justify="center",
        )
        self.input_preview.pack(fill="both", expand=True)

        self.output_preview = self.ttk.Label(
            output_frame,
            text="Prediction preview will appear here",
            anchor="center",
            justify="center",
        )
        self.output_preview.pack(fill="both", expand=True)

        summary_frame = self.ttk.LabelFrame(container, text="Prediction Details", padding=12)
        summary_frame.pack(fill="x", pady=(12, 0))

        self.summary_text = self.tk.Text(
            summary_frame,
            height=8,
            wrap="word",
            font=("Consolas", 10),
        )
        self.summary_text.pack(fill="x")
        self.summary_text.insert(
            "1.0",
            "Prediction details will appear here after you load an image.",
        )
        self.summary_text.config(state="disabled")

        self.root.bind("<Control-v>", self._handle_paste_shortcut)
        self.root.bind("<Control-V>", self._handle_paste_shortcut)

        if self.drag_and_drop_enabled:
            for widget in (
                self.root,
                self.input_preview,
                self.output_preview,
                input_frame,
                output_frame,
            ):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._handle_drop)

    def _handle_paste_shortcut(self, _event):
        self.paste_from_clipboard()
        return "break"

    def _handle_drop(self, event):
        candidates = self.root.tk.splitlist(event.data)
        for candidate in candidates:
            path = Path(candidate)
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
                self.load_image_from_path(path)
                return

        self.set_status("Drop a supported image file.")

    def set_status(self, message):
        self.status_var.set(message)

    def set_busy(self, busy):
        self.is_busy = busy
        button_state = "disabled" if busy else "normal"
        for button in (self.open_button, self.paste_button, self.clear_button):
            button.config(state=button_state)
        self.save_button.config(
            state="disabled" if busy or self.current_prediction_image is None else "normal"
        )

    def update_summary(self, text):
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", self.tk.END)
        self.summary_text.insert("1.0", text)
        self.summary_text.config(state="disabled")

    def set_preview_image(self, label, image):
        preview = Image.new("RGB", self.PREVIEW_SIZE, (246, 246, 246))
        contained = ImageOps.contain(image, self.PREVIEW_SIZE, RESAMPLE_LANCZOS)
        offset_x = (self.PREVIEW_SIZE[0] - contained.width) // 2
        offset_y = (self.PREVIEW_SIZE[1] - contained.height) // 2
        preview.paste(contained, (offset_x, offset_y))

        photo = self.ImageTk.PhotoImage(preview)
        label.configure(image=photo, text="")
        label.image = photo

    def open_image_dialog(self):
        file_path = self.filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.load_image_from_path(Path(file_path))

    def load_image_from_path(self, path):
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            self.messagebox.showerror("Open image failed", str(exc))
            return

        self.start_prediction(image, str(path))

    def paste_from_clipboard(self):
        if ImageGrab is None:
            self.messagebox.showerror(
                "Clipboard unavailable",
                "Pillow ImageGrab is not available in this environment.",
            )
            return

        try:
            clipboard = ImageGrab.grabclipboard()
        except Exception as exc:
            self.messagebox.showerror("Clipboard unavailable", str(exc))
            return

        if clipboard is None:
            self.set_status("Clipboard does not currently contain an image.")
            return

        if isinstance(clipboard, list):
            image_paths = [
                Path(item)
                for item in clipboard
                if Path(item).suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not image_paths:
                self.set_status("Clipboard contains files, but none of them are supported images.")
                return
            self.load_image_from_path(image_paths[0])
            return

        if isinstance(clipboard, Image.Image):
            self.start_prediction(clipboard.convert("RGB"), "clipboard image")
            return

        self.set_status("Clipboard content is not a supported image.")

    def start_prediction(self, image, source_name):
        if self.is_busy:
            return

        self.current_source_image = image.copy()
        self.current_source_name = source_name
        self.current_prediction_image = None
        self.current_result = None

        self.set_preview_image(self.input_preview, self.current_source_image)
        self.output_preview.configure(image="", text="Running inference...")
        self.output_preview.image = None
        self.update_summary("Running board detection and inference...")
        self.set_status(f"Running board detection and inference on {source_name}...")
        self.set_busy(True)

        worker = threading.Thread(
            target=self._predict_in_background,
            args=(image.copy(), source_name),
            daemon=True,
        )
        worker.start()

    def _predict_in_background(self, image, source_name):
        try:
            result = self.inference.predict_image(image)
            source_preview = self.inference.annotate_source_image(image, result)
            prediction_image = self.inference.visualize_to_image(
                result,
                piece_theme=self.piece_theme,
                board_theme=self.board_theme,
            )
            summary = self.inference.summarize_result(result)
        except Exception as exc:
            self.root.after(0, lambda: self._prediction_failed(str(exc)))
            return

        self.root.after(
            0,
            lambda: self._prediction_complete(
                source_name,
                result,
                source_preview,
                prediction_image,
                summary,
            ),
        )

    def _prediction_complete(
        self,
        source_name,
        result,
        source_preview,
        prediction_image,
        summary,
    ):
        self.current_result = result
        self.current_prediction_image = prediction_image
        self.set_preview_image(self.input_preview, source_preview)
        self.set_preview_image(self.output_preview, prediction_image)
        self.update_summary(summary)
        self.set_status(f"Prediction ready for {source_name}.")
        self.set_busy(False)

    def _prediction_failed(self, error_message):
        self.output_preview.configure(image="", text="Prediction failed")
        self.output_preview.image = None
        self.update_summary(error_message)
        self.set_status("Prediction failed.")
        self.set_busy(False)

    def save_prediction(self):
        if self.current_prediction_image is None:
            return

        source_stem = "prediction"
        if self.current_source_name:
            source_stem = Path(self.current_source_name).stem or source_stem

        output_path = self.filedialog.asksaveasfilename(
            title="Save predicted board",
            defaultextension=".png",
            initialfile=f"{source_stem}_prediction.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not output_path:
            return

        self.current_prediction_image.save(output_path)
        self.set_status(f"Saved prediction to {output_path}")

    def clear_images(self):
        if self.is_busy:
            return

        self.current_source_image = None
        self.current_prediction_image = None
        self.current_result = None
        self.current_source_name = None

        self.input_preview.configure(image="", text="Input preview will appear here")
        self.input_preview.image = None
        self.output_preview.configure(image="", text="Prediction preview will appear here")
        self.output_preview.image = None
        self.update_summary("Prediction details will appear here after you load an image.")
        self.set_status("Load a screenshot or board image to begin.")
        self.set_busy(False)

    def run(self):
        self.root.mainloop()


def load_labels(labels_path):
    """Load labels from a CSV file and return as a dict keyed by filename."""

    labels = {}
    with open(labels_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row["filename"]
            fen = row["fen"]

            highlights_str = row["highlights"]
            if highlights_str and highlights_str != "{}":
                highlights = json.loads(highlights_str.replace('""', '"'))
                highlights = {int(k): v for k, v in highlights.items()}
            else:
                highlights = {}

            arrows_str = row["arrows"]
            if arrows_str and arrows_str != "[]":
                arrows = json.loads(arrows_str)
                arrows = [tuple(arrow) for arrow in arrows]
            else:
                arrows = []

            is_flipped = row["is_flipped"].lower() == "true"

            labels[filename] = {
                "fen": fen,
                "highlights": highlights,
                "arrows": arrows,
                "is_flipped": is_flipped,
            }

    return labels


def fen_to_piece_indices(fen):
    """Convert a FEN position (piece placement only) to 64-element piece index array."""

    piece_to_idx = {
        "P": 1,
        "N": 2,
        "B": 3,
        "R": 4,
        "Q": 5,
        "K": 6,
        "p": 7,
        "n": 8,
        "b": 9,
        "r": 10,
        "q": 11,
        "k": 12,
    }

    pieces = np.zeros(64, dtype=np.int64)
    ranks = fen.split("/")

    idx = 0
    for rank in ranks:
        for char in rank:
            if char.isdigit():
                idx += int(char)
            else:
                rank_idx = 7 - (idx // 8)
                file_idx = idx % 8
                square = rank_idx * 8 + file_idx
                pieces[square] = piece_to_idx.get(char, 0)
                idx += 1

    return pieces


def compute_statistics(all_results):
    """Compute and print performance statistics from all results."""

    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)

    total_squares = 0
    correct_squares = 0

    highlight_tp = 0
    highlight_fp = 0
    highlight_fn = 0

    red_tp = 0
    red_fp = 0
    red_fn = 0

    arrow_tp = 0
    arrow_fp = 0
    arrow_fn = 0

    flip_correct = 0
    flip_total = 0

    for result in all_results:
        pred = result["prediction"]
        gt = result["ground_truth"]

        gt_pieces = fen_to_piece_indices(gt["fen"])
        pred_pieces = pred["pieces"]
        total_squares += 64
        correct_squares += np.sum(gt_pieces == pred_pieces)

        gt_highlights = gt["highlights"]
        pred_highlights = pred["highlights"]

        for sq in range(64):
            gt_hl = gt_highlights.get(sq, 0)
            pred_hl = pred_highlights[sq]

            gt_has = gt_hl > 0
            pred_has = pred_hl > 0

            if gt_has and pred_has:
                highlight_tp += 1
            elif pred_has and not gt_has:
                highlight_fp += 1
            elif gt_has and not pred_has:
                highlight_fn += 1

            gt_red = gt_hl == 2
            pred_red = pred_hl == 2

            if gt_red and pred_red:
                red_tp += 1
            elif pred_red and not gt_red:
                red_fp += 1
            elif gt_red and not pred_red:
                red_fn += 1

        gt_arrows = set(gt["arrows"])
        pred_arrows = set(pred["arrows"])

        arrow_tp += len(gt_arrows & pred_arrows)
        arrow_fp += len(pred_arrows - gt_arrows)
        arrow_fn += len(gt_arrows - pred_arrows)

        flip_total += 1
        if pred["flipped"] == gt["is_flipped"]:
            flip_correct += 1

    piece_accuracy = correct_squares / total_squares if total_squares > 0 else 0

    highlight_precision = (
        highlight_tp / (highlight_tp + highlight_fp)
        if (highlight_tp + highlight_fp) > 0
        else 0
    )
    highlight_recall = (
        highlight_tp / (highlight_tp + highlight_fn)
        if (highlight_tp + highlight_fn) > 0
        else 0
    )
    highlight_f1 = (
        2 * highlight_precision * highlight_recall / (highlight_precision + highlight_recall)
        if (highlight_precision + highlight_recall) > 0
        else 0
    )

    red_precision = red_tp / (red_tp + red_fp) if (red_tp + red_fp) > 0 else 0
    red_recall = red_tp / (red_tp + red_fn) if (red_tp + red_fn) > 0 else 0
    red_f1 = (
        2 * red_precision * red_recall / (red_precision + red_recall)
        if (red_precision + red_recall) > 0
        else 0
    )

    arrow_precision = arrow_tp / (arrow_tp + arrow_fp) if (arrow_tp + arrow_fp) > 0 else 0
    arrow_recall = arrow_tp / (arrow_tp + arrow_fn) if (arrow_tp + arrow_fn) > 0 else 0
    arrow_f1 = (
        2 * arrow_precision * arrow_recall / (arrow_precision + arrow_recall)
        if (arrow_precision + arrow_recall) > 0
        else 0
    )

    flip_accuracy = flip_correct / flip_total if flip_total > 0 else 0

    print("\nPiece Recognition:")
    print(f"  Accuracy: {piece_accuracy:.4f} ({correct_squares}/{total_squares} squares)")

    print("\nHighlight Detection (any):")
    print(f"  Precision: {highlight_precision:.4f}")
    print(f"  Recall:    {highlight_recall:.4f}")
    print(f"  F1 Score:  {highlight_f1:.4f}")
    print(f"  (TP={highlight_tp}, FP={highlight_fp}, FN={highlight_fn})")

    print("\nRed Highlight Detection:")
    print(f"  Precision: {red_precision:.4f}")
    print(f"  Recall:    {red_recall:.4f}")
    print(f"  F1 Score:  {red_f1:.4f}")
    print(f"  (TP={red_tp}, FP={red_fp}, FN={red_fn})")

    print("\nArrow Detection:")
    print(f"  Precision: {arrow_precision:.4f}")
    print(f"  Recall:    {arrow_recall:.4f}")
    print(f"  F1 Score:  {arrow_f1:.4f}")
    print(f"  (TP={arrow_tp}, FP={arrow_fp}, FN={arrow_fn})")

    print("\nPerspective (Flip) Detection:")
    print(f"  Accuracy: {flip_accuracy:.4f} ({flip_correct}/{flip_total})")

    print("=" * 60)


def display_confidence(result, idx_to_piece):
    """Display confidence scores for predictions."""

    print("\n" + "=" * 60)
    print("CONFIDENCE SCORES")
    print("=" * 60)
    print(ChessInference.format_board_detection(result))

    print("\nPiece Predictions by Square:")
    print("-" * 40)
    pieces = result["pieces"]
    conf_p = result["pieces_confidence"]

    for rank in range(7, -1, -1):
        row_str = f"Rank {rank + 1}: "
        for file in range(8):
            sq = rank * 8 + file
            piece = idx_to_piece.get(int(pieces[sq]), ".")
            if piece is None:
                piece = "."
            conf = conf_p[sq]
            row_str += f"{piece}({conf:.2f}) "
        print(row_str)

    print("\nPiece Confidence Stats:")
    print(f"  Mean: {conf_p.mean():.4f}")
    print(f"  Min:  {conf_p.min():.4f}")
    print(f"  Max:  {conf_p.max():.4f}")

    low_conf_threshold = 0.7
    low_conf_squares = [
        (sq, pieces[sq], conf_p[sq]) for sq in range(64) if conf_p[sq] < low_conf_threshold
    ]
    if low_conf_squares:
        print(f"\n  Squares below {low_conf_threshold} confidence:")
        for sq, piece_idx, conf in sorted(low_conf_squares, key=lambda item: item[2]):
            piece = idx_to_piece.get(int(piece_idx), "?")
            if piece is None:
                piece = "empty"
            print(f"    {chess.square_name(sq)}: {piece} ({conf:.4f})")

    print("\nHighlight Predictions:")
    print("-" * 40)
    highlights = result["highlights"]
    conf_h = result["highlights_confidence"]
    highlight_labels = {0: "none", 1: "generic", 2: "red"}
    highlighted_squares = [
        (sq, highlights[sq], conf_h[sq]) for sq in range(64) if highlights[sq] > 0
    ]

    if highlighted_squares:
        for sq, hl_type, conf in highlighted_squares:
            label = highlight_labels.get(int(hl_type), "?")
            print(f"  {chess.square_name(sq)}: {label} ({conf:.4f})")
    else:
        print("  No highlights detected")

    print(
        f"\nPerspective: {'Black' if result['flipped'] else 'White'} "
        f"(confidence: {result['confidence_flipped']:.4f})"
    )

    print("\nArrow Predictions:")
    print("-" * 40)
    arrows = result["arrows"]
    probs_a = result["arrows_probs"]

    if arrows:
        print(f"  {len(arrows)} arrows detected:")
        for start, end in arrows[:10]:
            print(f"    {chess.square_name(start)} -> {chess.square_name(end)}")
        if len(arrows) > 10:
            print(f"    ... and {len(arrows) - 10} more")
    else:
        print("  No arrows detected")

    print(f"  Max arrow probability: {probs_a.max():.4f}")
    print("=" * 60)


def process_image(
    inference,
    image_path,
    output_path=None,
    show_confidence=False,
    piece_theme=None,
    board_theme=None,
    detect_board=True,
):
    if not output_path:
        base, _ = os.path.splitext(image_path)
        output_path = f"{base}_prediction.png"

    print(f"Processing {image_path} -> {output_path}")
    result = inference.predict(image_path, detect_board=detect_board)

    if result:
        inference.visualize(
            result,
            output_path,
            piece_theme=piece_theme,
            board_theme=board_theme,
        )
        print(ChessInference.format_board_detection(result))
        print(f"Predicted FEN: {result['fen']}")
        if show_confidence:
            display_confidence(result, inference.idx_to_piece)


def run_gui(inference, piece_theme=None, board_theme=None):
    gui = ChessInferenceGUI(
        inference,
        piece_theme=piece_theme,
        board_theme=board_theme,
    )
    gui.run()


def main():
    parser = argparse.ArgumentParser(description="Inference for Chess Vision")
    parser.add_argument("--image", type=str, help="Path to a single input image")
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to a directory of images to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for visualization in single-image mode",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Print confidence details for predictions",
    )
    parser.add_argument(
        "--piece-theme",
        type=str,
        default=DEFAULT_PIECE_THEME,
        help="Piece theme for rendered predictions",
    )
    parser.add_argument(
        "--board-theme",
        type=str,
        default=DEFAULT_BOARD_THEME,
        help="Board theme for rendered predictions",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop GUI explicitly",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Use stdin path mode instead of the GUI when no image or directory is provided",
    )
    parser.add_argument(
        "--skip-board-detection",
        action="store_true",
        help="Disable GLIP board detection and run inference on the full image as-is",
    )

    args = parser.parse_args()

    try:
        inference = ChessInference(args.model, args.device)
    except Exception as exc:
        print(f"Failed to initialize model: {exc}")
        return

    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: {args.directory} is not a valid directory")
            return

        image_files = [
            filename
            for filename in os.listdir(args.directory)
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
            and not filename.lower().endswith("_prediction.png")
        ]

        if not image_files:
            print(f"No image files found in {args.directory}")
            return

        labels_path = os.path.join(args.directory, "labels.csv")
        labels = None
        if os.path.exists(labels_path):
            print("Found labels.csv - will compute performance statistics")
            labels = load_labels(labels_path)

        print(f"Found {len(image_files)} images in {args.directory}")

        all_results = []
        for index, filename in enumerate(sorted(image_files), start=1):
            image_path = os.path.join(args.directory, filename)
            print(f"\n[{index}/{len(image_files)}] Processing {filename}")

            result = inference.predict(
                image_path,
                detect_board=not args.skip_board_detection,
            )
            if not result:
                continue

            base, _ = os.path.splitext(image_path)
            output_path = f"{base}_prediction.png"
            inference.visualize(
                result,
                output_path,
                piece_theme=args.piece_theme,
                board_theme=args.board_theme,
            )

            if args.show_confidence:
                display_confidence(result, inference.idx_to_piece)

            if labels and filename in labels:
                all_results.append(
                    {
                        "filename": filename,
                        "prediction": result,
                        "ground_truth": labels[filename],
                    }
                )

        print(f"\nCompleted processing {len(image_files)} images.")

        if labels and all_results:
            compute_statistics(all_results)
        return

    if args.image:
        process_image(
            inference,
            args.image,
            args.output,
            args.show_confidence,
            piece_theme=args.piece_theme,
            board_theme=args.board_theme,
            detect_board=not args.skip_board_detection,
        )
        return

    if args.stdin:
        print("Running in stdin mode. Enter image paths one per line.")
        try:
            import sys

            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]

                process_image(
                    inference,
                    line,
                    show_confidence=args.show_confidence,
                    piece_theme=args.piece_theme,
                    board_theme=args.board_theme,
                    detect_board=not args.skip_board_detection,
                )
                print("Ready for next image.")
        except KeyboardInterrupt:
            print("\nExiting.")
        return

    if args.gui or (not args.image and not args.directory and not args.stdin):
        try:
            run_gui(
                inference,
                piece_theme=args.piece_theme,
                board_theme=args.board_theme,
            )
        except Exception as exc:
            print(f"GUI mode failed: {exc}")
            print("Use --stdin to fall back to terminal path mode.")


if __name__ == "__main__":
    main()
