import os
import sys
import time
import json
import argparse
import collections
from pprint import pprint
import queue
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import chess

from localizer import BoardLocalizer
from model import ChessNet as EnhancedChessNet

IDX_TO_PIECE = {
    0: None, 
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}

SQUARE_NAMES = [
    chess.square_name(i) for i in range(64)
]

def format_board_ascii(fen, black_pov=False):
    board = chess.Board(fen)
    builder = []
    # If black_pov=True, we want Rank 1 at top? No, usually "Black POV" means visually flipped.
    # Standard string is Rank 8 top.
    # If flipped, we print Rank 1 top.
    
    ranks = range(8) if black_pov else range(7, -1, -1)
    files = range(7, -1, -1) if black_pov else range(8)
    
    for rank in ranks:
        row_pieces = []
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            symbol = piece.symbol() if piece else "."
            row_pieces.append(symbol)
        builder.append(" ".join(row_pieces))
    return "\n".join(builder)

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class PadToSquare:
    def __init__(self, size=256, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill
    def __call__(self, img):
        w, h = img.size
        new_img = Image.new("RGB", (self.size, self.size), self.fill)
        new_img.paste(img, ((self.size - w) // 2, (self.size - h) // 2))
        return new_img

def indices_to_fen(indices_tensor):
    rows = []
    for r in range(7, -1, -1): 
        row_str = ""
        empty_count = 0
        for f in range(8): 
            idx = r * 8 + f
            piece = IDX_TO_PIECE[indices_tensor[idx].item()]
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += piece
        if empty_count > 0:
            row_str += str(empty_count)
        rows.append(row_str)
    return "/".join(rows) + " w - - 0 1" 

def format_ts(seconds):
    """Formats seconds into HH:MM:SS.mmm string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def visual_to_logical(v_idx, black_pov):
    r_vis = v_idx // 8
    c_vis = v_idx % 8
    if black_pov:
        # Black POV: Visual Row 0 is Rank 1 (Logical Rank 0)
        # Visual Col 0 is File h (Logical File 7)
        rank = r_vis
        file = 7 - c_vis
    else:
        # White POV: Visual Row 0 is Rank 8 (Logical Rank 7)
        # Visual Col 0 is File a (Logical File 0)
        rank = 7 - r_vis
        file = c_vis
    return rank * 8 + file

def preprocess_for_localization(frames_bgr, target_size=224):
    """
    CPU-Bound: Converts BGR frames to tensors ready for Localizer.
    """
    pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    
    loc_inputs = []
    meta = [] # (pad_x, pad_y, scale, w_orig, h_orig)
    
    for pil_img in pil_imgs:
        w_orig, h_orig = pil_img.size
        scale = target_size / max(w_orig, h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        
        img_resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        img_padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        img_padded.paste(img_resized, (pad_x, pad_y))
        
        meta.append((pad_x, pad_y, scale, w_orig, h_orig))
        t_img = transforms.ToTensor()(img_padded)
        loc_inputs.append(t_img)
        
    tensor_batch = torch.stack(loc_inputs)
    return pil_imgs, tensor_batch, meta

def process_gotham_frame(frame, coords):
    """
    Process a single frame using OpenCV for speed.
    """
    y1, y2, x1, x2 = coords
    
    # 1. Crop
    crop = frame[y1:y2, x1:x2]
    
    # 2. Resize Logic (replacing ResizeMax(256))
    max_size = 256
    h, w = crop.shape[:2]
    if w == 0 or h == 0: return Image.fromarray(crop) # specific fallback
    
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # cv2.resize is usually faster than PIL
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Convert BGR->RGB and to PIL
    # We return PIL because cls_transform expects it (PadToSquare)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

class FrameLoader(threading.Thread):
    def __init__(self, video_path, batch_size, interval=0.1, max_duration=None, start_time=0.0, gotham_mode=False, queue_size=2):
        super().__init__()
        self.video_path = video_path
        self.batch_size = batch_size
        self.interval = interval
        self.max_duration = max_duration
        self.start_time = start_time
        self.gotham_mode = gotham_mode
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.daemon = True 
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not check video duration.")
            self.target_indices = []
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        print(f"Video Duration: {duration:.2f}s, FPS: {fps}, Gotham: {self.gotham_mode}")
        
        self.target_indices = []
        t = self.start_time
        end_time = duration
        if self.max_duration:
            end_time = min(duration, self.start_time + self.max_duration)

        while t < end_time:
            idx = int(t * fps)
            if idx < total_frames:
                self.target_indices.append((idx, t))
            t += self.interval
            
        print(f"Loader prepared: {len(self.target_indices)} targets (Start: {self.start_time}s).")
        
        # Determine workers
        self.workers = 12

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        current_batch_frames = []
        current_batch_ts = []
        
        frame_idx = 0
        target_idx_ptr = 0
        total_targets = len(self.target_indices)
        
        G_Y1, G_Y2 = 5, 1068
        G_X1, G_X2 = 69, 1130
        
        executor = ThreadPoolExecutor(max_workers=self.workers)
        
        # Seek to start_time if specified
        if self.start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time * 1000)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Seeking video to {self.start_time}s (frame {frame_idx})")
            
            # Flush a few frames to clear decoding artifacts (common in H.264 seeks)
            for _ in range(30):
                cap.grab()
                frame_idx += 1

        while not self.stopped and target_idx_ptr < total_targets:
            target_frame_idx, target_ts = self.target_indices[target_idx_ptr]
            
            # Skip frames until we reach target
            while frame_idx < target_frame_idx:
                cap.grab()
                frame_idx += 1
            
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            target_idx_ptr += 1
            
            current_batch_frames.append(frame)
            current_batch_ts.append(target_ts)
            
            if len(current_batch_frames) >= self.batch_size or target_idx_ptr >= total_targets:
                if self.stopped: break
                
                if self.gotham_mode:
                    # Parallel Processing
                    futures = [executor.submit(process_gotham_frame, f, (G_Y1, G_Y2, G_X1, G_X2)) for f in current_batch_frames]
                    pil_imgs = [f.result() for f in futures]
                        
                    loc_batch = None
                    meta = None
                else:
                    pil_imgs, loc_batch, meta = preprocess_for_localization(current_batch_frames)
                
                self.queue.put((pil_imgs, loc_batch, meta, current_batch_ts))
                
                current_batch_frames = []
                current_batch_ts = []
        
        executor.shutdown()
        cap.release()
        self.queue.put(None)

    def stop(self):
        self.stopped = True
        try:
             while not self.queue.empty():
                 self.queue.get_nowait()
        except: pass

from PIL import Image, ImageDraw

def log_debug_probs(h_probs_tensor, a_probs_tensor, timestamp_str, is_black_pov=False):
    """
    Logs raw probabilities for highlights and arrows to a JSONL file.
    h_probs_tensor: [64, 1] or [64, 3]
    a_probs_tensor: [64, 64] (Start x End)
    """
    debug_dir = r"d:\python_code\project-12-4\chess_commentary_pipeline\data\debug_visuals"
    os.makedirs(debug_dir, exist_ok=True)
    log_path = os.path.join(debug_dir, "debug_log.jsonl")
    
    # 1. Highlights
    h_candidates = []
    
    if h_probs_tensor.dim() >= 2 and h_probs_tensor.shape[-1] > 1:
         # Multi-channel [64, C]
         # Channel 1 = Generic/Yellow, Channel 2 = Red
         c1 = h_probs_tensor[:, 1].cpu().numpy()  # Yellow/Generic
         c2 = h_probs_tensor[:, 2].cpu().numpy() if h_probs_tensor.shape[-1] > 2 else None  # Red
         
         for idx, prob in enumerate(c1):
             if idx >= 64: break
             msg = ""
             if prob > 0.05:
                 msg += f"Y:{prob:.3f}"  # Y for Yellow/Generic
             
             if c2 is not None and c2[idx] > 0.05:
                 msg += f"|R:{c2[idx]:.3f}"  # R for Red
                 
             if msg:
                 log_idx = visual_to_logical(idx, is_black_pov)
                 h_candidates.append(f"{SQUARE_NAMES[log_idx]}({msg})")
    else:
         # Single channel [64]
         h_flat = h_probs_tensor.reshape(-1).cpu().numpy()
         for idx, prob in enumerate(h_flat):
             if idx >= 64: break
             if prob > 0.1:
                 log_idx = visual_to_logical(idx, is_black_pov)
                 h_candidates.append(f"{SQUARE_NAMES[log_idx]}:{prob:.3f}")
            
    # 2. Arrows
    # a_probs_tensor is [64, 64] (StartVisual x EndVisual)
    # Get all indices > threshold
    a_np = a_probs_tensor.cpu().numpy()
    rows, cols = np.where(a_np > 0.1)
    
    a_candidates = []
    for r, c in zip(rows, cols):
        prob = a_np[r, c]
        
        fr_log = visual_to_logical(r, is_black_pov)
        to_log = visual_to_logical(c, is_black_pov)
        
        arr_str = f"{SQUARE_NAMES[fr_log]}{SQUARE_NAMES[to_log]}"
        a_candidates.append(f"{arr_str}:{prob:.3f}")
            
    if not h_candidates and not a_candidates:
        return
        
    entry = {
        "timestamp": timestamp_str,
        "highlights": h_candidates,
        "arrows": a_candidates
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")

class GamestateExtractor:
    def __init__(self, model_path="enhanced_model_new.pth", localizer_path="localizer.pth", device=None, gotham_mode=False, debug_mode=False):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gotham_mode = gotham_mode
        self.debug_mode = debug_mode
        print(f"Using device: {self.device}, Debug Visuals: {self.debug_mode}")
        
        self.localizer = BoardLocalizer().to(self.device)
        self.localizer.load_state_dict(torch.load(localizer_path, map_location=self.device))
        self.localizer.eval()
        
        # Load Enhanced Model
        self.classifier = EnhancedChessNet(pretrained=False).to(self.device)
        self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        self.classifier.eval()
        
        self.cls_transform = transforms.Compose([
            PadToSquare(256),
            transforms.ToTensor(),
            NORMALIZE
        ])
        
        self.normalize_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def predict_batch_from_preprocessed(self, pil_imgs, loc_batch, meta, timestamps=None):
        if not pil_imgs: return [], [], [], []
        
        crops = []
        
        if self.gotham_mode:
            # pil_imgs are ALREADY cropped AND resized
            crops = pil_imgs
        else:
            # Standard mode (Localizer)
            loc_batch_gpu = loc_batch.to(self.device)
            loc_batch_gpu = self.normalize_tensor(loc_batch_gpu)
            
            with torch.no_grad():
                bbox_preds = self.localizer(loc_batch_gpu)
            bbox_preds = bbox_preds.cpu().numpy()
            target_size = 224
            
            for i, (bx, by, bw, bh) in enumerate(bbox_preds):
                pad_x, pad_y, scale, w_orig, h_orig = meta[i]
                x_raw = (bx * target_size) - pad_x
                y_raw = (by * target_size) - pad_y
                w_final = (bw * target_size) / scale
                h_final = (bh * target_size) / scale
                x_final = x_raw / scale
                y_final = y_raw / scale
                pad_w = w_final * 0.03
                pad_h = h_final * 0.03
                x1 = max(0, int(x_final - pad_w))
                y1 = max(0, int(y_final - pad_h))
                x2 = min(w_orig, int(x_final + w_final + pad_w))
                y2 = min(h_orig, int(y_final + h_final + pad_h))
                
                crop = pil_imgs[i] if (x2 <= x1 or y2 <= y1) else pil_imgs[i].crop((x1, y1, x2, y2))
                crop_resized = crop.resize((256, int(256 * crop.height / crop.width)), Image.BICUBIC) if crop.width > crop.height else crop.resize((int(256 * crop.width / crop.height), 256), Image.BICUBIC)
                crops.append(crop_resized)
        
        # Batch Classification
        cls_inputs = [self.cls_transform(c) for c in crops]
        cls_batch = torch.stack(cls_inputs).to(self.device)
        
        with torch.no_grad():
            p, h, a, flip = self.classifier(cls_batch)
            
            # 1. Pieces
            _, preds = torch.max(p, 2)
            
            # 2. Highlights
            # Model output is [B, 64, 3] (Background=0, Generic/Yellow=1, Red=2)
            # Use Softmax for multi-class
            h_probs = torch.softmax(h, dim=2) 
            
            # Channel 1 is Generic (yellow/green), Channel 2 is Red
            h_generic = (h_probs[:, :, 1] > 0.5) 
            h_red = (h_probs[:, :, 2] > 0.5)
            
            # 3. Arrows
            a_probs = torch.sigmoid(a) # [B, 4096] or [B, 64, 64]
            # Threshold
            a_detect = (a_probs > 0.5)
            
            # 4. Perspective
            flip_prob = torch.sigmoid(flip)
            batch_flip_int = (flip_prob > 0.5).long().cpu().numpy().flatten()
            
            # Convert to Square Lists
            preds_np = preds.cpu().numpy()
            h_generic_np = h_generic.cpu().numpy()
            h_red_np = h_red.cpu().numpy()
            a_detect_np = a_detect.cpu().numpy()
            
            fens = []
            batch_highlights = []
            batch_arrows = []
            batch_perspectives = []
            
            for i in range(len(preds_np)):
                # Perspective FIRST (needed for visual->logical conversion)
                pov = "black" if batch_flip_int[i] > 0 else "white"
                batch_perspectives.append(pov)
                
                is_black_pov = (batch_flip_int[i] > 0)
                
                # Convert pieces from visual to logical order for FEN
                preds_visual = preds_np[i]
                preds_logical = np.zeros(64, dtype=np.int64)
                for vis_sq in range(64):
                    log_sq = visual_to_logical(vis_sq, is_black_pov)
                    preds_logical[log_sq] = preds_visual[vis_sq]
                
                # FEN (now using logical order)
                fen = indices_to_fen(preds_logical)
                fens.append(fen)
                
                # Highlights: include color info
                # Generic (yellow) from channel 1, Red from channel 2
                generic_indices = torch.nonzero(torch.tensor(h_generic_np[i])).flatten().numpy()
                red_indices = torch.nonzero(torch.tensor(h_red_np[i])).flatten().numpy()
                
                hl_list = []
                for idx in generic_indices:
                    log_idx = visual_to_logical(idx, is_black_pov)
                    hl_list.append({"square": SQUARE_NAMES[log_idx], "color": "yellow"})
                for idx in red_indices:
                    log_idx = visual_to_logical(idx, is_black_pov)
                    # Check if already added as generic (shouldn't happen but just in case)
                    if not any(h["square"] == SQUARE_NAMES[log_idx] for h in hl_list):
                        hl_list.append({"square": SQUARE_NAMES[log_idx], "color": "red"})
                
                if len(hl_list) > 16:
                   hl_list = [] # Noise filter
                   
                batch_highlights.append(hl_list)

                # Arrows
                # Model outputs Visual Indices (0=TopLeft). Dataset maps Logical->Visual based on POV.
                # We must reverse this mappings: Visual -> Logical.
                
                # a_detect_np[i] is (64, 64) boolean mask
                
                # a_detect_np[i] is (64, 64) boolean mask
                # torch.nonzero or numpy.argwhere to get pairs
                arrow_indices = np.argwhere(a_detect_np[i]) # [[r1,c1], [r2,c2]...]
                
                arrow_strs = []
                for (fr_vis, to_vis) in arrow_indices:
                     fr_log = visual_to_logical(fr_vis, is_black_pov)
                     to_log = visual_to_logical(to_vis, is_black_pov)
                     
                     arrow_strs.append(f"{SQUARE_NAMES[fr_log]}{SQUARE_NAMES[to_log]}")
                batch_arrows.append(arrow_strs)

                if self.debug_mode:
                     # Pass raw tensors to logger
                     if timestamps:
                         ts_str = format_ts(timestamps[i])
                     else:
                         ts_str = f"{time.time()}_{i}"
                     
                     # Log Channel 1 (Red) and Ch 2 (Green)
                     log_debug_probs(h_probs[i], a_probs[i], ts_str, is_black_pov=is_black_pov)
            
        return fens, batch_highlights, batch_arrows, batch_perspectives

def robust_fen_diff(fen1, fen2):
    p1 = fen1.split()[0]
    p2 = fen2.split()[0]
    return p1 != p2

def is_start_position(fen):
    parts = fen.split()[0].split('/')
    if len(parts) != 8: return False
    if parts[0] != "rnbqkbnr": return False
    if parts[1] != "pppppppp": return False
    if parts[7] != "RNBQKBNR": return False
    if parts[6] != "PPPPPPPP": return False
    return True



def process_video_fast(video_path, output_path, extractor, max_duration=None, start_time=0.0, include_board=False, black_pov=False):
    # Config
    INTERVAL = 0.1 
    BATCH_SIZE = 128 
    
    loader = FrameLoader(video_path, BATCH_SIZE, INTERVAL, max_duration, start_time, gotham_mode=extractor.gotham_mode, queue_size=3)
    loader.start()
    
    # State
    gamestate_log = []
    last_fen = None
    last_game_start_time = -999.0
    game_board = None
    
    # Probation State
    pending_reset_fen = None
    pending_reset_ts = None
    PROBATION_DURATION = 10.0 
    
    # Visual Event Tracking State
    # Track active highlights: {(square, color): start_timestamp}
    active_highlights = {}
    # Track active arrows: {arrow_str: start_timestamp}
    active_arrows = {}
    # Log of visual events (separate from gamestate events)
    visual_events_log = []
    
    batch_idx = 0
    
    try:
        while True:
            t_wait = time.time()
            item = loader.queue.get()
            t_got = time.time()
            if item is None: break
            
            pil_imgs, loc_batch, meta, batch_ts_list = item
            
            wait_time = t_got - t_wait
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            
            t_inf_start = time.time()
            pred_fens, pred_highs, pred_arrows, pred_povs = extractor.predict_batch_from_preprocessed(pil_imgs, loc_batch, meta, timestamps=batch_ts_list)
            t_inf_end = time.time()
            
            print(f"Batch {batch_idx}: Wait={wait_time*1000:.1f}ms | Inference={(t_inf_end-t_inf_start)*1000:.1f}ms | RAM={mem:.0f}MB", flush=True)

            for i, predicted_fen in enumerate(pred_fens):
                current_timestamp = batch_ts_list[i]
                timestamp_str = format_ts(current_timestamp)
                
                curr_viz = {
                    "highlights": pred_highs[i],
                    "arrows": pred_arrows[i],
                    "perspective": pred_povs[i]
                }
                
                # --- Visual Event Tracking ---
                # Build current highlights set: {(square, color)}
                current_hl_set = set()
                for hl in pred_highs[i]:
                    if isinstance(hl, dict):
                        current_hl_set.add((hl["square"], hl["color"]))
                    else:
                        current_hl_set.add((hl, "unknown"))
                
                # Build current arrows set
                current_arrow_set = set(pred_arrows[i])
                
                # Detect ended highlights (were active, now gone)
                ended_highlights = set(active_highlights.keys()) - current_hl_set
                for hl_key in ended_highlights:
                    start_ts = active_highlights.pop(hl_key)
                    duration = current_timestamp - start_ts
                    if duration >= 0.3:  # Min duration to log (filter noise)
                        visual_events_log.append({
                            "type": "highlight",
                            "square": hl_key[0],
                            "color": hl_key[1],
                            "start_time": start_ts,
                            "start_time_str": format_ts(start_ts),
                            "end_time": current_timestamp,
                            "end_time_str": timestamp_str,
                            "duration": round(duration, 2)
                        })
                
                # Detect new highlights (not active, now present)
                new_highlights = current_hl_set - set(active_highlights.keys())
                for hl_key in new_highlights:
                    active_highlights[hl_key] = current_timestamp
                
                # Detect ended arrows (were active, now gone)
                ended_arrows = set(active_arrows.keys()) - current_arrow_set
                for arrow in ended_arrows:
                    start_ts = active_arrows.pop(arrow)
                    duration = current_timestamp - start_ts
                    if duration >= 0.3:  # Min duration to log
                        visual_events_log.append({
                            "type": "arrow",
                            "arrow": arrow,
                            "start_time": start_ts,
                            "start_time_str": format_ts(start_ts),
                            "end_time": current_timestamp,
                            "end_time_str": timestamp_str,
                            "duration": round(duration, 2)
                        })
                
                # Detect new arrows
                new_arrows = current_arrow_set - set(active_arrows.keys())
                for arrow in new_arrows:
                    active_arrows[arrow] = current_timestamp
                
                # --- Probation / Debouncing Logic ---
                # "Debouncing" means we only accept a new FEN (whether move or correction)
                # if it persists for N frames.
                
                if pending_reset_fen:
                    # We have a candidate FEN waiting for stability
                    if not robust_fen_diff(predicted_fen, pending_reset_fen):
                        # Matches candidate
                        pending_match_count += 1
                    else:
                        # Candidate failed stability check, reset
                        pending_reset_fen = predicted_fen
                        pending_reset_ts = current_timestamp
                        pending_match_count = 1
                        continue

                    # If stable enough, try to process it
                    STABILITY_THRESHOLD = 3 # frames (approx 0.3s)
                    if pending_match_count >= STABILITY_THRESHOLD:
                        pass # proceed to process pending_reset_fen
                    else:
                        continue # keep waiting
                
                else:
                    # No candidate, check if we need one
                    change_detected = False
                    if last_fen is None:
                        # Initial state: just accept immediately if valid
                        pass 
                    elif robust_fen_diff(predicted_fen, last_fen):
                        # Potential change detected, start probation
                        pending_reset_fen = predicted_fen
                        pending_reset_ts = current_timestamp
                        pending_match_count = 1
                        continue
                    else:
                        # Stable state ( matches last_fen )
                        pass

                # If we are here, we are either:
                # 1. Processing initial state (last_fen is None)
                # 2. Processing a stabilized new FEN (from pending_reset_fen)

                target_fen = pending_reset_fen if pending_reset_fen else predicted_fen
                # Use the timestamp of when the change STARTED (pending_reset_ts) if applicable
                target_ts = pending_reset_ts if pending_reset_fen else current_timestamp
                target_ts_str = format_ts(target_ts)

                # Reset pending state after processing
                pending_reset_fen = None
                pending_reset_ts = None
                pending_match_count = 0

                if last_fen is None:
                     try:
                         # Only initialize on fresh starting positions
                         if is_start_position(target_fen):
                             init_board = chess.Board() 
                             last_fen = target_fen
                             game_board = init_board
                             gamestate_log.append({
                                 "timestamp": target_ts, 
                                 "timestamp_str": target_ts_str, 
                                 "type": "init", 
                                 "fen": game_board.fen(),
                                 "visuals": curr_viz
                             })
                     except ValueError: pass
                else:
                     # We know robust_fen_diff(target_fen, last_fen) is True if we reached here from probation
                     # Or we are just re-verifying if logic fell through (shouldn't happen for stable state)
                     if not robust_fen_diff(target_fen, last_fen):
                        continue

                     # Change confirmed and stabilized. Check if move or correction.
                     if is_start_position(target_fen):
                         if (target_ts - last_game_start_time) > 60.0:
                              game_board = chess.Board() 
                              last_fen = target_fen
                              last_game_start_time = target_ts
                              gamestate_log.append({
                                  "timestamp": target_ts, 
                                  "timestamp_str": target_ts_str, 
                                  "type": "game_start", 
                                  "val": "New Game Detected", 
                                  "fen": game_board.fen(),
                                  "visuals": curr_viz
                              })
                              continue

                     try:
                        temp = game_board.copy()
                        pred_board = chess.Board(target_fen)
                        found = None
                        
                        # Optimization: Check exact FEN match first
                        target_board_fen = pred_board.board_fen()
                        
                        for m in temp.legal_moves:
                            temp.push(m)
                            if temp.board_fen() == target_board_fen:
                                found = m
                                break
                            temp.pop()
                        
                        if found:
                            san = game_board.san(found)
                            game_board.push(found)
                            gamestate_log.append({
                                "timestamp": target_ts, 
                                "timestamp_str": target_ts_str, 
                                "type": "move", 
                                "val": san, 
                                "fen": game_board.fen(), 
                                "uci": found.uci(),
                                "visuals": curr_viz
                            })
                            last_fen = target_fen
                        else:
                            # Exact match failed. Try RELAXED match.
                            # Visual model might be noisy (e.g. 1-2 squares wrong).
                            # Check if any legal move results in a board very close to target_fen.
                            best_move = None
                            min_dist = 999
                            
                            for m in temp.legal_moves:
                                temp.push(m)
                                # Simple hamming distance on piece map (all squares)
                                dist = 0
                                for sq in chess.SQUARES:
                                    p1 = temp.piece_at(sq)
                                    p2 = pred_board.piece_at(sq)
                                    if p1 != p2: dist += 1
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_move = m
                                
                                temp.pop()
                            
                            # Threshold check for relaxed match
                            if best_move and min_dist <= 2:
                                san = game_board.san(best_move)
                                game_board.push(best_move)
                                clean_fen = game_board.fen()
                                
                                gamestate_log.append({
                                    "timestamp": target_ts, 
                                    "timestamp_str": target_ts_str, 
                                    "type": "move", 
                                    "val": f"{san} (Inferred)", 
                                    "fen": clean_fen, 
                                    "uci": best_move.uci(),
                                    "visuals": curr_viz
                                })
                                last_fen = clean_fen
                            else:
                                # Multi-step inference attempt (BFS) - Depth limit 3
                                # If we can't find *one* move, maybe 2 or 3 moves happened?
                                found_path = None
                                # BFS Queue: (board_obj, path_moves)
                                q = collections.deque()
                                q.append((game_board.copy(), []))
                                
                                # Optimization: Don't search too broad.
                                visited = set()
                                visited.add(game_board.board_fen())
                                
                                MAX_DEPTH = 3
                                # Target FEN is potentially noisy, so we can't match it exactly in BFS easily.
                                # Instead, we can try to match the *piece map* with tolerance, OR
                                # if confidence is high, assume target_fen IS the goal.
                                # Let's try exact match to target_fen first (assuming target is stable and correct).
                                # If target is noisy, multi-step is very hard.
                                
                                # Let's assume target_fen is "ground truth" for the new state.
                                target_board_fen = chess.Board(target_fen).board_fen()
                                
                                while q:
                                    curr, path = q.popleft()
                                    
                                    if len(path) >= MAX_DEPTH: continue
                                    
                                    for m in curr.legal_moves:
                                        curr.push(m)
                                        bf = curr.board_fen()
                                        
                                        if bf == target_board_fen:
                                            found_path = path + [m]
                                            break
                                        
                                        if bf not in visited:
                                            visited.add(bf)
                                            # Pruning: If depth < MAX_DEPTH, enqueue
                                            if len(path) + 1 < MAX_DEPTH:
                                                q.append((curr.copy(), path + [m]))
                                        
                                        curr.pop()
                                    if found_path: break
                                    
                                if found_path:
                                    # We found a sequence of moves!
                                    # Log them ALL, perhaps with interpolated timestamps? 
                                    # For now, just log them sequentially at the same timestamp or slightly offset.
                                    step_ts = target_ts - 0.1 * len(found_path)
                                    for fm in found_path:
                                         step_ts += 0.1
                                         step_ts_str = format_ts(step_ts)
                                         san = game_board.san(fm)
                                         game_board.push(fm)
                                         clean_fen = game_board.fen()
                                         
                                         gamestate_log.append({
                                            "timestamp": step_ts, 
                                            "timestamp_str": step_ts_str, 
                                            "type": "move", 
                                            "val": f"{san} (Multi-Inferred)", 
                                            "fen": clean_fen, 
                                            "uci": fm.uci(),
                                            "visuals": curr_viz
                                         })
                                    last_fen = clean_fen
                                    
                                else:
                                    # Fallback to Correction
                                    val_msg = "Board Updated"
                                    
                                    new_board = chess.Board(target_fen)
                                    new_board.castling_rights = game_board.castling_rights
                                            
                                    game_board = new_board
                                    last_fen = target_fen
                                    gamestate_log.append({
                                        "timestamp": target_ts, 
                                        "timestamp_str": target_ts_str, 
                                        "type": "correction", 
                                        "val": val_msg, 
                                        "fen": game_board.fen(),
                                        "visuals": curr_viz
                                    })
                     except ValueError: pass
            
            # Incremental Save every 5 batches
            if batch_idx % 5 == 0:
                 output_data = {
                     "gamestate_events": gamestate_log,
                     "visual_events": visual_events_log
                 }
                 with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)

            batch_idx += 1

    except KeyboardInterrupt:
        print("Stopping...")
        loader.stop()
    
    # Final save with combined output
    output_data = {
        "gamestate_events": gamestate_log,
        "visual_events": visual_events_log
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"Done. Saved to {output_path}")
    print(f"  - {len(gamestate_log)} gamestate events")
    print(f"  - {len(visual_events_log)} visual events")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw/JalxHcGnpEc.webm", help="Path to video file")
    parser.add_argument("--output", default="data/processed/gamestate_log_fast.json", help="Path to output JSON")
    parser.add_argument("--duration", type=float, default=300, help="Process only first N seconds")
    parser.add_argument("--start", type=float, default=200.0, help="Start processing from N seconds")
    parser.add_argument("--gotham", default=True, action="store_true", help="Use GothamChess fixed crop")
    parser.add_argument("--include_board", default=True, action="store_true", help="Include ASCII board state in output")
    parser.add_argument("--black", default=False, action="store_true", help="Render board from Black's POV")
    parser.add_argument("--debug_visuals", default=True, action="store_true", help="Enable visual debugging (save drawn frames)")
    
    args = parser.parse_args()
    if not os.path.exists(args.video):
        alt = args.video.replace(".webm", ".mp4")
        if os.path.exists(alt): args.video = alt
            
    extractor = GamestateExtractor(gotham_mode=args.gotham, debug_mode=args.debug_visuals)
    process_video_fast(args.video, args.output, extractor, max_duration=args.duration, start_time=args.start, include_board=args.include_board, black_pov=args.black)
