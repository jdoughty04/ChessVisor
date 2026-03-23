import os
import csv
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import Image

class ChessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Mapping for pieces
        self.piece_to_idx = {
            None: 0,
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }
        
        # Load CSV
        csv_path = os.path.join(root_dir, "labels.csv")
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                # Row format depends on crop mode but we appended new columns at the end.
                # Standard: filename, fen, bbox_x, y, w, h, highlights, arrows, flipped
                # Crop: filename, fen, highlights, arrows, flipped
                # The last 3 columns are what we want + fen + filename.
                if len(row) >= 5:
                    self.data.append(row)
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # filename, fen = self.data[idx] # Error: too many values to unpack
        filename = self.data[idx][0]
        fen = self.data[idx][1]
        img_path = os.path.join(self.root_dir, filename)
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Parse FEN to get logical piece positions first
        simple_fen = fen.split(" ")[0]
        rows = simple_fen.split("/")
        
        pieces_logical = torch.zeros(64, dtype=torch.long)
        
        for r_idx, row_str in enumerate(rows):
            actual_rank = 7 - r_idx
            file_idx = 0
            
            for char in row_str:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    square_idx = actual_rank * 8 + file_idx
                    pieces_logical[square_idx] = self.piece_to_idx[char]
                    file_idx += 1
                
        # Parse row data to get flip status first (needed for visual conversion)
        row_len = len(self.data[idx])
        if row_len == 5:
             hl_str = self.data[idx][2]
             arr_str = self.data[idx][3]
             flip_str = self.data[idx][4]
        else:
             hl_str = self.data[idx][6]
             arr_str = self.data[idx][7]
             flip_str = self.data[idx][8]
        
        # Helper to map Logical(0..63) to Visual(0..63)
        # Visual 0 is Top-Left (0,0)
        def to_visual(logical_idx, flipped_str):
            rank = logical_idx // 8
            file = logical_idx % 8
            
            if flipped_str == "True":
                # Black Perspective:
                # Rank 0 (1) is Top (Row 0)
                # File 0 (a) is Right (Col 7)
                r = rank
                c = 7 - file
            else:
                # White Perspective (Standard):
                # Rank 0 (1) is Bottom (Row 7)
                # File 0 (a) is Left (Col 0)
                r = 7 - rank
                c = file
                
            return r * 8 + c
        
        # Convert pieces from logical to visual order
        pieces_visual = torch.zeros(64, dtype=torch.long)
        for logical_sq in range(64):
            visual_sq = to_visual(logical_sq, flip_str)
            pieces_visual[visual_sq] = pieces_logical[logical_sq]
             
        # Parse and convert highlights from logical to visual order
        hl_dict = json.loads(hl_str)
        hl_visual = torch.zeros(64, dtype=torch.long)
        for sq_str, val in hl_dict.items():
            logical_sq = int(sq_str)
            visual_sq = to_visual(logical_sq, flip_str)
            hl_visual[visual_sq] = int(val)
            
        # Parse Arrows (already converted to visual in original code)
        arr_list = json.loads(arr_str)
        arr_tensor = torch.zeros((64, 64), dtype=torch.float32)

        for arrow in arr_list:
            start_logical, end_logical = arrow
            
            start_vis = to_visual(start_logical, flip_str)
            end_vis = to_visual(end_logical, flip_str)
            
            arr_tensor[start_vis, end_vis] = 1.0
            
        # Parse Perspective
        is_flipped = 1.0 if flip_str == "True" else 0.0
        flip_tensor = torch.tensor([is_flipped], dtype=torch.float32)

        # Arrow Count
        arrow_cnt = float(len(arr_list))
        cnt_tensor = torch.tensor([arrow_cnt], dtype=torch.float32)

        return {
            "image": image,
            "pieces": pieces_visual,  # Now in visual order
            "highlights": hl_visual,  # Now in visual order
            "arrows": arr_tensor,
            "flipped": flip_tensor,
            "arrow_count": cnt_tensor
        }

import io
class MemmapDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load Index
        index_path = os.path.join(root_dir, "dataset_index.json")
        bin_path = os.path.join(root_dir, "dataset.bin")
        
        if not os.path.exists(index_path) or not os.path.exists(bin_path):
            raise FileNotFoundError("Packed dataset not found. Run pack_dataset.py first.")
            
        with open(index_path, "r") as f:
            self.index = json.load(f)
            
        # Flatten index to list for integer indexing
        # self.index is dict {filename: data}
        self.keys = [k for k in self.index.keys() if k != "_meta"]
        
        self.bin_path = bin_path
        # We open the file lazily per worker or keep a handle?
        # File handles can't be pickled for multiprocessing.
        # We'll open in __getitem__ or use a singleton per worker style?
        # Actually simplest is just open/seek/read/close. OS caching handles the rest.
        # Ideally keep open if num_workers=0. For workers, each needs its own handle.
        self.bin_file = None 
        
        # Mappings
        self.piece_to_idx = {
            None: 0,
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }

    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self, idx):
        if self.bin_file is None:
            self.bin_file = open(self.bin_path, "rb")
            
        filename = self.keys[idx]
        entry = self.index[filename]
        
        offset = entry["offset"]
        length = entry["length"]
        labels = entry["labels"]
        
        # Mode detection
        mode = self.index.get("_meta", {}).get("mode", "file")
        
        # Read Data
        self.bin_file.seek(offset)
        data = self.bin_file.read(length)
        
        if mode == "raw":
             # Raw RGB Bytes of size 256x256
             image = Image.frombytes("RGB", (256, 256), data)
        else:
             # Standard encoded image
             image = Image.open(io.BytesIO(data)).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Process Labels (Copied from ChessDataset logic)
        # Labels list depends on row format (Full vs Crop)
        # We need logic to detect format.
        # Crop: fen, hl, arr, flip (len 4)
        # Full: fen, x, y, w, h, hl, arr, flip (len 8)
        
        fen = labels[0]
        
        if len(labels) == 4:
             hl_str = labels[1]
             arr_str = labels[2]
             flip_str = labels[3]
        elif len(labels) == 8:
             hl_str = labels[5]
             arr_str = labels[6]
             flip_str = labels[7]
        else:
             # Fallback
             hl_str = "{}"
             arr_str = "[]"
             flip_str = "False"

        # ... (Duplicate parsing logic or refactor? Let's duplicate for speed/independence)
        
        # Parse FEN to get logical positions
        simple_fen = fen.split(" ")[0]
        rows = simple_fen.split("/")
        pieces_logical = torch.zeros(64, dtype=torch.long)
        for r_idx, row_str in enumerate(rows):
            actual_rank = 7 - r_idx
            file_idx = 0
            for char in row_str:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    square_idx = actual_rank * 8 + file_idx
                    pieces_logical[square_idx] = self.piece_to_idx[char]
                    file_idx += 1
        
        # Helper to map Logical(0..63) to Visual(0..63)
        def to_visual(logical_idx, flipped_str):
            rank = logical_idx // 8
            file = logical_idx % 8
            if flipped_str == "True":
                r = rank
                c = 7 - file
            else:
                r = 7 - rank
                c = file
            return r * 8 + c
        
        # Convert pieces from logical to visual order
        pieces_visual = torch.zeros(64, dtype=torch.long)
        for logical_sq in range(64):
            visual_sq = to_visual(logical_sq, flip_str)
            pieces_visual[visual_sq] = pieces_logical[logical_sq]
                    
        # Parse and convert highlights to visual order
        hl_dict = json.loads(hl_str)
        hl_visual = torch.zeros(64, dtype=torch.long)
        for sq_str, val in hl_dict.items():
            logical_sq = int(sq_str)
            visual_sq = to_visual(logical_sq, flip_str)
            hl_visual[visual_sq] = int(val)
            
        # Arrows (already uses visual order)
        arr_list = json.loads(arr_str)
        arr_tensor = torch.zeros((64, 64), dtype=torch.float32)
            
        for arrow in arr_list:
            start_logical, end_logical = arrow
            start_vis = to_visual(start_logical, flip_str)
            end_vis = to_visual(end_logical, flip_str)
            arr_tensor[start_vis, end_vis] = 1.0
            
        is_flipped = 1.0 if flip_str == "True" else 0.0
        flip_tensor = torch.tensor([is_flipped], dtype=torch.float32)
        
        cnt_tensor = torch.tensor([float(len(arr_list))], dtype=torch.float32)
        
        return {
            "image": image,
            "pieces": pieces_visual,  # Now in visual order
            "highlights": hl_visual,  # Now in visual order
            "arrows": arr_tensor,
            "flipped": flip_tensor,
            "arrow_count": cnt_tensor
        }
