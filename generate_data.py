import os
import random
import cv2
import numpy as np
import chess
from pathlib import Path
from PIL import Image
import json

# Configuration
ASSETS_DIR = Path("assets")
PIECES_DIR = ASSETS_DIR / "pieces"
BOARDS_DIR = ASSETS_DIR / "boards"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
OUTPUT_DIR = Path("output")
PREVIEW_DIR = OUTPUT_DIR / "preview"

# Ensure output directories
os.makedirs(PREVIEW_DIR, exist_ok=True)

class ChessGenerator:
    def __init__(self):
        self.piece_images = self._load_pieces()
        self.board_images = self._load_boards()
        self.background_images = self._load_backgrounds()
        
    def _load_pieces(self):
        """Load all piece images into memory organized by theme."""
        pieces = {}
        for theme_dir in PIECES_DIR.iterdir():
            if theme_dir.is_dir():
                theme = theme_dir.name
                pieces[theme] = {}
                for img_path in theme_dir.glob("*.png"):
                    # piece key: wp, bk, etc.
                    key = img_path.stem
                    pieces[theme][key] = Image.open(img_path).convert("RGBA")
        print(f"Loaded pieces for {len(pieces)} themes.")
        return pieces

    def _load_boards(self):
        """Load board images."""
        boards = {}
        for img_path in BOARDS_DIR.glob("*.png"):
            theme = img_path.stem
            boards[theme] = Image.open(img_path).convert("RGB")
        print(f"Loaded {len(boards)} board themes.")
        return boards

    def _load_backgrounds(self):
        """Load background images."""
        backgrounds = []
        if BACKGROUNDS_DIR.exists():
            for img_path in BACKGROUNDS_DIR.glob("*.jpg"):
                try:
                    backgrounds.append(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    print(f"Error loading background {img_path}: {e}")
        print(f"Loaded {len(backgrounds)} background images.")
        return backgrounds

    def get_random_assets(self):
        piece_theme = random.choice(list(self.piece_images.keys()))
        board_theme = random.choice(list(self.board_images.keys()))
        return self.piece_images[piece_theme], self.board_images[board_theme]

    def generate_fen(self):
        """Generate a random legal FEN position."""
        # Simple random game simulation
        board = chess.Board()
        moves = random.randint(0, 80)
        for _ in range(moves):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        return board

    def generate_scrambled_fen(self):
        """
        Generate a completely random (scrambled) position.
        Does not follow chess rules. Pieces are placed randomly.
        Used to train the model to look at the board, not infer from logic.
        """
        board = chess.Board(None) # Empty board
        
        # Pieces to place: King is not required, anything goes.
        # Let's pick 3 to 30 random pieces.
        num_pieces = random.randint(3, 30)
        
        squares = list(range(64))
        random.shuffle(squares)
        
        # Piece types with weighted probabilities (pawns more common)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        weights = [0.4, 0.15, 0.15, 0.15, 0.1, 0.05]
        
        for i in range(num_pieces):
            sq = squares[i]
            ptype = random.choices(piece_types, weights=weights, k=1)[0]
            color = random.choice([chess.WHITE, chess.BLACK])
            
            # Avoid pawns on back ranks (visually confusing/impossible even in scramble?)
            # Actually, let's allow it to be truly robust, but typically engines/fens hate it.
            # python-chess might complain if we try to get FEN or standard ops.
            # But we just use it for rendering.
            # Actually, python-chess board object handles it fine usually?
            # Let's keep pawns off rank 1 and 8 just to be 'somewhat' sane visual.
            rank = sq // 8
            if ptype == chess.PAWN and (rank == 0 or rank == 7):
                continue
                
            board.set_piece_at(sq, chess.Piece(ptype, color))
            
        return board

    def render_board(self, board, pieces, board_img, size=800, flipped=False, highlights=None):
        """
        Render the board state to an image.
        flipped: If True, render from Black's perspective.
        highlights: Optional iterable of (square_index, RGB_color) pairs
        rendered beneath pieces so highlights only tint the square background.
        """
        # Resize board background
        # Note: Some board textures are tiles, some are full boards. 
        # For simplicity, if it's small (< 200px), assume it's a tile and tile it.
        # But our downloader tried to get '150.png', which implies tiles.
        # Let's check size and tile if needed.
        
        target_size = (size, size)
        
        if board_img.width < size:
            # Tile it
            # Assuming the downloaded image is a single square texture?
            # Or maybe it's the whole board but small?
            # From chess.com URLs, /boards/{theme}/150.png is usually a single square texture background 
            # OR the whole board. Let's assume it's the texture for the whole board or a pattern.
            # Actually, standard chess.com board backgrounds are often just repeated textures or a single large image.
            # If it's small, we'll just resize it to cover the board for now, or tile it.
            # Let's try resizing logic:
            bg = board_img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            bg = board_img.resize(target_size, Image.Resampling.LANCZOS)
            
        # Create a new image for the composition
        # If the background is just a texture (like 'wood'), we need to draw the squares?
        # Chess.com board assets usually INCLUDE the light/dark squares if they are 'boards'.
        # But if we downloaded /boards/green/150.png, it might just be the green square?
        # WE NEED TO CHECK THIS. 
        # If the asset is just one color, we might need to generating the checkerboard pattern.
        # For now, let's assume the asset is a full board or we treat it as the "dark" square and white as "light"?
        # Actually, let's look at the file size/content later.
        # For valid checkerboard, we might need to draw it if the image is just a texture.
        
        canvas = bg.copy()

        if highlights:
            for sq_idx, color in highlights:
                canvas = augmentations.add_highlight(
                    canvas,
                    sq_idx,
                    color=color,
                    flipped=flipped,
                )
        
        square_size = size // 8

        # If the background image doesn't look like a board (e.g. it's just a uniform texture),
        # we might want to overlay a checkerboard pattern.
        # Let's assume for now the downloaded simple assets are just textures and we need to draw the squares.
        # But wait, 'green' board usually implies green and white squares.
        # If the image at .../green/150.png is just the green color, we need to construct the board.
        # Let's implement a simple verify step in the main to check this later.
        # I'll implement a fallback checkerboard generator just in case.
        
        # Helper to draw pieces
        for i in range(64):
            # Chess.com square mapping: 0 is a1.
            # If not flipped (White bottom): a1 is bottom left (x=0, y=7)
            # If flipped (Black bottom): a1 is top right (x=7, y=0) ?? 
            # Wait.
            # Standard:
            # Rank 0 (1) -> y=7
            # Rank 7 (8) -> y=0
            # File 0 (a) -> x=0
            # File 7 (h) -> x=7
            
            rank = i // 8
            file = i % 8
            
            if flipped:
                # Rotate 180
                draw_rank = rank
                draw_file = 7 - file
            else:
                draw_rank = 7 - rank
                draw_file = file
                
            piece = board.piece_at(i)
            if piece:
                symbol = piece.symbol() # P, n, b... (upper=white)
                color = 'w' if piece.color == chess.WHITE else 'b'
                key = f"{color}{symbol.lower()}"
                
                if key in pieces:
                    p_img = pieces[key]
                    p_img = p_img.resize((square_size, square_size), Image.Resampling.BILINEAR)
                    
                    x = draw_file * square_size
                    y = draw_rank * square_size
                    
                    # Canvas paste
                    canvas.paste(p_img, (x, y), p_img)
                    
        return canvas


# Import augmentations
import sys
import argparse
import uuid
import csv
import multiprocessing
from functools import partial
# import tqdm

sys.path.append(str(Path(__file__).parent))
import augmentations

# Global variable for the generator instance in worker processes
_generator = None

def init_worker():
    """Initialize the generator in the worker process."""
    global _generator
    # We re-seed random here to ensure different processes don't produce same sequence if they forked
    random.seed() 
    np.random.seed()
    _generator = ChessGenerator()

def generate_single_image(args):
    """
    Worker function to generate a single image.
    args: (output_dir, preview_mode, crop_mode)
    Returns: (filename, fen) or (filename, fen, bbox)
    """
    output_dir, preview_mode, crop_mode = args
    global _generator
    
    if _generator is None:
        init_worker()
        
    try:
        # Always use scrambled/shuffled positions as requested to prevent structure memorization
        is_scrambled = True
        board = _generator.generate_scrambled_fen()
            
        fen = board.fen()
        
        pieces, board_img = _generator.get_random_assets()
        flipped = random.choice([True, False])
        
        # 1. Highlights
        # Track highlights for labels: {sq_idx: label_id}
        # Labels: 1 = Generic Highlight, 2 = Red Highlight
        highlights = {}
        highlight_specs = []

        if random.random() < 0.5:
            num_highlights = random.randint(1, 4)
            for _ in range(num_highlights):
                sq = random.randint(0, 63)
                
                # Decision: Red vs Other.
                if random.random() < 0.3:
                    # Red
                    color = (255, 0, 0)
                    label = 2
                else:
                    # Generic
                    nice_colors = [
                        (255, 255, 0), # Yellow
                        (0, 255, 0),   # Green
                        (0, 0, 255),   # Blue
                        (100, 200, 255) # Light Blue
                    ]
                    color = random.choice(nice_colors)
                    label = 1
                
                highlights[sq] = label
                highlight_specs.append((sq, color))

        img = _generator.render_board(
            board,
            pieces,
            board_img,
            flipped=flipped,
            highlights=highlight_specs,
        )

        # Initial BBox (Full Image)
        bbox = (0, 0, img.width, img.height)

        # Apply random augmentations

        # 0. Coordinate Labels (Always)
        if True:
            img = augmentations.draw_coordinates(img, flipped=flipped)

        # 2. Arrows
        # Track arrows: list of [start, end]
        arrows = []
        if random.random() < 0.7:
            num_arrows = random.randint(1, 12)
            # Select consistent color for all arrows in this image
            # Fixed to Yellow/Orange to simplify learning
            image_arrow_color = (255, 170, 0)
            
            # Helper to get valid knight move destinations from a square
            def get_knight_moves(sq):
                rank, file = sq // 8, sq % 8
                moves = []
                knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                                  (1, -2), (1, 2), (2, -1), (2, 1)]
                for dr, df in knight_offsets:
                    nr, nf = rank + dr, file + df
                    if 0 <= nr < 8 and 0 <= nf < 8:
                        moves.append(nr * 8 + nf)
                return moves
            
            for _ in range(num_arrows):
                # ~15% chance to generate a knight move arrow
                if random.random() < 0.15:
                    start = random.randint(0, 63)
                    knight_dests = get_knight_moves(start)
                    if knight_dests:
                        end = random.choice(knight_dests)
                    else:
                        end = random.randint(0, 63)
                else:
                    start = random.randint(0, 63)
                    end = random.randint(0, 63)
                
                if start != end:
                    thick = random.uniform(0.15, 0.25)
                    
                    arrows.append((start, end))
                    img = augmentations.draw_arrow(img, start, end, color=image_arrow_color, thickness=thick, flipped=flipped)
                
        # 3. Perspective - intentionally skipped or handled via background placement?
        # User commented out perspective in original code.
            
        # 4. Noise/Blur - REMOVED per user request
        # if random.random() < 0.5:
        #     img = augmentations.add_random_artifacts(img)

        # 5. Cursor
        if random.random() < 0.5:
            img = augmentations.add_cursor(img)

        # 6. Background Placement
        if random.random() < 0.4:
            bg_image = None
            if _generator.background_images and random.random() > 0.2: 
                bg_image = random.choice(_generator.background_images)
            img, bbox = augmentations.place_on_background(img, background_image=bg_image)
        
        filename = f"{uuid.uuid4().hex}.png"
        save_path = Path(output_dir) / filename
        
        if crop_mode:
            # Crop Logic with Jitter
            x, y, w, h = bbox
            
            # Add random jitter/expansion to simulate imperfect localization usage
            # Expand by 1-5%
            expansion = random.uniform(0.01, 0.03)
            
            # Random shift
            shift_x = random.uniform(-0.01, 0.01) * w
            shift_y = random.uniform(-0.01, 0.01) * h
            
            x_center = x + w/2 + shift_x
            y_center = y + h/2 + shift_y
            
            new_w = w * (1 + 2*expansion)
            new_h = h * (1 + 2*expansion)
            
            left = int(x_center - new_w/2)
            top = int(y_center - new_h/2)
            right = int(x_center + new_w/2)
            bottom = int(y_center + new_h/2)
            
            # Clamp to image bounds? Or fill? 
            # Pillow crop handles out of bounds by filling with 0 if we assume standard crop, 
            # but actually it just clamps context.
            # Ideally we want to pad if out of bounds.
            # Let's crop intersection then pad.
            
            # Simple clamp for now
            left = max(0, left)
            top = max(0, top)
            right = min(img.width, right)
            bottom = min(img.height, bottom)
            
            if right <= left or bottom <= top:
                 # Fallback
                 output_img = img
            else:
                 output_img = img.crop((left, top, right, bottom))
                 
            # Resize Max 256
            max_size = 256
            w_c, h_c = output_img.size
            scale = max_size / max(w_c, h_c)
            new_w_c, new_h_c = int(w_c * scale), int(h_c * scale)
            output_img = output_img.resize((new_w_c, new_h_c), Image.Resampling.BICUBIC)
            
            # Pad to Square 256
            final_img = Image.new("RGB", (max_size, max_size), (0, 0, 0))
            final_img.paste(output_img, ((max_size - new_w_c) // 2, (max_size - new_h_c) // 2))
            
            # Use compress_level=1 for faster saving (default is 6)
            final_img.save(save_path, compress_level=1)

            # Dump dicts to json string string for CSV
            return (filename, fen, json.dumps(highlights), json.dumps(arrows), flipped)
            
        else:
            # Standard Save, fast compression
            img.save(save_path, compress_level=1)
            return (filename, fen, bbox, json.dumps(highlights), json.dumps(arrows), flipped)
        
    except Exception as e:
        print(f"Error in worker: {e}")
        return None

def generate_dataset(count, output_dir, preview_mode=False, crop_mode=False):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check existing files
    if not preview_mode:
        existing_files = list(output_dir.glob("*.png"))
        num_existing = len(existing_files)
        print(f"Found {num_existing} existing images in {output_dir}")
        
        if num_existing >= count:
            print(f"Already have {num_existing} images (>= {count}). Stopping.")
            return

        needed = count - num_existing
        print(f"generating {needed} more images...")
    else:
        needed = count
        print(f"Generating {needed} preview images...")

    csv_mode = 'a' if (output_dir / "labels.csv").exists() else 'w'
    if not preview_mode:
        csv_file = open(output_dir / "labels.csv", csv_mode, newline="")
        writer = csv.writer(csv_file)
        if csv_mode == 'w':
            if crop_mode:
                writer.writerow(["filename", "fen", "highlights", "arrows", "is_flipped"])
            else:
                writer.writerow(["filename", "fen", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "highlights", "arrows", "is_flipped"])
    
    # Define processing logic
    def process_batch(results):
        rows_to_write = []
        for result in results:
            if not result: continue
            
            if crop_mode:
                filename, fen, hl, arr, flipped = result
                if not preview_mode:
                    simple_fen = fen.split(" ")[0]
                    rows_to_write.append([filename, simple_fen, hl, arr, flipped])
            else:
                filename, fen, bbox, hl, arr, flipped = result
                # User requested simplified FEN (pieces only)
                simple_fen = fen.split(" ")[0]
                if not preview_mode:
                    x, y, w, h = bbox
                    rows_to_write.append([filename, simple_fen, x, y, w, h, hl, arr, flipped])
        
        if rows_to_write:
            writer.writerows(rows_to_write)

    # Determine cpu count
    #num_workers = multiprocessing.cpu_count()
    num_workers = 6 # Force single process to debug crash
    
    pool_args = [(output_dir, preview_mode, crop_mode) for _ in range(needed)]

    # Batching config
    BATCH_SIZE = 1000
    buffer = []

    if num_workers > 1:
        print(f"Using {num_workers} worker processes.")
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
            # Use imap_unordered for better responsiveness
            iterator = pool.imap_unordered(generate_single_image, pool_args, chunksize=10)
            
            for result in iterator:
                 buffer.append(result)
                 if len(buffer) >= BATCH_SIZE:
                     process_batch(buffer)
                     buffer = []
            
            # Process remaining
            if buffer:
                process_batch(buffer)
                
    else:
        print("Running in single process mode (Debug)")
        init_worker()
        for arg in pool_args:
            result = generate_single_image(arg)
            buffer.append(result)
            if len(buffer) >= BATCH_SIZE:
                process_batch(buffer)
                buffer = []
        
        if buffer:
            process_batch(buffer)
            
    if not preview_mode:
        csv_file.close()
            
    if not preview_mode:
        csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic chess board data.")
    parser.add_argument("--count", type=int, default=10, help="Total number of images desired")
    parser.add_argument("--output", type=str, default="output/dataset", help="Output directory")
    parser.add_argument("--preview", action="store_true", help="Preview mode (human readable filenames, no CSV)")
    parser.add_argument("--crop", default=True, action="store_true", help="Directly save cropped board images (streamlined pipeline)")
    
    args = parser.parse_args()
    
    # If preview mode default to preview dir logic else user dir
    out_dir = args.output
    if args.preview and args.output == "output/dataset":
         out_dir = "output/preview"
         
    # On Windows, multiprocessing needs freeze_support check if using cx_Freeze etc, 
    # but strictly speaking simple scripts are fine if under if __name__ == "__main__".
    # multiprocessing.freeze_support() 
    
    generate_dataset(args.count, out_dir, preview_mode=args.preview, crop_mode=args.crop)

