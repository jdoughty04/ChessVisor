import os
import requests
import time
from pathlib import Path

# Configuration
ASSETS_DIR = Path("assets")
PIECES_DIR = ASSETS_DIR / "pieces"
BOARDS_DIR = ASSETS_DIR / "boards"

# Known Chess.com themes (subset)
PIECE_THEMES = [
    "neo", "wood", "glass", "alpha", "game_room",
    "icy_sea", "graffiti", "bubblegum", "classic", "bases",
    "nature", "neon", "ocean", "space", "gothic"
]

BOARD_THEMES = [
    "green", "brown", "blue", "overlay", "translucent",
    "wood", "newspaper", "bases", "8-bit", "marble",
    "purple", "icy_sea", "metal", "walnut", "parchment"
]

PIECES = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]

# Base URLs
# Confirmed piece URL: https://www.chess.com/chess-themes/pieces/neo/150/wp.png
PIECE_URL_TEMPLATE = "https://www.chess.com/chess-themes/pieces/{theme}/150/{piece}.png"

# Potential Board URLs - we try a few specific patterns if one fails, but standard is similar
BOARD_URL_TEMPLATES = [
    "https://www.chess.com/chess-themes/boards/{theme}/150.png",
    "https://images.chesscomfiles.com/chess-themes/boards/{theme}/150.png"
]

def ensure_dirs():
    os.makedirs(PIECES_DIR, exist_ok=True)
    os.makedirs(BOARDS_DIR, exist_ok=True)

def download_file(url, filepath):
    if filepath.exists():
        print(f"Skipping {filepath.name}, already exists.")
        return True
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filepath.name}")
            return True
        else:
            print(f"Failed {url} - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_pieces():
    print("Downloading pieces...")
    for theme in PIECE_THEMES:
        theme_dir = PIECES_DIR / theme
        os.makedirs(theme_dir, exist_ok=True)
        
        for piece in PIECES:
            url = PIECE_URL_TEMPLATE.format(theme=theme, piece=piece)
            filepath = theme_dir / f"{piece}.png"
            download_file(url, filepath)
            # Be nice to the server
            # time.sleep(0.05) 

def download_boards():
    print("Downloading boards...")
    for theme in BOARD_THEMES:
        # Some board filenames might just be the theme name .png? 
        # Usually it's a tiled image or a single board image.
        # Let's try downloading '150.png' which is usually the tile size or board size.
        # Check standard behavior.
        
        filepath = BOARDS_DIR / f"{theme}.png"
        
        success = False
        for template in BOARD_URL_TEMPLATES:
            url = template.format(theme=theme)
            if download_file(url, filepath):
                success = True
                break
        
        if not success:
            print(f"Could not find board for theme: {theme}")

def download_backgrounds():
    print("Downloading backgrounds...")
    bg_dir = ASSETS_DIR / "backgrounds"
    os.makedirs(bg_dir, exist_ok=True)
    
    # Download 10 random images
    for i in range(10):
        url = f"https://picsum.photos/1024/768?random={i}"
        filepath = bg_dir / f"bg_{i}.jpg"
        download_file(url, filepath)

if __name__ == "__main__":
    ensure_dirs()
    download_pieces()
    download_boards()
    download_backgrounds()

