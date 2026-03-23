import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def draw_coordinates(img_pil, flipped=False):
    """
    Draw file (a-h) and rank (1-8) labels on the borders.
    Mimics chess.com / lichess style.
    flipped: If True, Board is from Black's perspective.
    """
    w, h = img_pil.size
    sq_w = w / 8
    sq_h = h / 8
    
    draw = ImageDraw.Draw(img_pil, "RGBA")
    
    # Font settings
    # Try to load Arial or similar, else default
    try:
        font_size = int(sq_h * 0.25) # 25% of square height
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Determine labels
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    if flipped:
        files = list(reversed(files)) # h, g, f...
        # Ranks: top is 1, bottom is 8.
        # Wait. Black perspective:
        # Top-left is h1.
        # Bottom-left is h8.
        # So ranks go 1 -> 8 from Top -> Bottom
        ranks = ranks # 1..8
    else:
        # Standard: White perspective. 
        # Rank 8 at top, Rank 1 at bottom.
        ranks = list(reversed(ranks)) # 8..1
    
    # Colors (Usually contrasting with the square color)
    # But coordinate labels are often on the edge squares themselves.
    # Light squares get dark text, Dark squares get light text.
    # Color logic:
    # A1 (Standard) is Dark (Black square).
    # H1 is Light.
    # Rank 1: A1(D), B1(L), C1(D)...
    # Rank index (from top 0..7):
    #   if (r+f) % 2 == 0: Light square
    #   else: Dark square
    # Text Color: if square is Light -> Dark Text. If Dark -> Light Text.
    
    for i in range(8):
        # Draw Files (at the bottom row)
        # y position: bottom edge of the image
        text = files[i]
        
        # Position: Top-Left of the text
        # x: cell_x + padding
        # y: height - padding
        
        # Determine square color of bottom-left corner of this file's square
        # Bottom row index is 7.
        # Square (row=7, col=i)
        # Is (7 + i) even? -> Light
        is_light = (7 + i) % 2 == 0
        text_color = (100, 100, 100) if is_light else (220, 220, 220)
        
        # Draw on bottom-right of the square (chess.com style)
        # x = (i + 1) * sq_w - font_size
        # y = h - font_size
        
        draw.text((i * sq_w + sq_w * 0.8, h - sq_h * 0.25), text, fill=text_color, font=font)
        
        # Draw Ranks (at the right column? or left?)
        # Let's draw on Left column for Ranks (or Right?)
        # Chess.com standard: Files (letters) on bottom right of squares. Ranks (numbers) on top left of squares.
        # Let's verify style. 
        # Actually usually: Ranks on Right (8..1) or Left.
        # Let's stick to: Files on Bottom margin, Ranks on Left margin.
        # Since we don't have margins, we draw ON the squares.
        # Ranks: Text on top-left of the square in the FIRST column (i=0) ?
        
        # Let's draw ranks on the Left-most column (File 'a' or 'h').
        text = ranks[i] # i is row index 0..7
        
        # Square (row=i, col=0)
        is_light = (i + 0) % 2 == 0
        text_color = (100, 100, 100) if is_light else (220, 220, 220)
        
        draw.text((sq_w * 0.05, i * sq_h + sq_h * 0.05), text, fill=text_color, font=font)
        
    return img_pil

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def add_perspective(img_pil, max_distortion=0.2):
    """
    Apply slight random perspective warp to the image.
    max_distortion: percentage of image size to shift corners.
    """
    img = pil_to_cv2(img_pil)
    rows, cols = img.shape[:2]
    
    # Source points: the 4 corners
    src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    
    # Destination points: Shifted corners
    # We want valid perspective, so we shift corners randomly
    w_shift = cols * max_distortion
    h_shift = rows * max_distortion
    
    # Random shifts
    # Prevent crossing
    pt1 = [random.uniform(0, w_shift), random.uniform(0, h_shift)]
    pt2 = [cols - random.uniform(0, w_shift), random.uniform(0, h_shift)]
    pt3 = [random.uniform(0, w_shift), rows - random.uniform(0, h_shift)]
    pt4 = [cols - random.uniform(0, w_shift), rows - random.uniform(0, h_shift)]
    
    dst_points = np.float32([pt1, pt2, pt3, pt4])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Use a border color or replicate?
    # Replicate might look weird if the shift is large.
    # Using a background color (like dark grey) is safer.
    warped = cv2.warpPerspective(img, matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))
    
    return cv2_to_pil(warped)

def _is_knight_move(start_sq, end_sq):
    """Check if the move from start_sq to end_sq is a knight's L-shape pattern."""
    start_rank = start_sq // 8
    start_file = start_sq % 8
    end_rank = end_sq // 8
    end_file = end_sq % 8
    
    rank_diff = abs(end_rank - start_rank)
    file_diff = abs(end_file - start_file)
    
    # Knight moves in an L-shape: 2 squares one direction, 1 square perpendicular
    return (rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2)


def _draw_straight_arrow(img, p_start_center, p_end, sq_size, color, thickness):
    """Draw a straight arrow from p_start_center towards p_end."""
    # Vector from start to end
    v = p_end - p_start_center
    length = np.linalg.norm(v)
    if length < 1: 
        return img
    
    # Normalize vector
    u = v / length

    # Offset start point
    offset_scale = random.uniform(0.4, 0.45)
    offset = u * (sq_size * offset_scale)
    p_start = p_start_center + offset
    
    # Re-calculate vector and length from new start
    v = p_end - p_start
    length = np.linalg.norm(v)
    if length < 1:
        return img
    u = v / length
    
    # Perpendicular vector
    u_perp = np.array([-u[1], u[0]])
    
    # Dimensions
    shaft_w = sq_size * thickness
    # Equilateral triangle arrowhead: side = 2 * shaft_w, height = side * sqrt(3)/2
    head_side = shaft_w * 2
    head_len = head_side * (3 ** 0.5) / 2  # height of equilateral triangle
    head_w = head_side  # base width equals side length
    
    # If arrow is too short, adjust head
    if length < head_len:
        head_len = length * 0.8
        head_w = head_len * 1.0

    # Points for the polygon
    p1 = p_start + u_perp * (shaft_w / 2)
    p2 = p_start - u_perp * (shaft_w / 2)
    
    p_neck = p_end - u * head_len
    p3 = p_neck - u_perp * (shaft_w / 2)
    p4 = p_neck + u_perp * (shaft_w / 2)
    
    p5 = p_neck - u_perp * (head_w / 2)
    p6 = p_end
    p7 = p_neck + u_perp * (head_w / 2)
    
    pts = np.array([p1, p2, p3, p5, p6, p7, p4], np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    cv2.fillPoly(img, [pts], color)
    return img


def _draw_l_shaped_arrow(img, p_start_center, p_end, sq_size, color, thickness, horizontal_first=True):
    """Draw an L-shaped arrow from p_start_center to p_end for knight moves."""
    shaft_w = sq_size * thickness
    # Equilateral triangle arrowhead: side = 2 * shaft_w, height = side * sqrt(3)/2
    head_side = shaft_w * 2
    head_len = head_side * (3 ** 0.5) / 2  # height of equilateral triangle
    head_w = head_side  # base width equals side length
    
    # Calculate the corner point of the L
    if horizontal_first:
        # Move horizontally first, then vertically
        p_corner = np.array([p_end[0], p_start_center[1]], dtype=float)
    else:
        # Move vertically first, then horizontally
        p_corner = np.array([p_start_center[0], p_end[1]], dtype=float)
    
    # --- First leg (from start to corner) ---
    v1 = p_corner - p_start_center
    len1 = np.linalg.norm(v1)
    if len1 < 1:
        # Fallback to straight arrow
        return _draw_straight_arrow(img, p_start_center, p_end, sq_size, color, thickness)
    
    u1 = v1 / len1
    u1_perp = np.array([-u1[1], u1[0]])
    
    # Offset start point
    offset_scale = random.uniform(0.4, 0.45)
    offset = u1 * (sq_size * offset_scale)
    p_start = p_start_center + offset
    
    # First leg corners
    leg1_p1 = p_start + u1_perp * (shaft_w / 2)
    leg1_p2 = p_start - u1_perp * (shaft_w / 2)
    leg1_p3 = p_corner - u1_perp * (shaft_w / 2)
    leg1_p4 = p_corner + u1_perp * (shaft_w / 2)
    
    # --- Second leg (from corner to end) ---
    v2 = p_end - p_corner
    len2 = np.linalg.norm(v2)
    if len2 < 1:
        # Fallback to straight arrow
        return _draw_straight_arrow(img, p_start_center, p_end, sq_size, color, thickness)
    
    u2 = v2 / len2
    u2_perp = np.array([-u2[1], u2[0]])
    
    # Second leg - adjust for arrowhead
    if len2 < head_len:
        head_len = len2 * 0.8
        head_w = head_len * 1.0
    
    p_neck = p_end - u2 * head_len
    
    # Second leg corners (starting from corner)
    leg2_p1 = p_corner + u2_perp * (shaft_w / 2)
    leg2_p2 = p_corner - u2_perp * (shaft_w / 2)
    leg2_p3 = p_neck - u2_perp * (shaft_w / 2)
    leg2_p4 = p_neck + u2_perp * (shaft_w / 2)
    
    # Arrowhead points
    head_p1 = p_neck - u2_perp * (head_w / 2)
    head_p2 = p_end
    head_p3 = p_neck + u2_perp * (head_w / 2)
    
    # Draw the first leg as a rectangle
    leg1_pts = np.array([leg1_p1, leg1_p2, leg1_p3, leg1_p4], np.int32)
    leg1_pts = leg1_pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [leg1_pts], color)
    
    # Draw the second leg with arrowhead
    leg2_pts = np.array([leg2_p1, leg2_p2, leg2_p3, head_p1, head_p2, head_p3, leg2_p4], np.int32)
    leg2_pts = leg2_pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [leg2_pts], color)
    
    # Fill the corner to avoid gaps
    corner_pts = np.array([
        p_corner + u1_perp * (shaft_w / 2) + u2_perp * (shaft_w / 2),
        p_corner + u1_perp * (shaft_w / 2) - u2_perp * (shaft_w / 2),
        p_corner - u1_perp * (shaft_w / 2) - u2_perp * (shaft_w / 2),
        p_corner - u1_perp * (shaft_w / 2) + u2_perp * (shaft_w / 2),
    ], np.int32)
    corner_pts = corner_pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [corner_pts], color)
    
    return img


def draw_arrow(img_pil, start_sq, end_sq, color=(0, 255, 0), thickness=0.2, tip_size=0.3, flipped=False):
    """
    Draw a chess.com-style thick arrow from start_sq to end_sq.
    For knight moves (L-shaped patterns), draws an L-shaped arrow.
    thickness: fraction of square size for the shaft width (default 0.2 ~ 20%)
    tip_size: fraction of square size for the tip width (relative variation)
    color: RGB tuple
    """
    img = np.array(img_pil)
    rows, cols = img.shape[:2]
    sq_size = rows // 8
    
    def get_center(sq_idx):
        rank = sq_idx // 8
        file = sq_idx % 8
        if flipped:
            r = rank
            f = 7 - file
        else:
            r = 7 - rank
            f = file
        return np.array([int(f * sq_size + sq_size / 2), int(r * sq_size + sq_size / 2)])

    p_start_center = get_center(start_sq)
    p_end = get_center(end_sq)
    
    # Check for zero-length arrows
    v = p_end - p_start_center
    length = np.linalg.norm(v)
    if length < 1: 
        return img_pil
    
    # Create overlay for transparency
    overlay = img.copy()
    
    # Check if this is a knight move pattern
    if _is_knight_move(start_sq, end_sq):
        # Draw L-shaped arrow for knight moves
        # Randomly choose horizontal-first or vertical-first
        horizontal_first = random.choice([True, False])
        overlay = _draw_l_shaped_arrow(overlay, p_start_center, p_end, sq_size, color, thickness, horizontal_first)
    else:
        # Draw straight arrow
        overlay = _draw_straight_arrow(overlay, p_start_center, p_end, sq_size, color, thickness)
    
    # Blend: chess.com arrows are quite transparent, maybe 0.5 - 0.7
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return Image.fromarray(img)

def add_cursor(img_pil):
    """
    Add a mouse cursor at a random position.
    """
    img = pil_to_cv2(img_pil)
    rows, cols = img.shape[:2]
    
    # Random position
    x = random.randint(0, cols)
    y = random.randint(0, rows)
    
    # Define cursor shape (standard pointer)
    # Relative size to image
    cursor_size = max(20, int(cols * 0.05))
    
    # Simple polygon for pointer
    # Tip at (0,0), then (0, 1), (0.7, 0.7), ...
    # This is a bit simplified. A standard cursor is a triangle with a tail.
    # Vertices relative to tip (0,0) with scale 1.0
    # (0,0) -> (0, 1) -> (0.3, 0.8) -> (0.5, 1.2) ... it's tricky to draw manually perfectly.
    # Let's simple triangle pointer: (0,0) -> (0, 1) -> (0.7, 0.7)
    
    pts_rel = np.array([
        [0, 0],
        [0, 1],
        [0.25, 0.65], # Notch start
        [0.55, 0.95], # Tail end 1
        [0.65, 0.85], # Tail end 2
        [0.35, 0.55], # Notch end
        [0.75, 0.55]  # Right corner
    ])
    
    # Scale and move
    pts = (pts_rel * cursor_size).astype(np.int32)
    pts[:, 0] += x
    pts[:, 1] += y
    
    # Draw outline (white) and fill (black) or vice versa. standard is white fill black outline usually?
    # Windows default: White fill, black outline.
    
    # Shadow?
    # No shadow for simplicity for now.
    
    # Outline is slightly larger
    cv2.fillPoly(img, [pts], (255, 255, 255)) # Fill white
    cv2.polylines(img, [pts], True, (0, 0, 0), 1) # Outline black
    
    return cv2_to_pil(img)

def place_on_background(img_pil, background_image=None):
    """
    Place the board on a larger background, potentially scaling it down.
    Mimics seeing the board in a full browser window or desktop.
    background_image: Optional PIL Image to use as background.
    """
    board_img = np.array(img_pil)
    h, w = board_img.shape[:2]
    
    # Random scaling
    scale = random.uniform(0.5, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if scale < 1.0:
        board_img = cv2.resize(board_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Target size (same as input to preserve dimensions)
    bg_h, bg_w = h, w 
    
    # Create background
    if background_image:
        # Use provided natural image
        bg_pil = background_image
        # If background is too small, resize it
        if bg_pil.width < bg_w or bg_pil.height < bg_h:
             bg_pil = bg_pil.resize((max(bg_w, bg_pil.width), max(bg_h, bg_pil.height)))
        
        # Random crop from background
        max_crop_x = bg_pil.width - bg_w
        max_crop_y = bg_pil.height - bg_h
        
        crop_x = random.randint(0, max(0, max_crop_x))
        crop_y = random.randint(0, max(0, max_crop_y))
        
        bg_crop = bg_pil.crop((crop_x, crop_y, crop_x + bg_w, crop_y + bg_h))
        background = np.array(bg_crop)
        
        # Ensure it has 3 channels (remove alpha if present)
        if background.shape[-1] == 4:
            background = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)
        elif len(background.shape) == 2: # Gray
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
            
    else:
        # Synthetic noise background
        bg_color = [random.randint(0, 255) for _ in range(3)]
        background = np.full((bg_h, bg_w, 3), bg_color, dtype=np.uint8)
        
        # Maybe noise background
        if random.random() > 0.5:
            noise = np.random.randint(0, 255, (bg_h, bg_w, 3), dtype=np.uint8)
            cv2.addWeighted(background, 0.7, noise, 0.3, 0, background)
        
    # Place board at random offset
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    
    off_x = random.randint(0, max(0, max_x))
    off_y = random.randint(0, max(0, max_y))
    
    # Paste
    # Ensure background is writable and correct format
    # PIL/Numpy interaction is sometimes tricky with read-only buffers
    background = background.copy()
    
    # Handle alpha channel of board?
    # Usually board_img is RGB from pil_to_cv2 conversion unless we changed it.
    # augmentations.py uses pil_to_cv2 which usually returns BGR?
    # Wait, pil_to_cv2 implementation: return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # So `board_img` is BGR.
    # `background` from PIL is RGB (np.array(bg_crop)).
    # We need consistency.
    
    # Let's work in BGR for cv2 operations.
    if background_image:
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        # Convert board to BGR to match background
        board_img = cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR)
        
    # Paste
    background[off_y:off_y+new_h, off_x:off_x+new_w] = board_img
    
    return cv2_to_pil(background), (off_x, off_y, new_w, new_h)

def add_highlight(img_pil, sq_idx, color=(255, 255, 0), alpha=0.5, flipped=False):
    """
    Highlight a square with a transparent color.
    color: RGB tuple
    Apply this to the board/background layer before pieces are composited
    if you need occupied squares to keep the piece pixels untouched.
    """
    img = np.array(img_pil)
    rows, cols = img.shape[:2]
    sq_size = rows // 8
    
    rank = sq_idx // 8
    file = sq_idx % 8
    
    if flipped:
        r = rank
        f = 7 - file
    else:
        r = 7 - rank
        f = file
        
    x = int(f * sq_size)
    y = int(r * sq_size)
    
    # Create overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + sq_size, y + sq_size), color, -1)
    
    # Blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return Image.fromarray(img)

def add_random_artifacts(img_pil):
    """Add noise, blur, or jpeg artifacts."""
    img = np.array(img_pil)
    
    # Random Blur
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    # Random Noise
    if random.random() > 0.5:
        row, col, ch = img.shape
        mean = 0
        var = random.uniform(0.01, 0.05)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        # Add and clip
        noisy = img + gauss * 255
        img = np.clip(noisy, 0, 255).astype(np.uint8)
        
    return Image.fromarray(img)
