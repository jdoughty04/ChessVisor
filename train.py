import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import wandb
from PIL import Image

from dataset import ChessDataset
from model import ChessNet

import json

class LiveWeightManager:
    """Reads loss weights from a JSON file for live adjustment during training."""
    
    DEFAULT_WEIGHTS = {
        'pieces': 1.0,
        'highlights': 1.0,
        'arrows': 3.0,
        'perspective': 1.0,
        'learning_rate': 1e-4,
        'max_grad_norm': 10.0
    }
    
    def __init__(self, weight_file, refresh_interval=10, normalize_weights=True):
        self.weight_file = weight_file
        self.refresh_interval = refresh_interval
        self.normalize_weights = normalize_weights
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.batch_count = 0
        self._last_mtime = 0
        
        # Initialize file if it doesn't exist
        if not os.path.exists(weight_file):
            self._save_weights()
        else:
            self._load_weights()
    
    def _load_weights(self):
        """Load weights from JSON file."""
        try:
            mtime = os.path.getmtime(self.weight_file)
            if mtime != self._last_mtime:
                with open(self.weight_file, 'r') as f:
                    loaded = json.load(f)
                    for key in self.DEFAULT_WEIGHTS:
                        if key in loaded:
                            self.weights[key] = float(loaded[key])
                self._last_mtime = mtime
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # Keep current weights if file is invalid
    
    def _save_weights(self):
        """Save current weights to JSON file."""
        with open(self.weight_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
        self._last_mtime = os.path.getmtime(self.weight_file)
    
    def get_weights(self):
        """Get current weights, refreshing from file periodically."""
        self.batch_count += 1
        if self.batch_count % self.refresh_interval == 0:
            self._load_weights()
        return self.weights.copy()
    
    def compute_loss(self, losses_dict):
        """
        Compute normalized total loss.
        losses_dict: {name: loss_tensor}
        Returns: normalized total loss, dict of weights used
        """
        weights = self.get_weights()
        
        weighted_sum = 0.0
        for name, loss in losses_dict.items():
            w = weights.get(name, 1.0)
            weighted_sum = weighted_sum + loss * w
        
        # Normalize by sum of weights (excluding learning_rate) if enabled
        if self.normalize_weights:
            loss_weights = {k: v for k, v in weights.items() if k != 'learning_rate'}
            weight_sum = sum(loss_weights.values())
            total_loss = weighted_sum / max(weight_sum, 1e-6)
        else:
            total_loss = weighted_sum
        
        return total_loss, weights
    
    def get_learning_rate(self):
        """Get current learning rate from weights."""
        return self.weights.get('learning_rate', 1e-4)
    
    def get_max_grad_norm(self):
        """Get current max gradient norm from weights."""
        return self.weights.get('max_grad_norm', 10.0)

class ResizeMax:
    """Resize the image so the largest side is max_size, preserving aspect ratio."""
    def __init__(self, max_size=256):
        self.max_size = max_size
        
    def __call__(self, img):
        w, h = img.size
        scale = self.max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.BICUBIC)

class PadToSquare:
    """Pad the image to be a square of side `size` with `fill` color."""
    def __init__(self, size=256, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill
        
    def __call__(self, img):
        w, h = img.size
        new_img = Image.new("RGB", (self.size, self.size), self.fill)
        # Paste in center
        new_img.paste(img, ((self.size - w) // 2, (self.size - h) // 2))
        return new_img

# File paths for GUI communication
EVAL_FLAG_FILE = "run_eval.flag"
EVAL_RESULTS_FILE = "eval_results.json"

def evaluate_dataset(model, data_loader, device, criterion_pieces, criterion_highlights, 
                     criterion_arrows, criterion_perspective):
    """Evaluate model on a dataset and return metrics dict."""
    model.eval()
    
    acc_pieces = 0.0
    acc_boards = 0.0
    acc_highlights = 0.0
    acc_perspective = 0.0
    arrow_tp, arrow_fp, arrow_fn = 0, 0, 0
    highlight_tp, highlight_fp, highlight_fn = 0, 0, 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            l_p = batch['pieces'].to(device)
            l_h = batch['highlights'].to(device)
            l_a = batch['arrows'].to(device)
            l_f = batch['flipped'].to(device)
            
            o_p, o_h, o_a, o_f = model(images)
            
            # Loss
            loss_p = criterion_pieces(o_p.permute(0, 2, 1), l_p)
            loss_h = criterion_highlights(o_h.permute(0, 2, 1), l_h)
            loss_a = criterion_arrows(o_a, l_a)
            loss_f = criterion_perspective(o_f, l_f)
            total_loss += (loss_p + loss_h + loss_a + loss_f).item()
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Pieces
            _, preds_p = torch.max(o_p, 2)
            acc_pieces += (preds_p == l_p).sum().item() / 64.0
            acc_boards += (preds_p == l_p).all(dim=1).sum().item()
            
            # Highlights
            _, preds_h = torch.max(o_h, 2)
            acc_highlights += (preds_h == l_h).sum().item() / 64.0
            
            preds_h_binary = (preds_h > 0).float()
            l_h_binary = (l_h > 0).float()
            highlight_tp += (preds_h_binary * l_h_binary).sum().item()
            highlight_fp += (preds_h_binary * (1 - l_h_binary)).sum().item()
            highlight_fn += ((1 - preds_h_binary) * l_h_binary).sum().item()
            
            # Arrows
            probs_a = torch.sigmoid(o_a)
            preds_a = (probs_a > 0.5).float()
            arrow_tp += (preds_a * l_a).sum().item()
            arrow_fp += (preds_a * (1 - l_a)).sum().item()
            arrow_fn += ((1 - preds_a) * l_a).sum().item()
            
            # Perspective
            preds_f = (torch.sigmoid(o_f) > 0.5).float()
            acc_perspective += (preds_f == l_f).sum().item()
    
    epsilon = 1e-7
    metrics = {
        'piece_acc': acc_pieces / total_samples,
        'board_acc': acc_boards / total_samples,
        'highlight_acc': acc_highlights / total_samples,
        'perspective_acc': acc_perspective / total_samples,
        'arrow_precision': arrow_tp / (arrow_tp + arrow_fp + epsilon),
        'arrow_recall': arrow_tp / (arrow_tp + arrow_fn + epsilon),
        'highlight_precision': highlight_tp / (highlight_tp + highlight_fp + epsilon),
        'highlight_recall': highlight_tp / (highlight_tp + highlight_fn + epsilon),
        'loss': total_loss / len(data_loader) if len(data_loader) > 0 else 0
    }
    
    # Add F1 scores
    metrics['arrow_f1'] = 2 * metrics['arrow_precision'] * metrics['arrow_recall'] / (metrics['arrow_precision'] + metrics['arrow_recall'] + epsilon)
    metrics['highlight_f1'] = 2 * metrics['highlight_precision'] * metrics['highlight_recall'] / (metrics['highlight_precision'] + metrics['highlight_recall'] + epsilon)
    
    return metrics

def check_and_run_eval(model, test_loader, device, criterions, batch_idx):
    """Check for eval flag and run evaluation if requested. Returns True if eval was run."""
    if batch_idx % 10 != 0:  # Only check every 10 batches
        return False
    
    if not os.path.exists(EVAL_FLAG_FILE):
        return False
    
    # Remove flag
    try:
        os.remove(EVAL_FLAG_FILE)
    except:
        pass
    
    print("\n[On-demand eval requested - running...]")
    
    metrics = evaluate_dataset(
        model, test_loader, device,
        criterions['pieces'], criterions['highlights'],
        criterions['arrows'], criterions['perspective']
    )
    
    # Write results to JSON
    with open(EVAL_RESULTS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[Eval complete] Board Acc: {metrics['board_acc']:.4f}, Arrow F1: {metrics['arrow_f1']:.4f}")
    
    return True

def train(args):
    # Initialize WandB
    wandb.init(project=args.wandb_project, config=vars(args))
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    # Resize to standard size (e.g. 256x256) but preserve aspect ratio (letterbox)
    transform = transforms.Compose([
        ResizeMax(256),
        PadToSquare(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    print(f"Loading dataset from {args.data_dir}...")
    try:
        full_dataset = ChessDataset(args.data_dir, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if len(full_dataset) == 0:
        print("Dataset is empty. Did you run metadata generation?")
        return
        
    # Split: 10% val, 1% test (from remaining 90%)
    val_size = int(len(full_dataset) * 0.1)
    test_size = int(len(full_dataset) * 0.01)
    train_size = len(full_dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(args.seed))
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = ChessNet(pretrained=args.pretrained).to(device)
    
    # Load checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        try:
            state_dict = torch.load(args.resume, map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

    # Log model gradients and topology
    wandb.watch(model, log="all")
    
    criterion_pieces = nn.CrossEntropyLoss()
    criterion_highlights = nn.CrossEntropyLoss()
    # Add pos_weight to handle clean imbalance (sparsity) of arrows
    # Approx 1-3 positives per 4096 squares => weight ~100-200
    pos_weight = torch.tensor([1.0]).to(device)
    criterion_arrows = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_perspective = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Live weight manager for real-time adjustment
    weight_manager = LiveWeightManager(
        args.weight_file, 
        refresh_interval=10, 
        normalize_weights=not args.no_weight_norm
    )
    print(f"Using live weights from: {args.weight_file}")
    print(f"Weight normalization: {'disabled' if args.no_weight_norm else 'enabled'}")
    print(f"Initial weights: {weight_manager.weights}")
    
    # Criterions dict for on-demand eval
    criterions = {
        'pieces': criterion_pieces,
        'highlights': criterion_highlights,
        'arrows': criterion_arrows,
        'perspective': criterion_perspective
    }
    
    best_acc = 0.0
    batch_counter = 0  # Global batch counter for eval trigger
    
    # If loading pretrained, maybe valid first? Or just start training.
    # We'll just start training loop.

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Tqdm loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels_pieces = batch['pieces'].to(device)
            labels_highlights = batch['highlights'].to(device)
            labels_arrows = batch['arrows'].to(device)
            labels_flipped = batch['flipped'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            out_p, out_h, out_a, out_f = model(images)
            
            # Losses
            # Pieces: (B, 64, 13) -> (B, 13, 64)
            loss_p = criterion_pieces(out_p.permute(0, 2, 1), labels_pieces)
            
            # Highlights: (B, 64, 3) -> (B, 3, 64)
            loss_h = criterion_highlights(out_h.permute(0, 2, 1), labels_highlights)
            
            # Arrows: (B, 64, 64) matches labels (B, 64, 64)
            loss_a = criterion_arrows(out_a, labels_arrows)
            
            # Perspective: (B, 1) matches labels (B, 1)
            loss_f = criterion_perspective(out_f, labels_flipped)

            # Total Loss (Weighted sum with live weights)
            losses_dict = {
                'pieces': loss_p,
                'highlights': loss_h,
                'arrows': loss_a,
                'perspective': loss_f
            }
            loss, current_weights = weight_manager.compute_loss(losses_dict)
            
            # Update learning rate from GUI
            new_lr = weight_manager.get_learning_rate()
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            loss.backward()

            # Clip gradients and get total norm
            max_norm = weight_manager.get_max_grad_norm()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            # Log training loss & arrow stats
            with torch.no_grad():
                a_sig = torch.sigmoid(out_a)
                log_dict = {
                    "train_loss": loss.item(),
                    "grad_norm": total_norm,
                    "loss_pieces": loss_p.item(),
                    "loss_highlights": loss_h.item(),
                    "loss_arrows": loss_a.item(),
                    "loss_flipped": loss_f.item(),
                    "arrow_logit_max": out_a.max().item(),
                    "arrow_logit_min": out_a.min().item(),
                    "arrow_prob_mean": a_sig.mean().item(),
                    "arrow_prob_max": a_sig.max().item()
                }
                # Log current weights
                log_dict["weight_pieces"] = current_weights['pieces']
                log_dict["weight_highlights"] = current_weights['highlights']
                log_dict["weight_arrows"] = current_weights['arrows']
                log_dict["weight_perspective"] = current_weights['perspective']
                log_dict["learning_rate_live"] = new_lr
                wandb.log(log_dict)
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
            # Check for on-demand evaluation
            batch_counter += 1
            if check_and_run_eval(model, test_loader, device, criterions, batch_counter):
                model.train()  # Switch back to training mode
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        
        # Metrics accumulators
        acc_pieces = 0.0
        acc_boards = 0.0
        acc_highlights = 0.0 # Per square
        acc_perspective = 0.0
        
        # Arrow Metrics
        arrow_tp = 0
        arrow_fp = 0
        arrow_fn = 0
        
        # Highlight Metrics (treat class 1 and 2 as positive, class 0 as negative)
        highlight_tp = 0
        highlight_fp = 0
        highlight_fn = 0
        
        
        count_batches = 0
        total_samples = 0
        
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                l_p = batch['pieces'].to(device)
                l_h = batch['highlights'].to(device)
                l_a = batch['arrows'].to(device)
                l_f = batch['flipped'].to(device)
                
                o_p, o_h, o_a, o_f = model(images)
                
                # Validation Loss usually useful too
                # Calculate losses
                v_loss_p = criterion_pieces(o_p.permute(0, 2, 1), l_p)
                v_loss_h = criterion_highlights(o_h.permute(0, 2, 1), l_h)
                v_loss_a = criterion_arrows(o_a, l_a)
                v_loss_f = criterion_perspective(o_f, l_f)
                val_loss_sum += ((v_loss_p * args.piece_loss_weight) + (v_loss_h * args.highlight_loss_weight) + (v_loss_a * args.arrow_loss_weight) + v_loss_f).item()
                
                # Metrics
                batch_size = images.size(0)
                total_samples += batch_size
                count_batches += 1
                
                # Pieces
                _, preds_p = torch.max(o_p, 2)
                acc_pieces += (preds_p == l_p).sum().item() / 64.0
                # Board (all squares match)
                acc_boards += (preds_p == l_p).all(dim=1).sum().item()
                
                # Highlights
                _, preds_h = torch.max(o_h, 2)
                acc_highlights += (preds_h == l_h).sum().item() / 64.0
                
                # Highlight precision/recall (class 1 or 2 = positive, class 0 = negative)
                preds_h_binary = (preds_h > 0).float()
                l_h_binary = (l_h > 0).float()
                p_h_flat = preds_h_binary.view(-1)
                l_h_flat = l_h_binary.view(-1)
                
                highlight_tp += (p_h_flat * l_h_flat).sum().item()
                highlight_fp += (p_h_flat * (1 - l_h_flat)).sum().item()
                highlight_fn += ((1 - p_h_flat) * l_h_flat).sum().item()
                
                # Arrows (Threshold 0.5)
                # Use sigmoid to get probs
                probs_a = torch.sigmoid(o_a)
                preds_a = (probs_a > 0.5).float()
                
                # TP, FP, FN calculation
                # Flatten for metric calc
                p_flat = preds_a.view(-1)
                l_flat = l_a.view(-1)
                
                arrow_tp += (p_flat * l_flat).sum().item()
                arrow_fp += (p_flat * (1 - l_flat)).sum().item()
                arrow_fn += ((1 - p_flat) * l_flat).sum().item()
                
                # Perspective
                preds_f = (torch.sigmoid(o_f) > 0.5).float()
                acc_perspective += (preds_f == l_f).sum().item()
                
        # Average metrics
        metric_piece_acc = acc_pieces / total_samples
        metric_board_acc = acc_boards / total_samples
        metric_highlight_acc = acc_highlights / total_samples # Average per-image accuracy
        metric_perspective_acc = acc_perspective / total_samples
        
        # Arrow Precision/Recall
        epsilon = 1e-7
        metric_arrow_precision = arrow_tp / (arrow_tp + arrow_fp + epsilon)
        metric_arrow_recall = arrow_tp / (arrow_tp + arrow_fn + epsilon)
        metric_arrow_f1 = 2 * (metric_arrow_precision * metric_arrow_recall) / (metric_arrow_precision + metric_arrow_recall + epsilon)
        
        # Highlight Precision/Recall
        metric_highlight_precision = highlight_tp / (highlight_tp + highlight_fp + epsilon)
        metric_highlight_recall = highlight_tp / (highlight_tp + highlight_fn + epsilon)
        metric_highlight_f1 = 2 * (metric_highlight_precision * metric_highlight_recall) / (metric_highlight_precision + metric_highlight_recall + epsilon)
        
        avg_val_loss = val_loss_sum / count_batches
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Log validation metrics
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_piece_acc": metric_piece_acc,
            "val_board_acc": metric_board_acc,
            "val_highlight_acc": metric_highlight_acc,
            "val_highlight_precision": metric_highlight_precision,
            "val_highlight_recall": metric_highlight_recall,
            "val_highlight_f1": metric_highlight_f1,
            "val_arrow_precision": metric_arrow_precision,
            "val_arrow_recall": metric_arrow_recall,
            "val_arrow_f1": metric_arrow_f1,
            "val_perspective_acc": metric_perspective_acc,
            "learning_rate": current_lr
        })
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracies -> Board: {metric_board_acc:.4f}, Pieces: {metric_piece_acc:.4f}, "
              f"HighLt: {metric_highlight_acc:.4f}, Persp: {metric_perspective_acc:.4f}")
        print(f"  Highlights -> Precision: {metric_highlight_precision:.4f}, Recall: {metric_highlight_recall:.4f}, F1: {metric_highlight_f1:.4f}")
        print(f"  Arrows -> Precision: {metric_arrow_precision:.4f}, Recall: {metric_arrow_recall:.4f}, F1: {metric_arrow_f1:.4f}")
        
        if metric_board_acc > best_acc:
            best_acc = metric_board_acc
            torch.save(model.state_dict(), "model.pth")
            wandb.save("model.pth") # Save best model to wandb
            print("Saved model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing labels.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="chess-vision", help="WandB project name")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pth) to resume from")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights for backbone")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Do not use pretrained weights")
    parser.add_argument("--piece_loss_weight", type=float, default=200.0, help="Weight for the piece classification loss")
    parser.add_argument("--arrow_loss_weight", type=float, default=900.0, help="Weight for the arrow binary classification loss")
    parser.add_argument("--highlight_loss_weight", type=float, default=300.0, help="Weight for the highlight classification loss")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Max gradient norm for clipping")
    parser.add_argument("--weight_file", type=str, default="weights.json", help="Path to JSON file for live weight adjustment")
    parser.add_argument("--no_weight_norm", action="store_true", help="Disable weight normalization (use raw weighted sum)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        wandb.finish()
        raise
