"""
Weight GUI - Live adjustment of training loss weights.

Run this alongside training to adjust weights in real-time.
The training script reads weights.json every 10 batches.

Usage: python weight_gui.py [--weight_file weights.json]
"""

import tkinter as tk
from tkinter import ttk
import json
import os
import argparse

DEFAULT_WEIGHT_FILE = "weights.json"
EVAL_FLAG_FILE = "run_eval.flag"
EVAL_RESULTS_FILE = "eval_results.json"

DEFAULT_WEIGHTS = {
    'pieces': 200.0,
    'highlights': 300.0,
    'arrows': 900.0,
    'perspective': 1.0,
    'learning_rate': 3e-5,
    'max_grad_norm': 10.0
}

# Slider configurations: (min, max, resolution, is_log_scale)
SLIDER_CONFIG = {
    'pieces': (0.1, 1000.0, 0.1, False),
    'highlights': (0.1, 1000.0, 0.1, False),
    'arrows': (0.1, 10000.0, 0.1, True),  # Log scale for arrows
    'perspective': (0.1, 100.0, 0.1, False),
    'learning_rate': (1e-6, 1e-2, 0.1, True),  # Log scale for LR
    'max_grad_norm': (1.0, 1000.0, 0.1, True)  # Log scale
}


class WeightGUI:
    def __init__(self, weight_file):
        self.weight_file = weight_file
        self.weights = DEFAULT_WEIGHTS.copy()
        self.sliders = {}
        self.value_labels = {}
        
        # Load existing weights if file exists
        self._load_weights()
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Training Weight Adjuster")
        self.root.geometry("500x550")
        self.root.resizable(False, False)
        
        self._create_widgets()
        
        # Start polling for eval results
        self._poll_eval_results()
        
    def _load_weights(self):
        """Load weights from JSON file."""
        if os.path.exists(self.weight_file):
            try:
                with open(self.weight_file, 'r') as f:
                    loaded = json.load(f)
                    for key in self.weights:
                        if key in loaded:
                            self.weights[key] = float(loaded[key])
            except (json.JSONDecodeError, ValueError):
                pass
    
    def _save_weights(self):
        """Save weights to JSON file."""
        with open(self.weight_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
    
    def _on_slider_change(self, name, value):
        """Handle slider value change."""
        config = SLIDER_CONFIG[name]
        is_log = config[3]
        
        if is_log:
            # Convert from log scale
            import math
            min_val, max_val = config[0], config[1]
            log_min, log_max = math.log10(min_val), math.log10(max_val)
            log_val = log_min + (float(value) / 100.0) * (log_max - log_min)
            actual_value = 10 ** log_val
        else:
            actual_value = float(value)
        
        # Store value with appropriate precision
        if name == 'learning_rate':
            self.weights[name] = actual_value  # Keep full precision for LR
        else:
            self.weights[name] = round(actual_value, 2)
        
        # Only update label if it exists (may not during initialization)
        if name in self.value_labels:
            if name == 'learning_rate':
                self.value_labels[name].config(text=f"{self.weights[name]:.2e}")
            else:
                self.value_labels[name].config(text=f"{self.weights[name]:.2f}")
        
        self._save_weights()
    
    def _create_widgets(self):
        """Create the GUI widgets."""
        # Title
        title = ttk.Label(self.root, text="Live Training Weights", font=('Helvetica', 14, 'bold'))
        title.pack(pady=10)
        
        # File indicator
        file_label = ttk.Label(self.root, text=f"File: {self.weight_file}", font=('Helvetica', 9))
        file_label.pack()
        
        # Sliders frame
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill='both', expand=True)
        
        import math
        
        for i, (name, config) in enumerate(SLIDER_CONFIG.items()):
            min_val, max_val, resolution, is_log = config
            
            # Label
            label = ttk.Label(frame, text=f"{name.capitalize()}:", width=12, anchor='e')
            label.grid(row=i, column=0, padx=5, pady=8, sticky='e')
            
            # Slider
            if is_log:
                # Log scale slider (0-100 range, mapped to log scale)
                slider = ttk.Scale(
                    frame,
                    from_=0,
                    to=100,
                    orient='horizontal',
                    length=200,
                    command=lambda v, n=name: self._on_slider_change(n, v)
                )
                # Set initial position on log scale
                log_min, log_max = math.log10(min_val), math.log10(max_val)
                if self.weights[name] > 0:
                    log_val = math.log10(self.weights[name])
                    position = (log_val - log_min) / (log_max - log_min) * 100
                    slider.set(max(0, min(100, position)))
            else:
                slider = ttk.Scale(
                    frame,
                    from_=min_val,
                    to=max_val,
                    orient='horizontal',
                    length=200,
                    command=lambda v, n=name: self._on_slider_change(n, v)
                )
                slider.set(self.weights[name])
            
            slider.grid(row=i, column=1, padx=5, pady=8)
            self.sliders[name] = slider
            
            # Value label - use scientific notation for LR
            if name == 'learning_rate':
                label_text = f"{self.weights[name]:.2e}"
            else:
                label_text = f"{self.weights[name]:.2f}"
            value_label = ttk.Label(frame, text=label_text, width=10)
            value_label.grid(row=i, column=2, padx=5, pady=8)
            self.value_labels[name] = value_label
        
        # Reset button
        reset_btn = ttk.Button(self.root, text="Reset to Defaults", command=self._reset_weights)
        reset_btn.pack(pady=5)
        
        # Eval section
        eval_frame = ttk.Frame(self.root, padding=5)
        eval_frame.pack(fill='x')
        
        self.eval_btn = ttk.Button(eval_frame, text="Run Test Eval", command=self._trigger_eval)
        self.eval_btn.pack(side='left', padx=10)
        
        self.eval_status = ttk.Label(eval_frame, text="", width=30)
        self.eval_status.pack(side='left')
        
        # Results display (multi-line)
        self.results_frame = ttk.LabelFrame(self.root, text="Eval Results", padding=5)
        self.results_frame.pack(fill='x', padx=10, pady=5)
        
        self.results_text = tk.Text(self.results_frame, height=5, width=50, font=('Consolas', 9))
        self.results_text.pack()
        self.results_text.insert('1.0', "Click 'Run Test Eval' to evaluate")
        self.results_text.config(state='disabled')
        
        # Status bar
        status = ttk.Label(self.root, text="Changes save automatically", font=('Helvetica', 8))
        status.pack(side='bottom', pady=5)
    
    def _trigger_eval(self):
        """Create flag file to request evaluation."""
        with open(EVAL_FLAG_FILE, 'w') as f:
            f.write('run')
        self.eval_status.config(text="Waiting for eval...")
        self.eval_btn.config(state='disabled')
    
    def _poll_eval_results(self):
        """Check for eval results periodically."""
        if os.path.exists(EVAL_RESULTS_FILE):
            try:
                mtime = os.path.getmtime(EVAL_RESULTS_FILE)
                if not hasattr(self, '_last_result_mtime') or mtime != self._last_result_mtime:
                    self._last_result_mtime = mtime
                    with open(EVAL_RESULTS_FILE, 'r') as f:
                        results = json.load(f)
                    
                    # Format all metrics
                    lines = [
                        f"Board: {results.get('board_acc', 0)*100:.1f}%    Piece: {results.get('piece_acc', 0)*100:.1f}%    Persp: {results.get('perspective_acc', 0)*100:.1f}%",
                        f"Highlight  P: {results.get('highlight_precision', 0)*100:.1f}%  R: {results.get('highlight_recall', 0)*100:.1f}%  F1: {results.get('highlight_f1', 0)*100:.1f}%",
                        f"Arrow      P: {results.get('arrow_precision', 0)*100:.1f}%  R: {results.get('arrow_recall', 0)*100:.1f}%  F1: {results.get('arrow_f1', 0)*100:.1f}%",
                        f"Loss: {results.get('loss', 0):.4f}"
                    ]
                    
                    # Update text widget
                    self.results_text.config(state='normal')
                    self.results_text.delete('1.0', tk.END)
                    self.results_text.insert('1.0', '\n'.join(lines))
                    self.results_text.config(state='disabled')
                    
                    self.eval_status.config(text="Done!")
                    self.eval_btn.config(state='normal')
            except:
                pass
        
        # Poll every 500ms
        self.root.after(500, self._poll_eval_results)
    
    def _reset_weights(self):
        """Reset all weights to defaults."""
        import math
        self.weights = DEFAULT_WEIGHTS.copy()
        
        for name, config in SLIDER_CONFIG.items():
            min_val, max_val, _, is_log = config
            if is_log:
                log_min, log_max = math.log10(min_val), math.log10(max_val)
                log_val = math.log10(self.weights[name])
                position = (log_val - log_min) / (log_max - log_min) * 100
                self.sliders[name].set(position)
            else:
                self.sliders[name].set(self.weights[name])
            self.value_labels[name].config(text=f"{self.weights[name]:.2f}")
        
        self._save_weights()
    
    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI for adjusting training weights")
    parser.add_argument("--weight_file", type=str, default=DEFAULT_WEIGHT_FILE,
                        help="Path to weights JSON file")
    args = parser.parse_args()
    
    gui = WeightGUI(args.weight_file)
    gui.run()
