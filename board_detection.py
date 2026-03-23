"""Shared chess-board detection helpers."""

import inspect

import torch


MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BOARD_QUERY = "chess board."
TEXT_THRESHOLD = 0.15
BOX_THRESHOLD = 0.15


def clamp_bbox(box, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [int(value) for value in box]

    x1 = max(0, min(x1, max(width - 1, 0)))
    y1 = max(0, min(y1, max(height - 1, 0)))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return x1, y1, x2, y2


def expand_bbox(box, image_size, padding_ratio=0.04):
    x1, y1, x2, y2 = clamp_bbox(box, image_size)
    pad_x = max(1, int(round((x2 - x1) * padding_ratio)))
    pad_y = max(1, int(round((y2 - y1) * padding_ratio)))
    return clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_size)


def crop_board(image, bbox, padding_ratio=0.04):
    return image.crop(expand_bbox(bbox, image.size, padding_ratio=padding_ratio))


class BoardDetector:
    """Detect a chess-board region in a PIL image with GroundingDINO."""

    def __init__(self, device="cpu"):
        self.device = device
        self._model = None
        self._processor = None
        self._threshold_arg = "threshold"

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(self.device)
        )

        try:
            params = inspect.signature(
                self._processor.post_process_grounded_object_detection
            ).parameters
            self._threshold_arg = (
                "box_threshold" if "box_threshold" in params else "threshold"
            )
        except (TypeError, ValueError):
            self._threshold_arg = "threshold"

        self._model.eval()

    def detect(self, image):
        self._load()
        inputs = self._processor(
            images=image,
            text=BOARD_QUERY,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
            **{self._threshold_arg: BOX_THRESHOLD},
        )[0]

        if len(results["boxes"]) == 0:
            return None

        best = results["scores"].argmax().item()
        box = results["boxes"][best].cpu().numpy().astype(int)
        return clamp_bbox(box.tolist(), image.size)

    def unload(self):
        self._model = None
        self._processor = None
        self._threshold_arg = "threshold"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
