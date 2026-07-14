"""Opt-in visual replay recording for local ARC-AGI game runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - exercised by dependency failures
    raise RuntimeError(
        "Visual replay recording requires Pillow. Install it in the project venv."
    ) from exc


ARC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (214, 40, 40),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (128, 128, 128),
    6: (255, 65, 255),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (133, 20, 75),
    10: (255, 23, 68),
    11: (57, 204, 204),
}
UNKNOWN_COLOR = (35, 35, 35)


def _as_board(frame: Any) -> list[list[int]]:
    """Extract the first board from FrameData or a raw board value."""
    value = getattr(frame, "frame", frame)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        if value[0] and isinstance(value[0][0], (list, tuple)):
            value = value[0]
    return [
        [int(cell) for cell in row]
        for row in value
        if isinstance(row, (list, tuple))
    ]


def board_to_image(board: Any, scale: int = 8, labels: Iterable[str] = ()) -> Image.Image:
    """Render a board as enlarged nearest-neighbor pixel art."""
    if scale < 1:
        raise ValueError("scale must be at least 1")
    rows = _as_board(board)
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    image = Image.new("RGB", (max(1, width), max(1, height)), UNKNOWN_COLOR)
    pixels = image.load()
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            pixels[x, y] = ARC_PALETTE.get(value, UNKNOWN_COLOR)
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    label_values = [str(label) for label in labels if str(label)]
    if label_values:
        bar_height = 22
        labeled = Image.new("RGB", (image.width, image.height + bar_height), (18, 18, 18))
        labeled.paste(image, (0, bar_height))
        ImageDraw.Draw(labeled).text((6, 5), " | ".join(label_values), fill=(245, 245, 245))
        image = labeled
    return image


def _state_value(frame: Any) -> str:
    state = getattr(frame, "state", "")
    return str(getattr(state, "value", state))


class VisualReplayRecorder:
    """Write selected frames, an animated GIF, and a machine-readable timeline."""

    def __init__(self, output_dir: str | Path, *, fmt: str = "both", every: int = 1,
                 scale: int = 8, labels: bool = True) -> None:
        if fmt not in {"png", "gif", "both"}:
            raise ValueError("fmt must be png, gif, or both")
        if every < 1:
            raise ValueError("every must be at least 1")
        if scale < 1:
            raise ValueError("scale must be at least 1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        if fmt in {"png", "both"}:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.every = every
        self.scale = scale
        self.labels = labels
        self._images: list[Image.Image] = []
        self._timeline = self.output_dir / "timeline.jsonl"
        self._frame_number = 0

    def record(self, frame: Any, *, action: str, step: int) -> None:
        if step % self.every != 0:
            return
        self._frame_number += 1
        state = _state_value(frame)
        levels = int(getattr(frame, "levels_completed", 0) or 0)
        labels = (
            [f"step {step}", str(action), f"level {levels}", state]
            if self.labels else []
        )
        image = board_to_image(frame, scale=self.scale, labels=labels)
        image_path: str | None = None
        if self.fmt in {"png", "both"}:
            path = self.frames_dir / f"frame_{self._frame_number:05d}.png"
            image.save(path)
            image_path = str(path)
        if self.fmt in {"gif", "both"}:
            self._images.append(image.copy())
        entry = {
            "step": step,
            "action": str(action),
            "state": state,
            "levels_completed": levels,
            "guid": getattr(frame, "guid", None),
            "image": image_path,
        }
        with self._timeline.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def close(self) -> dict[str, str | None]:
        gif_path: Path | None = None
        if self._images:
            gif_path = self.output_dir / "replay.gif"
            self._images[0].save(
                gif_path,
                save_all=True,
                append_images=self._images[1:],
                duration=250,
                loop=0,
            )
        return {
            "frames_dir": str(self.frames_dir) if self.fmt in {"png", "both"} else None,
            "gif": str(gif_path) if gif_path else None,
            "timeline": str(self._timeline),
        }
