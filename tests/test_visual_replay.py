from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from scripts.visual_replay import VisualReplayRecorder, board_to_image


@dataclass
class Frame:
    frame: list[list[list[int]]]
    state: str = "ACTIVE"
    levels_completed: int = 0
    guid: str = "g"


class VisualReplayTests(unittest.TestCase):
    def test_palette_and_nearest_neighbor_scaling(self) -> None:
        image = board_to_image([[[0, 1], [2, 99]]], scale=3, labels=[])
        self.assertEqual(image.size, (6, 6))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(image.getpixel((3, 0)), (0, 116, 217))
        self.assertEqual(image.getpixel((0, 3)), (214, 40, 40))
        self.assertEqual(image.getpixel((3, 3)), (35, 35, 35))

    def test_labels_add_metadata_bar(self) -> None:
        plain = board_to_image([[[1]]], scale=2, labels=[])
        labeled = board_to_image([[[1]]], scale=2, labels=["step 1", "ACTION2"])
        self.assertEqual(plain.size, (2, 2))
        self.assertEqual(labeled.size[0], 2)
        self.assertGreater(labeled.size[1], plain.size[1])

    def test_recorder_writes_png_gif_and_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = VisualReplayRecorder(directory, fmt="both", scale=2)
            recorder.record(Frame([[[1, 2], [3, 4]]]), action="ACTION1", step=1)
            recorder.record(Frame([[[5, 6], [7, 8]]], levels_completed=1), action="ACTION5", step=2)
            artifacts = recorder.close()
            self.assertTrue(Path(artifacts["gif"]).exists())
            self.assertEqual(len(list((Path(directory) / "frames").glob("*.png"))), 2)
            with Path(artifacts["timeline"]).open(encoding="utf-8") as handle:
                entries = [json.loads(line) for line in handle]
            self.assertEqual([entry["action"] for entry in entries], ["ACTION1", "ACTION5"])
            self.assertEqual(entries[1]["levels_completed"], 1)
            with Image.open(artifacts["gif"]) as image:
                self.assertEqual(getattr(image, "n_frames", 1), 2)

    def test_record_every_filters_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = VisualReplayRecorder(directory, fmt="png", every=2)
            for step in range(1, 6):
                recorder.record(Frame([[[step]]]), action=f"ACTION{step}", step=step)
            recorder.close()
            entries = (Path(directory) / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(entries), 2)
            self.assertEqual([json.loads(line)["step"] for line in entries], [2, 4])


if __name__ == "__main__":
    unittest.main()
