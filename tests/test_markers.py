"""Small deterministic checks for the frame-0 marker proposer."""

import unittest

import numpy as np

from arc3x.markers import marker_colors, markers


class MarkerTests(unittest.TestCase):
    def test_repeated_components_are_proposed(self):
        frame = np.zeros((12, 12), dtype=np.int16)
        frame[2:4, 2:4] = 7
        frame[8:10, 8:10] = 7

        proposals = markers(frame)

        self.assertEqual(proposals[0].signature, (7, 2, 2, 4))
        self.assertEqual(proposals[0].cells, ((2, 2), (8, 8)))

    def test_moved_colours_are_excluded(self):
        frame = np.zeros((12, 12), dtype=np.int16)
        frame[2:4, 2:4] = 7
        frame[8:10, 8:10] = 7

        self.assertEqual(markers(frame, moved={7}), [])

    def test_top_colour_projection_is_bounded(self):
        frame = np.zeros((12, 12), dtype=np.int16)
        frame[1:3, 1:3] = 3
        frame[5:7, 5:7] = 3
        frame[9:11, 1:3] = 5
        frame[9:11, 7:9] = 5

        proposals = markers(frame)

        self.assertEqual(marker_colors(proposals, top=1), {3, 5} & {proposals[0].color})


if __name__ == "__main__":
    unittest.main()
