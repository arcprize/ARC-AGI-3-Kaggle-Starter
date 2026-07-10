from __future__ import annotations

import unittest
from unittest.mock import patch

from agi_core.cuda import CUDA_AVAILABLE, init_cuda, is_enabled


class TestCudaModule(unittest.TestCase):
    """Tests for the CUDA acceleration module."""

    def test_init_cuda_no_gpu(self):
        """init_cuda should not crash even without a GPU."""
        result = init_cuda(device_id=0)
        # On systems without CUDA, this returns False without raising
        self.assertIsInstance(result, bool)

    def test_is_enabled_no_config(self):
        """is_enabled should work without a config object (reads env)."""
        result = is_enabled(config=None)
        self.assertIsInstance(result, bool)

    def test_is_enabled_with_config(self):
        """is_enabled should check the config object when provided."""
        class FakeConfig:
            cuda_enabled = True
        self.assertIsInstance(is_enabled(FakeConfig()), bool)

    def test_is_enabled_disabled(self):
        """is_enabled returns False when config disables CUDA."""
        class FakeConfig:
            cuda_enabled = False
        # If CUDA is actually available, config still controls it
        self.assertFalse(is_enabled(FakeConfig()))


class TestSalienceGpu(unittest.TestCase):
    """GPU salience tests (skip if CUDA unavailable)."""

    def setUp(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA not available")

    def test_salience_basic(self):
        from agi_core.cuda.salience import compute_salience_gpu
        board = [[9] * 64 for _ in range(64)]
        result = compute_salience_gpu(board, (32, 32), [(10, 10)])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 64)
        self.assertEqual(len(result[0]), 64)
        # Cell at player position should be salient
        self.assertGreater(result[32][32], 0.0)

    def test_salience_no_goals(self):
        from agi_core.cuda.salience import compute_salience_gpu
        board = [[9] * 64 for _ in range(64)]
        result = compute_salience_gpu(board, (32, 32), [])
        self.assertIsNotNone(result)

    def test_salience_no_player(self):
        from agi_core.cuda.salience import compute_salience_gpu
        board = [[9] * 64 for _ in range(64)]
        result = compute_salience_gpu(board, None, [(10, 10)])
        self.assertIsNotNone(result)


class TestAstarGpu(unittest.TestCase):
    """GPU A* tests (skip if CUDA unavailable)."""

    def setUp(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA not available")

    def test_astar_simple_path(self):
        from agi_core.cuda.astar import batch_astar_gpu
        walkable = {(x, y) for x in range(64) for y in range(64)}
        results = batch_astar_gpu(
            [walkable], [(0, 0)], [(5, 5)],
            width=64, height=64,
        )
        self.assertIsNotNone(results[0])
        self.assertGreater(len(results[0]), 0)
        self.assertEqual(results[0][0], (0, 0))
        self.assertEqual(results[0][-1], (5, 5))

    def test_astar_no_path(self):
        from agi_core.cuda.astar import batch_astar_gpu
        walkable = {(0, 0)}  # only start cell is walkable
        results = batch_astar_gpu(
            [walkable], [(0, 0)], [(5, 5)],
            width=64, height=64,
        )
        self.assertIsNone(results[0])


class TestCpuFallback(unittest.TestCase):
    """CPU fallback tests (always run)."""

    def test_batch_astar_fallback(self):
        from agi_core.cuda.astar import batch_astar_gpu
        walkable = {(x, y) for x in range(10) for y in range(10)}
        # Disable CUDA
        with patch("agi_core.cuda.CUDA_AVAILABLE", False):
            results = batch_astar_gpu(
                [walkable, walkable],
                [(0, 0), (9, 9)],
                [(5, 5), (0, 0)],
                width=10, height=10,
            )
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])

    def test_salience_fallback(self):
        """When CUDA is unavailable, compute_salience_gpu returns None."""
        with patch("agi_core.cuda.CUDA_AVAILABLE", False):
            from agi_core.cuda.salience import compute_salience_gpu
            result = compute_salience_gpu([[9] * 64 for _ in range(64)], (0, 0), [])
        self.assertIsNone(result)
