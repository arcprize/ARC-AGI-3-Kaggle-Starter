from __future__ import annotations

import importlib.util
import os
import pathlib
import unittest
import types
from unittest.mock import patch

import numpy as np

from agi_core.cuda import (
    CUDA_AVAILABLE,
    format_kernel_stats,
    get_cupy,
    init_cuda,
    is_enabled,
    note_kernel_stat,
    snapshot_kernel_stats,
)


def _load_live_ar25_opening_board() -> list[list[int]]:
    env_path = pathlib.Path("environment_files/ar25/0c556536/ar25.py")
    spec = importlib.util.spec_from_file_location("test_ar25_env", env_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    game = module.Ar25()
    return game.camera.render(game.current_level.get_sprites()).tolist()


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

    def test_is_enabled_defaults_env_to_on(self):
        with patch("agi_core.cuda.CUDA_AVAILABLE", True), patch.dict("os.environ", {}, clear=True):
            self.assertTrue(is_enabled(config=None))

    def test_get_cupy_importable(self):
        cp = get_cupy()
        self.assertTrue(hasattr(cp, "asarray"))

    def test_init_cuda_uses_cupy_device_probe(self):
        fake_cp = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                Device=lambda idx: types.SimpleNamespace(use=lambda: None),
                runtime=types.SimpleNamespace(
                    getDeviceProperties=lambda idx: {"name": b"Fake GPU"},
                ),
                is_available=lambda: True,
            )
        )
        with patch("agi_core.cuda.get_cupy", return_value=fake_cp):
            self.assertTrue(init_cuda(0))

    def test_kernel_stats_snapshot_and_reset(self):
        snapshot_kernel_stats(reset=True)
        note_kernel_stat("astar")
        note_kernel_stat("components.propagate", 3)
        self.assertEqual(
            snapshot_kernel_stats(),
            {"astar": 1, "components.propagate": 3},
        )
        self.assertEqual(
            format_kernel_stats(),
            "astar=1, components.propagate=3",
        )
        self.assertEqual(
            snapshot_kernel_stats(reset=True),
            {"astar": 1, "components.propagate": 3},
        )
        self.assertEqual(snapshot_kernel_stats(), {})


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

    def test_astar_respects_per_request_walkable_rasters(self):
        from agi_core.cuda.astar import batch_astar_gpu

        def fake_launch(walkable_np, starts_x, starts_y, goals_x, goals_y, paths_out, path_lens):
            self.assertEqual(walkable_np.shape, (2, 64, 64))
            self.assertEqual(int(walkable_np[0, 0, 1]), 1)
            self.assertEqual(int(walkable_np[1, 0, 1]), 0)
            paths_out[0, 0] = 0
            paths_out[0, 1] = 1
            path_lens[0] = 2
            path_lens[1] = 0

        with patch("agi_core.cuda.astar.cuda_runtime.CUDA_AVAILABLE", True), patch(
            "agi_core.cuda.astar.kernel_available",
            return_value=True,
        ), patch(
            "agi_core.cuda.astar._launch_astar_kernel",
            side_effect=fake_launch,
        ):
            results = batch_astar_gpu(
                [{(0, 0), (1, 0)}, {(0, 0)}],
                [(0, 0), (0, 0)],
                [(1, 0), (1, 0)],
                width=64,
                height=64,
            )

        self.assertEqual(results[0], [(0, 0), (1, 0)])
        self.assertIsNone(results[1])


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

    def test_batch_astar_falls_back_when_kernel_throws(self):
        from agi_core.cuda.astar import batch_astar_gpu

        walkable = {(x, y) for x in range(4) for y in range(4)}
        with patch("agi_core.cuda.astar.cuda_runtime.CUDA_AVAILABLE", True), patch(
            "agi_core.cuda.astar.kernel_available",
            return_value=True,
        ), patch(
            "agi_core.cuda.astar._launch_astar_kernel",
            side_effect=RuntimeError("boom"),
        ):
            results = batch_astar_gpu(
                [walkable],
                [(0, 0)],
                [(3, 3)],
                width=4,
                height=4,
            )

        self.assertIsNotNone(results[0])
        self.assertEqual(results[0][0], (0, 0))
        self.assertEqual(results[0][-1], (3, 3))

    def test_batch_astar_fallback_does_not_reenter_public_astar(self):
        from agi_core.cuda.astar import batch_astar_gpu

        walkable = {(x, y) for x in range(4) for y in range(4)}
        with patch("agi_core.cuda.astar.cuda_runtime.CUDA_AVAILABLE", False), patch(
            "agi_core.orch_or.search.astar",
            side_effect=AssertionError("public astar should not be re-entered"),
        ):
            results = batch_astar_gpu(
                [walkable],
                [(0, 0)],
                [(3, 3)],
                width=4,
                height=4,
            )

        self.assertIsNotNone(results[0])
        self.assertEqual(results[0][0], (0, 0))
        self.assertEqual(results[0][-1], (3, 3))


class TestCudaComponentDetection(unittest.TestCase):
    """GPU component detection tests (CPU fallback variant always runs)."""

    def setUp(self):
        self._real_cuda_available = CUDA_AVAILABLE

    def _gpu_or_fallback(self, board, bg_color=-1):
        """Try GPU; if CUDA unavailable, call the CPU implementations
        and wrap results into the same dict format as detect_components_gpu."""
        from agi_core.cuda.components import detect_components_gpu
        gpu_result = detect_components_gpu(board, bg_color=bg_color)
        if gpu_result is not None:
            return gpu_result
        # CPU fallback: compute components and colour frequencies manually
        return self._cpu_components(board, bg_color)

    def _cpu_components(self, board, bg_color=-1):
        """CPU reference — calls the actual CPU flood-fill used by the
        production code so we can verify GPU results match."""
        from agi_core.orch_or.analyzer import _find_components, _compute_color_frequencies
        h, w = len(board), len(board[0])
        comps = _find_components(board, w, h, bg_color)
        # Build same dict format as detect_components_gpu
        raw = []
        for c in comps:
            sx = sum(x for x, _ in c.cells) // c.size if c.size > 0 else 0
            sy = sum(y for _, y in c.cells) // c.size if c.size > 0 else 0
            raw.append({
                "label": 0,  # not used
                "color": c.dominant_color,
                "cells": c.cells,
                "bbox": c.bbox,
                "size": c.size,
                "center": (sx, sy),
                "touches_border": c.touches_border,
            })
        freq = _compute_color_frequencies(board, bg_color)
        return {"components": raw, "color_frequencies": freq}

    def _make_board(self, pattern, width=64, height=64, fill=-1):
        """Create a board from an ASCII-art list of strings."""
        board = [[fill] * width for _ in range(height)]
        for y, row in enumerate(pattern.splitlines()):
            row = row.strip()
            if not row:
                continue
            for x, ch in enumerate(row):
                if ch != ".":
                    board[y][x] = int(ch)
        return board

    def test_empty_board(self):
        board = self._make_board("")
        gpu = self._gpu_or_fallback(board, bg_color=9)
        self.assertEqual(len(gpu["components"]), 0)
        self.assertEqual(gpu["color_frequencies"], {})

    def test_single_cell_component(self):
        board = self._make_board("5", height=5, width=5)
        gpu = self._gpu_or_fallback(board, bg_color=9)
        self.assertEqual(len(gpu["components"]), 1)
        self.assertEqual(gpu["components"][0]["color"], 5)
        self.assertEqual(gpu["components"][0]["size"], 1)

    def test_two_separate_components_same_color(self):
        board = self._make_board(
            "5..5\n....\n....\n....", height=5, width=5,
        )
        gpu = self._gpu_or_fallback(board, bg_color=9)
        self.assertEqual(len(gpu["components"]), 2)
        colors = [c["color"] for c in gpu["components"]]
        self.assertEqual(colors, [5, 5])

    def test_two_colors(self):
        board = [[-1] * 10 for _ in range(10)]
        board[1][1] = 5
        board[1][5] = 10
        gpu = self._gpu_or_fallback(board, bg_color=9)
        self.assertEqual(len(gpu["components"]), 2)
        colors = sorted(c["color"] for c in gpu["components"])
        self.assertEqual(colors, [5, 10])

    def test_component_matches_cpu_bbox(self):
        """Verify GPU (or its CPU fallback) produces the same bbox
        and component counts as the direct CPU call."""
        from agi_core.orch_or.analyzer import _find_components
        board = [[-1] * 20 for _ in range(20)]
        # Draw a 5x5 block of color 7 at (3, 4)
        for y in range(4, 9):
            for x in range(3, 8):
                board[y][x] = 7

        gpu = self._gpu_or_fallback(board, bg_color=-1)
        cpu = _find_components(board, 20, 20, bg_color=-1)

        gpu_comp = gpu["components"][0] if gpu["components"] else None
        cpu_comp = cpu[0] if cpu else None

        self.assertIsNotNone(gpu_comp)
        self.assertIsNotNone(cpu_comp)
        self.assertEqual(gpu_comp["bbox"], cpu_comp.bbox)
        self.assertEqual(gpu_comp["size"], cpu_comp.size)
        self.assertEqual(gpu_comp["color"], cpu_comp.dominant_color)

    def test_ar25_level1_board(self):
        """Use a real AR25 Level 1 synthetic board."""
        from tests.test_puzzle_games import game_ar25_level1
        gm = game_ar25_level1()
        board = gm.board
        bg_color = next((c.dominant_color for c in gm.components
                         if c.touches_border and c.dominant_color != 0 and c.dominant_color != 10), 9)

        gpu = self._gpu_or_fallback(board, bg_color=bg_color)
        from agi_core.orch_or.analyzer import _find_components
        cpu = _find_components(board, gm.width, gm.height, bg_color)

        # Same number of components
        self.assertEqual(len(gpu["components"]), len(cpu),
                         f"GPU has {len(gpu['components'])} comps, CPU has {len(cpu)}")

    def test_color_frequencies_empty(self):
        board = [[-1] * 10 for _ in range(10)]
        gpu = self._gpu_or_fallback(board, bg_color=-1)
        self.assertEqual(gpu["color_frequencies"], {})

    def test_color_frequencies_multi(self):
        board = [[-1] * 10 for _ in range(10)]
        board[0][0] = 5
        board[1][1] = 5
        board[2][2] = 10
        gpu = self._gpu_or_fallback(board, bg_color=-1)
        self.assertEqual(gpu["color_frequencies"].get(5), 2)
        self.assertEqual(gpu["color_frequencies"].get(10), 1)

    def test_fallback_when_cuda_disabled(self):
        """When CUDA_AVAILABLE is patched to False, detect_components_gpu
        should return None (caller falls back to CPU)."""
        with patch("agi_core.cuda.CUDA_AVAILABLE", False):
            from agi_core.cuda.components import detect_components_gpu
            result = detect_components_gpu([[9] * 64 for _ in range(64)], bg_color=9)
        self.assertIsNone(result)

    def test_gpu_integration_shape_recognition(self):
        """Verify shape_recognition works correctly when GPU path is available.
        When CUDA is not available this just tests the CPU fallback."""
        from tests.test_puzzle_games import game_ar25_level1
        from agi_core.perception.shape_recognition import recognize_shape_recognition
        from agi_core.image_input import BoardImageInput

        gm = game_ar25_level1()
        image_input = BoardImageInput(
            board=gm.board, height=gm.height, width=gm.width,
            frame_signature="test", source_attr="test",
        )
        artifact = recognize_shape_recognition(image_input)
        self.assertTrue(artifact["mirror"]["present"])
        self.assertGreater(len(artifact["goal_clusters"]), 0)

    def test_gpu_integration_analyzer(self):
        """Verify analyzer.build_game_model works correctly when GPU path is
        available. When CUDA is not available this just tests the CPU fallback."""
        from tests.test_puzzle_games import game_ar25_level1
        from agi_core.orch_or.analyzer import build_game_model

        gm = game_ar25_level1()
        model = build_game_model(gm.board)
        self.assertIsNotNone(model.player_pos)
        self.assertGreater(len(model.goal_positions), 0)
        self.assertGreater(len(model.components), 0)


class TestGridScaleGpu(unittest.TestCase):
    """GPU grid scale detection tests (CPU fallback variant)."""

    def test_grid_scale_importable(self):
        from agi_core.cuda.grid_scale import detect_grid_scale_gpu, _best_scale
        self.assertTrue(callable(detect_grid_scale_gpu))

    def test_best_scale_returns_correct(self):
        from agi_core.cuda.grid_scale import _best_scale
        import numpy as np
        uniformity = np.zeros(17, dtype=np.float32)
        edge_scores = np.zeros(17, dtype=np.int32)
        uniformity[3] = 0.80  # 80% uniform at scale 3
        uniformity[4] = 0.60  # 60% uniform at scale 4
        result = _best_scale(uniformity, edge_scores)
        self.assertEqual(result, 3)

    def test_best_scale_returns_1_when_none_above_50pct(self):
        from agi_core.cuda.grid_scale import _best_scale
        import numpy as np
        uniformity = np.zeros(17, dtype=np.float32)
        uniformity[3] = 0.30  # only 30% uniform
        edge_scores = np.zeros(17, dtype=np.int32)
        result = _best_scale(uniformity, edge_scores)
        self.assertEqual(result, 1)

    def test_best_scale_uses_actual_frame_area_for_edge_bonus(self):
        from agi_core.cuda.grid_scale import _best_scale
        import numpy as np

        uniformity = np.zeros(17, dtype=np.float32)
        edge_scores = np.zeros(17, dtype=np.int32)
        uniformity[3] = 0.55
        uniformity[4] = 0.51
        edge_scores[4] = 1

        default_result = _best_scale(uniformity, edge_scores)
        small_frame_result = _best_scale(uniformity, edge_scores, total_cells=8 * 8)

        self.assertEqual(default_result, 3)
        self.assertEqual(small_frame_result, 4)

    def test_uniformity_kernel_importable(self):
        from agi_core.cuda.kernels import _launch_uniformity_kernel, _launch_edge_kernel
        self.assertTrue(callable(_launch_uniformity_kernel))
        self.assertTrue(callable(_launch_edge_kernel))

    def test_gpu_path_returns_none_when_cuda_disabled(self):
        with patch("agi_core.cuda.CUDA_AVAILABLE", False):
            from agi_core.cuda.grid_scale import detect_grid_scale_gpu
            result = detect_grid_scale_gpu([[9] * 64 for _ in range(64)])
        self.assertIsNone(result)

    def test_analyzer_fallback(self):
        """When CUDA is disabled, _detect_grid_scale still works via CPU."""
        from agi_core.orch_or.analyzer import _detect_grid_scale
        board = [[-1] * 64 for _ in range(64)]
        for y in range(21):
            for x in range(21):
                board[y][x] = 9
        scale = _detect_grid_scale(board)
        self.assertGreater(scale, 0)


class TestMirrorSolverGpu(unittest.TestCase):
    """GPU mirror solver tests (CPU fallback variant)."""

    def test_mirror_solver_importable(self):
        from agi_core.cuda.mirror_solver import best_reflection_placement_gpu
        self.assertTrue(callable(best_reflection_placement_gpu))

    def test_kernel_importable(self):
        from agi_core.cuda.kernels import _launch_mirror_placement_kernel
        self.assertTrue(callable(_launch_mirror_placement_kernel))

    def test_gpu_path_returns_none_when_cuda_disabled(self):
        with patch("agi_core.cuda.CUDA_AVAILABLE", False):
            from agi_core.cuda.mirror_solver import best_reflection_placement_gpu
            result = best_reflection_placement_gpu(
                width=64, height=64,
                component_cells=[(0, 0), (1, 0)],
                component_bbox=(0, 0, 1, 0),
                remaining_goals={(5, 5)},
                horizontal=True, axis=16,
            )
        self.assertIsNone(result)

    def test_mirror_solver_module_imports(self):
        from agi_core.cuda.mirror_solver import best_reflection_placement_gpu
        from agi_core.cuda.kernels import _launch_mirror_placement_kernel
        self.assertTrue(callable(best_reflection_placement_gpu))
        self.assertTrue(callable(_launch_mirror_placement_kernel))

    def test_reflection_scoring_helper_returns_candidate_grid(self):
        from agi_core.cuda.mirror_solver import score_reflection_candidates_gpu

        def fake_launch(_goal, _shape, shape_w, shape_h, _horizontal, _axis, _width, _height, score_out):
            self.assertEqual((shape_w, shape_h), (2, 1))
            score_out[4, 3] = 2
            score_out[5, 3] = 1

        fake_cupy = types.SimpleNamespace(
            uint8=np.uint8,
            int32=np.int32,
            asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
            zeros=lambda shape, dtype=None: np.zeros(shape, dtype=dtype),
            asnumpy=lambda arr: arr,
        )

        with patch("agi_core.cuda.mirror_solver.cuda_runtime.CUDA_AVAILABLE", True), patch(
            "agi_core.cuda.mirror_solver.kernel_available",
            return_value=True,
        ), patch(
            "agi_core.cuda.mirror_solver.get_cupy",
            return_value=fake_cupy,
        ), patch(
            "agi_core.cuda.mirror_solver._launch_reflection_score_kernel",
            side_effect=fake_launch,
        ):
            result = score_reflection_candidates_gpu(
                width=8,
                height=8,
                component_cells=[(1, 1), (2, 1)],
                component_bbox=(1, 1, 2, 1),
                goal_cells={(5, 5)},
                horizontal=True,
                axis=4,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result[0]["top_left_x"], 3)
        self.assertEqual(result[0]["top_left_y"], 4)
        self.assertEqual(result[0]["hits"], 2)

    def test_reflection_scoring_falls_back_on_exception(self):
        from agi_core.cuda.mirror_solver import score_reflection_candidates_gpu

        fake_cupy = types.SimpleNamespace(
            uint8=np.uint8,
            int32=np.int32,
            asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
            zeros=lambda shape, dtype=None: np.zeros(shape, dtype=dtype),
            asnumpy=lambda arr: arr,
        )

        with patch("agi_core.cuda.mirror_solver.cuda_runtime.CUDA_AVAILABLE", True), patch(
            "agi_core.cuda.mirror_solver.kernel_available",
            return_value=True,
        ), patch(
            "agi_core.cuda.mirror_solver.get_cupy",
            return_value=fake_cupy,
        ), patch(
            "agi_core.cuda.mirror_solver._launch_reflection_score_kernel",
            side_effect=RuntimeError("boom"),
        ):
            result = score_reflection_candidates_gpu(
                width=8,
                height=8,
                component_cells=[(1, 1)],
                component_bbox=(1, 1, 1, 1),
                goal_cells={(5, 5)},
                horizontal=True,
                axis=4,
            )

        self.assertIsNone(result)

    def test_best_reflection_placement_uses_true_shape_center_for_live_like_shape(self):
        from agi_core.cuda.mirror_solver import best_reflection_placement_gpu
        from agi_core.perception.shape_recognition import _color_components, _detect_background

        board = _load_live_ar25_opening_board()
        background = _detect_background(board)
        components = _color_components(board, background)
        component = next(
            item
            for item in components
            if item.color in (0, 5) and 4 <= len(item.cells) <= 100 and not item.touches_border
        )
        goal_cells = {
            cell
            for item in components
            if item.color == 11
            and not item.touches_border
            and any(x < len(board[0]) - 1 for x, _ in item.cells)
            for cell in item.cells
        }

        def fake_launch(_goal, _shape, _shape_w, _shape_h, _horizontal, _axis, _width, _height, score_out):
            score_out[45, 3] = 40

        fake_cupy = types.SimpleNamespace(
            uint8=np.uint8,
            int32=np.int32,
            asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
            zeros=lambda shape, dtype=None: np.zeros(shape, dtype=dtype),
            asnumpy=lambda arr: arr,
        )

        with patch("agi_core.cuda.mirror_solver.cuda_runtime.CUDA_AVAILABLE", True), patch(
            "agi_core.cuda.mirror_solver.kernel_available",
            return_value=True,
        ), patch(
            "agi_core.cuda.mirror_solver.get_cupy",
            return_value=fake_cupy,
        ), patch(
            "agi_core.cuda.mirror_solver._launch_reflection_score_kernel",
            side_effect=fake_launch,
        ):
            result = best_reflection_placement_gpu(
                width=len(board[0]),
                height=len(board),
                component_cells=component.cells,
                component_bbox=component.bbox,
                remaining_goals=goal_cells,
                horizontal=False,
                axis=31,
            )

        self.assertEqual(result, (40, 8, 47))


class TestGpuPerceptionPipeline(unittest.TestCase):
    """Unified GPU perception pipeline tests."""

    def test_gpu_perceive_importable(self):
        from agi_core.perception.gpu_perception import gpu_perceive, PerceptionResult
        self.assertTrue(callable(gpu_perceive))

    def test_gpu_perceive_returns_none_when_cuda_disabled(self):
        with patch("agi_core.cuda.CUDA_AVAILABLE", False):
            from agi_core.perception.gpu_perception import gpu_perceive
            result = gpu_perceive([[9] * 64 for _ in range(64)])
        self.assertIsNone(result)

    def test_gpu_perceive_populates_component_confidence(self):
        from agi_core.perception.gpu_perception import gpu_perceive

        board = [[9] * 8 for _ in range(8)]
        fake_components = {
            "components": [
                {
                    "cells": [(0, 0), (1, 0)],
                    "bbox": (0, 0, 1, 0),
                    "size": 2,
                    "color": 5,
                    "center": (0, 0),
                    "touches_border": True,
                }
            ],
            "color_frequencies": {5: 2},
        }

        with patch("agi_core.perception.gpu_perception.cuda_runtime.CUDA_AVAILABLE", True), patch(
            "agi_core.perception.gpu_perception.kernel_available",
            return_value=True,
        ), patch(
            "agi_core.perception.gpu_perception.detect_components_gpu",
            return_value=fake_components,
        ), patch(
            "agi_core.perception.gpu_perception.detect_grid_scale_gpu",
            return_value=1,
        ):
            result = gpu_perceive(board, bg_color=9)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.analyzer_components), 1)
        self.assertLess(result.analyzer_components[0].confidence, 1.0)

    def test_analyzer_build_game_model_with_gpu_fallback(self):
        """build_game_model works correctly via CPU fallback."""
        from tests.test_puzzle_games import game_ar25_level1
        from agi_core.orch_or.analyzer import build_game_model
        gm = game_ar25_level1()
        model = build_game_model(gm.board)
        self.assertIsNotNone(model.player_pos)
        self.assertGreater(len(model.components), 0)
        self.assertGreater(model.features.grid_scale, 0)

    def test_shape_recognition_with_gpu_fallback(self):
        """recognize_shape_recognition works correctly via CPU fallback."""
        from tests.test_puzzle_games import game_ar25_level1
        from agi_core.perception.shape_recognition import recognize_shape_recognition
        from agi_core.image_input import BoardImageInput
        gm = game_ar25_level1()
        image_input = BoardImageInput(
            board=gm.board, height=gm.height, width=gm.width,
            frame_signature="test", source_attr="test",
        )
        artifact = recognize_shape_recognition(image_input)
        self.assertIsNotNone(artifact)
        self.assertTrue(artifact["mirror"]["present"])

    def test_gpu_perceive_reuses_frame_local_cache(self):
        from agi_core.perception.gpu_perception import gpu_perceive

        board = [[9] * 8 for _ in range(8)]
        board[1][1] = 5
        fake_components = {
            "components": [
                {
                    "cells": [(1, 1)],
                    "bbox": (1, 1, 1, 1),
                    "size": 1,
                    "color": 5,
                    "center": (1, 1),
                    "touches_border": False,
                }
            ],
            "color_frequencies": {5: 1},
        }

        with patch("agi_core.perception.gpu_perception.cuda_runtime.CUDA_AVAILABLE", True), patch(
            "agi_core.perception.gpu_perception.kernel_available",
            return_value=True,
        ), patch(
            "agi_core.perception.gpu_perception.detect_components_gpu",
            return_value=fake_components,
        ) as detect_components, patch(
            "agi_core.perception.gpu_perception.detect_grid_scale_gpu",
            return_value=1,
        ):
            first = gpu_perceive(board, bg_color=9)
            second = gpu_perceive(board, bg_color=9)

        self.assertIsNotNone(first)
        self.assertIs(first, second)
        self.assertEqual(detect_components.call_count, 1)

    def test_get_cached_perception_skips_board_key_when_cuda_unavailable(self):
        from agi_core.perception.gpu_perception import get_cached_perception

        with patch("agi_core.perception.gpu_perception.cuda_runtime.CUDA_AVAILABLE", False), patch(
            "agi_core.perception.gpu_perception._board_cache_key",
            side_effect=AssertionError("board key should not be computed"),
        ):
            result = get_cached_perception([[9] * 4 for _ in range(4)], bg_color=9)

        self.assertIsNone(result)
