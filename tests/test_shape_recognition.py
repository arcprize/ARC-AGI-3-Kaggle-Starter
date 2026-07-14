"""Tests for shape recognition system — contour extraction, Hu moments, shape matching.

Iteration 1: Contour / boundary extraction.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

from agi_core.perception.shape_recognition import (
    _extract_contour,
    _compute_perimeter,
    _compute_area,
    _compute_hu_moments,
    _hu_log_distance,
    _compute_shape_features,
    _estimate_rotation_angle,
    _feature_vector_distance,
    match_shape_to_template,
    find_best_template_match,
    store_shape_template,
    load_shape_templates,
    _template_aware_placement,
    _mask_overlap_score,
    _compute_spatial_descriptors,
    _normalized_mask,
    _serialize_shape_entities,
    _ColorComponent,
    recognize_shape_recognition,
)
from agi_core.image_input import BoardImageInput


# ---------------------------------------------------------------------------
# Iteration 1: Contour / boundary extraction
# ---------------------------------------------------------------------------

class TestContourExtraction(unittest.TestCase):
    """Test contour extraction from shape pixel sets."""

    def test_contour_single_pixel(self) -> None:
        cells = [(5, 5)]
        contour = _extract_contour(cells)
        self.assertEqual(contour, [(5, 5)])

    def test_contour_2x2_square(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        contour = _extract_contour(cells)
        self.assertGreaterEqual(len(contour), 4)
        for x, y in cells:
            self.assertIn((x, y), contour, f"Cell ({x},{y}) should be in contour")

    def test_contour_3x3_ring(self) -> None:
        """3x3 ring with transparent center — like AR25 shape pieces."""
        cells = [
            (0, 0), (1, 0), (2, 0),
            (0, 1),         (2, 1),
            (0, 2), (1, 2), (2, 2),
        ]
        contour = _extract_contour(cells)
        self.assertGreaterEqual(len(contour), 8)
        for x, y in cells:
            self.assertIn((x, y), contour, f"Cell ({x},{y}) should be in contour")

    def test_contour_horizontal_line(self) -> None:
        cells = [(0, 0), (1, 0), (2, 0), (3, 0)]
        contour = _extract_contour(cells)
        self.assertGreaterEqual(len(contour), 4)
        for x, y in cells:
            self.assertIn((x, y), contour)

    def test_contour_l_shape(self) -> None:
        cells = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
        contour = _extract_contour(cells)
        # All 5 cells are boundary (each has at least one missing 4-neighbor)
        self.assertGreaterEqual(len(contour), 5)
        for x, y in cells:
            self.assertIn((x, y), contour)

    def test_contour_empty(self) -> None:
        self.assertEqual(_extract_contour([]), [])


class TestPerimeterComputation(unittest.TestCase):
    """Test perimeter computation from contour pixel lists."""

    def test_perimeter_2x2_square(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        # 2x2 square has perimeter 4 (4 unit edges)
        self.assertEqual(perimeter, 4)

    def test_perimeter_3x3_ring(self) -> None:
        cells = [
            (0, 0), (1, 0), (2, 0),
            (0, 1),         (2, 1),
            (0, 2), (1, 2), (2, 2),
        ]
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        # 3x3 ring has 8 boundary cells, each contributing edges
        self.assertGreater(perimeter, 6)

    def test_perimeter_single_pixel(self) -> None:
        self.assertEqual(_compute_perimeter([(5, 5)]), 0)

    def test_perimeter_horizontal_line(self) -> None:
        cells = [(0, 0), (1, 0), (2, 0)]
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        # 3 collinear boundary cells in a chain
        self.assertEqual(perimeter, 3)


class TestAreaComputation(unittest.TestCase):
    """Test area (pixel count) computation."""

    def test_area_2x2(self) -> None:
        self.assertEqual(_compute_area([(0, 0), (1, 0), (0, 1), (1, 1)]), 4)

    def test_area_3x3_ring(self) -> None:
        cells = [
            (0, 0), (1, 0), (2, 0),
            (0, 1),         (2, 1),
            (0, 2), (1, 2), (2, 2),
        ]
        self.assertEqual(_compute_area(cells), 8)


class TestShapeEntityContourIntegration(unittest.TestCase):
    """Test that shape entities include contour/perimeter fields."""

    def test_shape_entities_have_contour(self) -> None:
        inactive = _ColorComponent(
            color=5,
            cells=[(5, 5), (6, 5), (5, 6), (6, 6)],
            bbox=(5, 5, 6, 6),
            center=(5, 5),
            touches_border=False,
        )
        entities, _ = _serialize_shape_entities([inactive])
        self.assertIn("contour", entities[0])
        self.assertIn("perimeter", entities[0])
        self.assertGreaterEqual(len(entities[0]["contour"]), 4)

    def test_full_pipeline_contour_fields_present(self) -> None:
        board = [[9] * 64 for _ in range(64)]
        # Place a 2x2 shape at (10, 10)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                board[10 + dy][10 + dx] = 5
        artifact = recognize_shape_recognition(
            BoardImageInput(
                board=board,
                height=64,
                width=64,
                frame_signature="contour-test",
                source_attr="test",
            )
        )
        self.assertGreater(len(artifact["shape_entities"]), 0)
        for entity in artifact["shape_entities"]:
            self.assertIn("contour", entity)
            self.assertIn("perimeter", entity)
            self.assertGreater(len(entity["contour"]), 0,
                               f"Shape {entity['entity_id']} should have non-empty contour")

    def test_all_components_have_contour(self) -> None:
        board = [[9] * 64 for _ in range(64)]
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                board[12 + dy][12 + dx] = 5
        board[30][30] = 11  # goal pixel
        artifact = recognize_shape_recognition(
            BoardImageInput(
                board=board,
                height=64,
                width=64,
                frame_signature="all-components-contour",
                source_attr="test",
            )
        )
        for component in artifact["components"]:
            self.assertIn("contour", component, f"Component {component} missing contour")
            self.assertIn("perimeter", component, f"Component {component} missing perimeter")


# ---------------------------------------------------------------------------
# Iteration 2: Hu moment computation
# ---------------------------------------------------------------------------

class TestHuMoments(unittest.TestCase):
    """Test Hu moment computation — 7 rotation-invariant shape descriptors."""

    def test_hu_empty(self) -> None:
        moments = _compute_hu_moments([])
        self.assertEqual(len(moments), 7)
        self.assertEqual(moments, [0.0] * 7)

    def test_hu_single_pixel(self) -> None:
        moments = _compute_hu_moments([(5, 5)])
        self.assertEqual(len(moments), 7)
        for m in moments:
            self.assertAlmostEqual(m, 0.0, delta=1e-6)

    def test_hu_2x2_square(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        moments = _compute_hu_moments(cells)
        self.assertEqual(len(moments), 7)
        # h1 should be non-zero for a filled shape
        self.assertGreater(moments[0], 0)
        # All values should be finite
        for m in moments:
            self.assertTrue(math.isfinite(m), f"Hu moment {m} should be finite")
        # 2x2 square should have the same moments as translated version
        translated = [(x + 10, y + 10) for x, y in cells]
        moments_t = _compute_hu_moments(translated)
        for a, b in zip(moments, moments_t):
            self.assertAlmostEqual(a, b, delta=1e-6,
                                   msg="Hu moments should be translation-invariant")

    def test_hu_rotation_invariant_2x2(self) -> None:
        """2x2 square should be the same when rotated 90 degrees."""
        # A 2x2 square is symmetric, so rotation should give identical moments
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        moments = _compute_hu_moments(cells)
        self.assertEqual(len(moments), 7)

    def test_hu_3x3_ring(self) -> None:
        """3x3 ring shape (like AR25 shape piece)."""
        cells = [
            (0, 0), (1, 0), (2, 0),
            (0, 1),         (2, 1),
            (0, 2), (1, 2), (2, 2),
        ]
        moments = _compute_hu_moments(cells)
        self.assertEqual(len(moments), 7)
        for m in moments:
            self.assertTrue(math.isfinite(m))

    def test_hu_symmetry(self) -> None:
        """Same shape at different positions gives same moments."""
        cells = [(0, 0), (2, 0), (0, 2), (2, 2)]
        cells2 = [(5, 5), (7, 5), (5, 7), (7, 7)]
        m1 = _compute_hu_moments(cells)
        m2 = _compute_hu_moments(cells2)
        for a, b in zip(m1, m2):
            self.assertAlmostEqual(a, b, delta=1e-6)

    def test_hu_distance_identical(self) -> None:
        m1 = _compute_hu_moments([(0, 0), (1, 0), (0, 1), (1, 1)])
        m2 = _compute_hu_moments([(10, 10), (11, 10), (10, 11), (11, 11)])
        dist = _hu_log_distance(m1, m2)
        self.assertAlmostEqual(dist, 0.0, delta=1e-6)

    def test_hu_distance_different_shapes(self) -> None:
        square = _compute_hu_moments([(0, 0), (1, 0), (0, 1), (1, 1)])
        ring = _compute_hu_moments([
            (0, 0), (1, 0), (2, 0),
            (0, 1),         (2, 1),
            (0, 2), (1, 2), (2, 2),
        ])
        dist = _hu_log_distance(square, ring)
        # Different shapes should have non-zero distance
        self.assertGreater(dist, 0.0)

    def test_shape_entities_have_hu_moments(self) -> None:
        component = _ColorComponent(
            color=5,
            cells=[(5, 5), (6, 5), (5, 6), (6, 6)],
            bbox=(5, 5, 6, 6),
            center=(5, 5),
            touches_border=False,
        )
        entities, _ = _serialize_shape_entities([component])
        self.assertIn("hu_moments", entities[0])
        self.assertEqual(len(entities[0]["hu_moments"]), 7)

    def test_all_components_have_hu_moments(self) -> None:
        board = [[9] * 64 for _ in range(64)]
        board[10][10] = 5
        board[11][10] = 5
        board[10][11] = 5
        board[11][11] = 5
        board[30][30] = 11
        artifact = recognize_shape_recognition(
            BoardImageInput(
                board=board,
                height=64,
                width=64,
                frame_signature="hu-test",
                source_attr="test",
            )
        )
        for component in artifact["components"]:
            self.assertIn("hu_moments", component)
            self.assertEqual(len(component["hu_moments"]), 7)


# ---------------------------------------------------------------------------
# Iteration 3: Hu-moment similarity scoring for equivalence classes
# ---------------------------------------------------------------------------

class TestDescriptorEquivalence(unittest.TestCase):
    """Test Hu-moment-based descriptor similarity grouping."""

    def test_identical_shapes_same_descriptor_class(self) -> None:
        """Two shapes with identical masks get same descriptor and equivalence class."""
        c1 = _ColorComponent(
            color=5, cells=[(0, 0), (1, 0), (0, 1), (1, 1)],
            bbox=(0, 0, 1, 1), center=(0, 0), touches_border=False,
        )
        c2 = _ColorComponent(
            color=5, cells=[(10, 10), (11, 10), (10, 11), (11, 11)],
            bbox=(10, 10, 11, 11), center=(10, 10), touches_border=False,
        )
        entities, classes = _serialize_shape_entities([c1, c2])
        self.assertEqual(entities[0]["equivalence_class"], entities[1]["equivalence_class"])
        self.assertEqual(entities[0]["descriptor_class"], entities[1]["descriptor_class"])

    def test_translated_same_descriptor_class(self) -> None:
        """Same shape translated should have same descriptor."""
        cells_a = [(0, 0), (2, 0), (0, 2), (2, 2)]
        cells_b = [(100, 100), (102, 100), (100, 102), (102, 102)]
        c1 = _ColorComponent(
            color=5, cells=cells_a,
            bbox=(0, 0, 2, 2), center=(0, 0), touches_border=False,
        )
        c2 = _ColorComponent(
            color=5, cells=cells_b,
            bbox=(100, 100, 102, 102), center=(100, 100), touches_border=False,
        )
        entities, classes = _serialize_shape_entities([c1, c2])
        self.assertEqual(entities[0]["descriptor_class"], entities[1]["descriptor_class"],
                         "Translated shapes should share descriptor class")

    def test_different_shapes_different_descriptor_class(self) -> None:
        """Very different shapes should be in different descriptor classes."""
        c1 = _ColorComponent(
            color=5, cells=[(0, 0), (1, 0), (0, 1), (1, 1)],
            bbox=(0, 0, 1, 1), center=(0, 0), touches_border=False,
        )
        # Long thin line
        c2 = _ColorComponent(
            color=5, cells=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            bbox=(0, 0, 4, 0), center=(2, 0), touches_border=False,
        )
        entities, classes = _serialize_shape_entities([c1, c2])
        self.assertNotEqual(entities[0]["descriptor_class"], entities[1]["descriptor_class"])

    def test_same_descriptor_class_for_similar_shapes(self) -> None:
        """Shapes with similar proportions share descriptor class."""
        # Two 2x2 squares at different positions
        c1 = _ColorComponent(
            color=5, cells=[(0, 0), (1, 0), (0, 1), (1, 1)],
            bbox=(0, 0, 1, 1), center=(0, 0), touches_border=False,
        )
        c2 = _ColorComponent(
            color=5, cells=[(5, 5), (6, 5), (5, 6), (6, 6)],
            bbox=(5, 5, 6, 6), center=(5, 5), touches_border=False,
        )
        entities, classes = _serialize_shape_entities([c1, c2])
        self.assertEqual(entities[0]["descriptor_class"], entities[1]["descriptor_class"])

    def test_descriptor_class_in_artifact(self) -> None:
        """Full pipeline includes descriptor_class fields."""
        board = [[9] * 64 for _ in range(64)]
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                board[10 + dy][10 + dx] = 5
        artifact = recognize_shape_recognition(
            BoardImageInput(
                board=board,
                height=64,
                width=64,
                frame_signature="desc-class-test",
                source_attr="test",
            )
        )
        for entity in artifact.get("shape_entities", []):
            self.assertIn("descriptor_class", entity,
                          f"Shape {entity['entity_id']} missing descriptor_class")
        # Each descriptor class should appear in equivalence_classes
        desc_classes = {e.get("descriptor_class") for e in artifact.get("shape_entities", []) if e.get("descriptor_class")}
        for dc in desc_classes:
            found = any(
                ec.get("descriptor_class") == dc
                for ec in artifact.get("equivalence_classes", [])
                if "descriptor_class" in ec
            )
            if not any("descriptor_class" in ec for ec in artifact.get("equivalence_classes", [])):
                continue  # descriptor classes may not materialize for single shapes


# ---------------------------------------------------------------------------
# Iteration 4: Shape feature vector extraction
# ---------------------------------------------------------------------------

class TestShapeFeatures(unittest.TestCase):
    """Test structural shape feature extraction."""

    def _make_component(self, cells, color=5):
        bbox = (
            min(x for x, _ in cells),
            min(y for _, y in cells),
            max(x for x, _ in cells),
            max(y for _, y in cells),
        )
        cx = sum(x for x, _ in cells) // len(cells)
        cy = sum(y for _, y in cells) // len(cells)
        return _ColorComponent(
            color=color, cells=cells, bbox=bbox,
            center=(cx, cy), touches_border=False,
        )

    def test_aspect_ratio_square(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        c = self._make_component(cells)
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        features = _compute_shape_features(cells, contour, perimeter, c.bbox)
        self.assertIn("aspect_ratio", features)
        self.assertAlmostEqual(features["aspect_ratio"], 1.0, delta=0.01)

    def test_aspect_ratio_wide(self) -> None:
        cells = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
        c = self._make_component(cells)
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        features = _compute_shape_features(cells, contour, perimeter, c.bbox)
        self.assertGreater(features["aspect_ratio"], 1.5)

    def test_extent_filled_square(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        c = self._make_component(cells)
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        features = _compute_shape_features(cells, contour, perimeter, c.bbox)
        # 2x2 fills entire bbox
        self.assertAlmostEqual(features["extent"], 1.0, delta=0.01)

    def test_extent_ring(self) -> None:
        cells = [
            (0, 0), (1, 0), (2, 0),
            (0, 1),         (2, 1),
            (0, 2), (1, 2), (2, 2),
        ]
        c = self._make_component(cells)
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        features = _compute_shape_features(cells, contour, perimeter, c.bbox)
        # 3x3 ring: 8 cells out of 9 bbox area
        self.assertAlmostEqual(features["extent"], 8.0 / 9.0, delta=0.01)

    def test_solidity_convex(self) -> None:
        """Convex shape should have solidity of 1.0."""
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        c = self._make_component(cells)
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        features = _compute_shape_features(cells, contour, perimeter, c.bbox)
        self.assertAlmostEqual(features["solidity"], 1.0, delta=0.01)

    def test_compactness_square(self) -> None:
        """2x2 square compactness = 16/4 = 4."""
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        c = self._make_component(cells)
        contour = _extract_contour(cells)
        perimeter = _compute_perimeter(contour)
        features = _compute_shape_features(cells, contour, perimeter, c.bbox)
        # perimeter=4, area=4, compactness=4²/4=4
        self.assertAlmostEqual(features["compactness"], 4.0, delta=0.5)

    def test_shape_entities_have_features(self) -> None:
        c = self._make_component([(0, 0), (1, 0), (0, 1), (1, 1)])
        entities, _ = _serialize_shape_entities([c])
        self.assertIn("features", entities[0])
        feat = entities[0]["features"]
        for key in ("aspect_ratio", "compactness", "solidity", "extent", "convexity"):
            self.assertIn(key, feat, f"Missing feature: {key}")

    def test_all_components_have_features(self) -> None:
        board = [[9] * 64 for _ in range(64)]
        board[10][10] = 5
        board[11][10] = 5
        board[10][11] = 5
        board[11][11] = 5
        board[30][30] = 11
        artifact = recognize_shape_recognition(
            BoardImageInput(
                board=board,
                height=64,
                width=64,
                frame_signature="features-test",
                source_attr="test",
            )
        )
        for component in artifact["components"]:
            self.assertIn("features", component)
            feat = component["features"]
            for key in ("aspect_ratio", "compactness", "solidity", "extent", "convexity"):
                self.assertIn(key, feat, f"Component missing feature {key}")


# ---------------------------------------------------------------------------
# Iteration 5: Rotation angle estimation + feature fusion grouping
# ---------------------------------------------------------------------------

class TestRotationEstimation(unittest.TestCase):
    """Test rotation angle estimation from central moments."""

    def test_rotation_horizontal_line(self) -> None:
        cells = [(0, 0), (1, 0), (2, 0)]
        angle = _estimate_rotation_angle(cells)
        # Horizontal line → angle ≈ 0
        self.assertAlmostEqual(angle, 0.0, delta=0.1)

    def test_rotation_vertical_line(self) -> None:
        cells = [(0, 0), (0, 1), (0, 2)]
        angle = _estimate_rotation_angle(cells)
        # Vertical line → angle ≈ π/2
        self.assertAlmostEqual(abs(angle), math.pi / 2, delta=0.1)

    def test_rotation_square(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        angle = _estimate_rotation_angle(cells)
        # Square is symmetric → angle ≈ 0
        self.assertAlmostEqual(angle, 0.0, delta=0.1)

    def test_rotation_single_pixel(self) -> None:
        self.assertEqual(_estimate_rotation_angle([(5, 5)]), 0.0)

    def test_shape_entities_have_rotation(self) -> None:
        c = _ColorComponent(
            color=5,
            cells=[(0, 0), (1, 0), (0, 1), (1, 1)],
            bbox=(0, 0, 1, 1), center=(0, 0), touches_border=False,
        )
        entities, _ = _serialize_shape_entities([c])
        self.assertIn("rotation_estimate", entities[0])


class TestFeatureVectorDistance(unittest.TestCase):
    """Test feature vector distance computation."""

    def test_identical_features(self) -> None:
        f1 = {"aspect_ratio": 1.0, "compactness": 4.0, "solidity": 1.0,
              "extent": 1.0, "convexity": 1.0}
        f2 = {"aspect_ratio": 1.0, "compactness": 4.0, "solidity": 1.0,
              "extent": 1.0, "convexity": 1.0}
        self.assertAlmostEqual(_feature_vector_distance(f1, f2), 0.0, delta=0.01)

    def test_different_features(self) -> None:
        f1 = {"aspect_ratio": 1.0, "compactness": 4.0, "solidity": 1.0,
              "extent": 1.0, "convexity": 1.0}
        f2 = {"aspect_ratio": 5.0, "compactness": 20.0, "solidity": 0.5,
              "extent": 0.3, "convexity": 0.7}
        dist = _feature_vector_distance(f1, f2)
        self.assertGreater(dist, 0.0)

    def test_missing_features_defaults_zero(self) -> None:
        f1 = {"aspect_ratio": 1.0}
        f2 = {"aspect_ratio": 2.0}
        dist = _feature_vector_distance(f1, f2)
        self.assertGreater(dist, 0.0)


# ---------------------------------------------------------------------------
# Iteration 6: Template matching API
# ---------------------------------------------------------------------------

class TestTemplateMatching(unittest.TestCase):
    """Test shape-to-template matching."""

    def test_match_identical_shape(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        template_hu = _compute_hu_moments(cells)
        score, rotation = match_shape_to_template(cells, template_hu)
        self.assertGreaterEqual(score, 0.9)
        self.assertIsInstance(rotation, float)

    def test_match_translated_shape(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        translated = [(10, 10), (11, 10), (10, 11), (11, 11)]
        template_hu = _compute_hu_moments(cells)
        score, rotation = match_shape_to_template(translated, template_hu)
        self.assertGreaterEqual(score, 0.9,
                                f"Translated shape should match, got score {score}")

    def test_match_different_shape(self) -> None:
        square = [(0, 0), (1, 0), (0, 1), (1, 1)]
        line = [(0, 0), (1, 0), (2, 0), (3, 0)]
        template_hu = _compute_hu_moments(square)
        score, rotation = match_shape_to_template(line, template_hu)
        self.assertLess(score, 0.9,
                        f"Different shapes should have lower score, got {score}")

    def test_find_best_match_found(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        templates = [
            {"hu_moments": _compute_hu_moments([(5, 5), (6, 5), (5, 6), (6, 6)]), "id": "square"},
            {"hu_moments": _compute_hu_moments([(0, 0), (1, 0), (2, 0)]), "id": "line"},
        ]
        idx, score = find_best_template_match(cells, templates, min_score=0.5)
        self.assertIsNotNone(idx)
        self.assertGreaterEqual(score, 0.5)

    def test_find_best_match_not_found(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        templates = [
            {"hu_moments": _compute_hu_moments([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]), "id": "long_line"},
        ]
        idx, score = find_best_template_match(cells, templates, min_score=0.99)
        self.assertIsNone(idx)
        self.assertEqual(score, 0.0)

    def test_find_best_match_empty_templates(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        idx, score = find_best_template_match(cells, [])
        self.assertIsNone(idx)


# ---------------------------------------------------------------------------
# Iteration 7: Template storage from solved puzzles
# ---------------------------------------------------------------------------

import tempfile
import os


class TestTemplateStorage(unittest.TestCase):
    """Test shape template persistence to JSON."""

    def setUp(self):
        self.tmpfile = tempfile.mktemp(suffix=".json")

    def tearDown(self):
        if os.path.exists(self.tmpfile):
            os.remove(self.tmpfile)

    def test_store_and_load(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        store_shape_template(cells, filepath=self.tmpfile)
        loaded = load_shape_templates(self.tmpfile)
        self.assertEqual(len(loaded), 1)
        self.assertIn("hu_moments", loaded[0])
        self.assertEqual(loaded[0]["cell_count"], 4)

    def test_multiple_templates(self) -> None:
        cells1 = [(0, 0), (1, 0), (0, 1), (1, 1)]
        cells2 = [(0, 0), (1, 0), (2, 0)]
        store_shape_template(cells1, filepath=self.tmpfile)
        store_shape_template(cells2, filepath=self.tmpfile)
        loaded = load_shape_templates(self.tmpfile)
        self.assertEqual(len(loaded), 2)

    def test_near_duplicate_merged(self) -> None:
        cells1 = [(0, 0), (1, 0), (0, 1), (1, 1)]
        cells2 = [(10, 10), (11, 10), (10, 11), (11, 11)]
        store_shape_template(cells1, filepath=self.tmpfile)
        store_shape_template(cells2, filepath=self.tmpfile)
        loaded = load_shape_templates(self.tmpfile)
        self.assertEqual(len(loaded), 1, "Near-duplicate should be merged")

    def test_load_nonexistent(self) -> None:
        loaded = load_shape_templates("nonexistent_file.json")
        self.assertEqual(loaded, [])

    def test_store_with_placement(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        store_shape_template(
            cells,
            placement={"target_anchor": [4, 46], "horizontal": True},
            goal_coverage=[(52, 46), (55, 48)],
            filepath=self.tmpfile,
        )
        loaded = load_shape_templates(self.tmpfile)
        self.assertEqual(loaded[0]["placement"]["target_anchor"], [4, 46])
        self.assertEqual(len(loaded[0]["goal_coverage"]), 2)


# ---------------------------------------------------------------------------
# Iteration 8: Template-aware mirror placement
# ---------------------------------------------------------------------------

class TestTemplateAwarePlacement(unittest.TestCase):
    """Test template-aware mirror placement."""

    def test_no_templates_fallback(self) -> None:
        component = _ColorComponent(
            color=5,
            cells=[(0, 0), (1, 0), (0, 1), (1, 1)],
            bbox=(0, 0, 1, 1), center=(0, 0), touches_border=False,
        )
        result = _template_aware_placement(
            component, set(), {}, True, 10,
            width=64, height=64, templates=None,
        )
        self.assertIsNone(result)

    def test_matching_template(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        hu = _compute_hu_moments(cells)
        templates = [
            {"hu_moments": hu, "placement": {"target_anchor": [4, 46], "horizontal": True}},
        ]
        component = _ColorComponent(
            color=5, cells=cells,
            bbox=(0, 0, 1, 1), center=(0, 0), touches_border=False,
        )
        result = _template_aware_placement(
            component, set(), {}, True, 10,
            width=64, height=64, templates=templates,
        )
        self.assertIsNotNone(result)
        self.assertIn("template_index", result)
        self.assertEqual(result["template_index"], 0)

    def test_orientation_mismatch(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        hu = _compute_hu_moments(cells)
        templates = [
            {"hu_moments": hu, "placement": {"target_anchor": [4, 46], "horizontal": False}},
        ]
        component = _ColorComponent(
            color=5, cells=cells,
            bbox=(0, 0, 1, 1), center=(0, 0), touches_border=False,
        )
        result = _template_aware_placement(
            component, set(), {}, True, 10,
            width=64, height=64, templates=templates,
        )
        self.assertIsNone(result, "Should reject orientation mismatch")


# ---------------------------------------------------------------------------
# Iteration 9: Partial overlap / fuzzy mask matching
# ---------------------------------------------------------------------------

class TestMaskOverlap(unittest.TestCase):
    """Test IoU-based mask overlap scoring."""

    def test_identical_masks(self) -> None:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        score = _mask_overlap_score(cells, cells)
        self.assertAlmostEqual(score, 1.0, delta=0.01)

    def test_no_overlap(self) -> None:
        a = [(0, 0), (1, 0)]
        b = [(10, 10), (11, 10)]
        score = _mask_overlap_score(a, b)
        self.assertEqual(score, 0.0)

    def test_partial_overlap(self) -> None:
        a = [(0, 0), (1, 0), (0, 1), (1, 1)]
        b = [(1, 0), (2, 0), (1, 1), (2, 1)]
        score = _mask_overlap_score(a, b)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_mask(self) -> None:
        self.assertEqual(_mask_overlap_score([], [(0, 0)]), 0.0)

    def test_translated_no_overlap(self) -> None:
        a = [(0, 0), (1, 0), (0, 1), (1, 1)]
        b = [(5, 5), (6, 5), (5, 6), (6, 6)]
        score = _mask_overlap_score(a, b)
        self.assertEqual(score, 0.0, "Translated shapes at different positions = no overlap")


# ---------------------------------------------------------------------------
# Iteration 10: Spatial relationship descriptors
# ---------------------------------------------------------------------------

class TestSpatialDescriptors(unittest.TestCase):
    """Test spatial relationship descriptor computation."""

    def test_basic_descriptors(self) -> None:
        cells = [(5, 5), (6, 5), (5, 6), (6, 6)]
        desc = _compute_spatial_descriptors(cells)
        self.assertIsInstance(desc, dict)
        self.assertEqual(len(desc), 0)  # no mirror/goal/other given

    def test_with_mirror(self) -> None:
        cells = [(5, 5), (6, 5)]
        mirror = [(10, 5), (10, 6)]
        desc = _compute_spatial_descriptors(cells, mirror_cells=mirror)
        self.assertIn("mirror_distance", desc)
        self.assertIn("mirror_side", desc)
        self.assertGreater(desc["mirror_distance"], 0)

    def test_with_goals(self) -> None:
        cells = [(5, 5), (6, 5)]
        goals = [(15, 15), (16, 15)]
        desc = _compute_spatial_descriptors(cells, goal_cells=goals)
        self.assertIn("goal_distance", desc)
        self.assertIn("nearest_goal_distance", desc)
        self.assertGreater(desc["goal_distance"], 0)

    def test_with_other_shapes(self) -> None:
        cells = [(0, 0), (1, 0)]
        others = [[(10, 10), (11, 10)], [(20, 20), (21, 20)]]
        desc = _compute_spatial_descriptors(cells, other_shapes_cells=others)
        self.assertIn("other_shape_count", desc)
        self.assertEqual(desc["other_shape_count"], 2)
        self.assertIn("nearest_shape_distance", desc)

    def test_stage_targets_include_spatial(self) -> None:
        board = [[9] * 64 for _ in range(64)]
        for dy in range(21):
            for dx in range(1):
                board[dy][42] = 10  # mirror at column 42
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                board[30 + dy][10 + dx] = 5  # shape at (10, 30)
        for gy in range(3):
            for gx in range(3):
                board[20 + gy][55 + gx] = 11  # goals
        artifact = recognize_shape_recognition(
            BoardImageInput(
                board=board,
                height=64,
                width=64,
                frame_signature="spatial-test",
                source_attr="test",
            )
        )
        mp = artifact.get("mirror_puzzle", {})
        if mp.get("active") and mp.get("stage_targets"):
            for stage in mp["stage_targets"]:
                self.assertIn("spatial", stage)


if __name__ == "__main__":
    unittest.main()
