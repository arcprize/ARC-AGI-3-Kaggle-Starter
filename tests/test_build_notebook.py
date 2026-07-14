from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from scripts import build_notebook


class BuildNotebookTests(unittest.TestCase):
    def test_stage_runtime_bundle_and_notebook_cells_include_runtime_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build_notebook_test_") as tmpdir:
            root = Path(tmpdir)
            (root / "agent").mkdir()
            (root / "agi_core").mkdir()
            (root / "models").mkdir()
            (root / "third_party" / "hyperon").mkdir(parents=True)
            (root / "third_party" / "wheels").mkdir(parents=True)
            (root / "notebooks").mkdir()

            (root / "agent" / "my_agent.py").write_text("print('agent')\n")
            (root / "agi_core" / "__init__.py").write_text("__all__ = []\n")
            (root / "models" / "tiny.gguf").write_text("gguf\n")
            (root / "third_party" / "hyperon" / "hyperon.whl").write_text("wheel\n")
            (root / "third_party" / "wheels" / "llama.whl").write_text("wheel\n")

            bundle_dir = root / "notebooks" / "runtime_bundle"
            manifest = build_notebook.stage_runtime_bundle(root=root, bundle_dir=bundle_dir)
            notebook = build_notebook.build(agent_src=root / "agent" / "my_agent.py")

            self.assertTrue((bundle_dir / "agi_core" / "__init__.py").exists())
            self.assertTrue((bundle_dir / "models" / "tiny.gguf").exists())
            self.assertEqual(manifest["hyperon"], ["hyperon.whl"])
            self.assertEqual(manifest["wheels"], ["llama.whl"])

            install_cell = notebook["cells"][1]["source"]
            run_cell = notebook["cells"][3]["source"]
            self.assertIn('manifest.get("hyperon", [])', install_cell)
            self.assertIn("AGI_HYPERON_RULES_PATH", run_cell)
            self.assertIn("AGI_HYPERON_LEARNING_PATH", run_cell)
            self.assertIn("AGI_HYPERON_SKIP_PROBE=1", run_cell)
            self.assertIn("runtime_models", run_cell)
