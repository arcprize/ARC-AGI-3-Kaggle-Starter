import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_notebook.py"
SPEC = importlib.util.spec_from_file_location("build_notebook", SCRIPT)
build_notebook = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_notebook)


class BuildNotebookTests(unittest.TestCase):
    def setUp(self):
        self.original_agent_dir = build_notebook.AGENT_DIR
        self.temp_dir = tempfile.TemporaryDirectory()
        self.agent_dir = Path(self.temp_dir.name) / "agent"
        self.agent_dir.mkdir()
        build_notebook.AGENT_DIR = self.agent_dir

    def tearDown(self):
        build_notebook.AGENT_DIR = self.original_agent_dir
        self.temp_dir.cleanup()

    def test_single_file_agent_remains_supported(self):
        source = "class MyAgent:\n    pass\n"
        (self.agent_dir / "my_agent.py").write_text(source)

        notebook = build_notebook.build()
        cell_sources = [cell["source"] for cell in notebook["cells"]]

        self.assertIn("!mkdir -p /tmp/submission_agent", cell_sources)
        self.assertIn("%%writefile /tmp/submission_agent/__init__.py\n", cell_sources)
        self.assertIn(
            "%%writefile /tmp/submission_agent/my_agent.py\n" + source,
            cell_sources,
        )
        self.assertTrue(
            any(
                "from submission_agent.my_agent import MyAgent" in source
                for source in cell_sources
            )
        )

    def test_supporting_modules_are_packaged_in_name_order(self):
        (self.agent_dir / "my_agent.py").write_text(
            "from .memory import Memory\n\nclass MyAgent:\n    pass\n"
        )
        (self.agent_dir / "memory.py").write_text("class Memory:\n    pass\n")
        (self.agent_dir / "__init__.py").write_text('"""Agent package."""\n')

        notebook = build_notebook.build()
        write_cells = [
            cell["source"]
            for cell in notebook["cells"]
            if cell["source"].startswith("%%writefile /tmp/submission_agent/")
        ]

        self.assertEqual(
            [source.splitlines()[0] for source in write_cells],
            [
                "%%writefile /tmp/submission_agent/__init__.py",
                "%%writefile /tmp/submission_agent/memory.py",
                "%%writefile /tmp/submission_agent/my_agent.py",
            ],
        )

    def test_missing_entry_point_fails_with_clear_message(self):
        with self.assertRaisesRegex(SystemExit, "my_agent.py"):
            build_notebook.build()


if __name__ == "__main__":
    unittest.main()
