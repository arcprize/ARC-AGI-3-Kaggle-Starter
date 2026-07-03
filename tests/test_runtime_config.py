from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agi_core.config import RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_from_env_keeps_memory_and_prior_databases_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = {
                "AGI_MEMORY_DB_PATH": "var/live-memory.sqlite3",
                "AGI_TRAINED_PRIORS_DB_PATH": "var/offline-priors.sqlite3",
            }

            with patch.dict("os.environ", env, clear=False):
                config = RuntimeConfig.from_env(project_root=root)

            self.assertEqual(
                config.memory_db_path,
                root / "var" / "live-memory.sqlite3",
            )
            self.assertEqual(
                config.trained_priors_db_path,
                root / "var" / "offline-priors.sqlite3",
            )
            self.assertNotEqual(
                config.memory_db_path,
                config.trained_priors_db_path,
            )

    def test_from_env_reads_custom_terminal_state_names(self) -> None:
        env = {
            "AGI_RESET_STATE_NAMES": "READY,LOSE",
            "AGI_WIN_STATE_NAME": "DONE",
        }

        with patch.dict("os.environ", env, clear=False):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.reset_state_names, ("READY", "LOSE"))
        self.assertEqual(config.win_state_name, "DONE")


if __name__ == "__main__":
    unittest.main()
