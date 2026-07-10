from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agi_core.config import RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_from_env_defaults_to_local_provider(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.llm_provider, "local")
        self.assertEqual(config.openai_model, "gpt-4.1-mini")
        self.assertEqual(config.openai_api_key_env, "OPENAI_API_KEY")
        self.assertIsNone(config.openai_base_url)
        self.assertIsNone(config.openai_timeout_seconds)
        self.assertTrue(config.multi_agent_enabled)
        self.assertFalse(config.gameplay_disk_io_enabled)
        self.assertIsNone(config.gameplay_memory_db_path())
        self.assertIsNone(config.gameplay_archive_memory_db_path())
        self.assertIsNone(config.gameplay_trained_priors_db_path())
        self.assertTrue(config.cuda_enabled)

    def test_from_env_keeps_memory_and_prior_databases_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = {
                "AGI_MEMORY_DB_PATH": "var/live-memory.sqlite3",
                "AGI_ARCHIVE_MEMORY_DB_PATH": "var/archive-memory.sqlite3",
                "AGI_TRAINED_PRIORS_DB_PATH": "var/offline-priors.sqlite3",
            }

            with patch.dict("os.environ", env, clear=True):
                config = RuntimeConfig.from_env(project_root=root)

            self.assertEqual(
                config.memory_db_path,
                root / "var" / "live-memory.sqlite3",
            )
            self.assertEqual(
                config.archive_memory_db_path,
                root / "var" / "archive-memory.sqlite3",
            )
            self.assertEqual(
                config.trained_priors_db_path,
                root / "var" / "offline-priors.sqlite3",
            )
            self.assertNotEqual(
                config.memory_db_path,
                config.trained_priors_db_path,
            )
            self.assertNotEqual(
                config.memory_db_path,
                config.archive_memory_db_path,
            )

    def test_from_env_reads_custom_terminal_state_names(self) -> None:
        env = {
            "AGI_RESET_STATE_NAMES": "READY,LOSE",
            "AGI_WIN_STATE_NAME": "DONE",
        }

        with patch.dict("os.environ", env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.reset_state_names, ("READY", "LOSE"))
        self.assertEqual(config.win_state_name, "DONE")

    def test_from_env_reads_openai_provider_settings(self) -> None:
        env = {
            "AGI_LLM_PROVIDER": "openai",
            "AGI_OPENAI_MODEL": "gpt-4.1",
            "AGI_OPENAI_BASE_URL": "https://example.invalid/v1",
            "AGI_OPENAI_TIMEOUT_SECONDS": "17.5",
        }

        with patch.dict("os.environ", env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.llm_provider, "openai")
        self.assertEqual(config.openai_model, "gpt-4.1")
        self.assertEqual(config.openai_base_url, "https://example.invalid/v1")
        self.assertEqual(config.openai_timeout_seconds, 17.5)

    def test_from_env_can_enable_gameplay_disk_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = {
                "AGI_GAMEPLAY_DISK_IO": "1",
                "AGI_MEMORY_DB_PATH": "var/live-memory.sqlite3",
                "AGI_ARCHIVE_MEMORY_DB_PATH": "var/archive-memory.sqlite3",
                "AGI_TRAINED_PRIORS_DB_PATH": "var/offline-priors.sqlite3",
            }

            with patch.dict("os.environ", env, clear=True):
                config = RuntimeConfig.from_env(project_root=root)

        self.assertTrue(config.gameplay_disk_io_enabled)
        self.assertEqual(
            config.gameplay_memory_db_path(),
            root / "var" / "live-memory.sqlite3",
        )
        self.assertEqual(
            config.gameplay_archive_memory_db_path(),
            root / "var" / "archive-memory.sqlite3",
        )
        self.assertEqual(
            config.gameplay_trained_priors_db_path(),
            root / "var" / "offline-priors.sqlite3",
        )


if __name__ == "__main__":
    unittest.main()
