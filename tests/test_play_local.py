from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


class PlayLocalTests(unittest.TestCase):
    def _import_play_local_module(self):
        fake_arc_agi = types.ModuleType("arc_agi")
        fake_arc_agi.OperationMode = types.SimpleNamespace(NORMAL="NORMAL")
        fake_arc_agi.Arcade = object
        with patch.dict(sys.modules, {"arc_agi": fake_arc_agi}, clear=False):
            sys.modules.pop("scripts.play_local", None)
            module = importlib.import_module("scripts.play_local")
            return importlib.reload(module)

    def test_local_runner_env_defaults_to_local(self) -> None:
        module = self._import_play_local_module()
        with patch.dict(os.environ, {"AGI_LLM_PROVIDER": "openai"}, clear=False):
            module._configure_local_runner_env("local")
            self.assertEqual(os.environ["AGI_LLM_PROVIDER"], "local")

    def test_local_runner_env_can_explicitly_use_openai(self) -> None:
        module = self._import_play_local_module()
        with patch.dict(os.environ, {}, clear=False):
            module._configure_local_runner_env("openai")
            self.assertEqual(os.environ["AGI_LLM_PROVIDER"], "openai")

    def test_local_runner_env_clears_broken_proxy_defaults(self) -> None:
        module = self._import_play_local_module()
        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "example.com",
            },
            clear=False,
        ):
            module._configure_local_runner_env("local")
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertNotIn("ALL_PROXY", os.environ)
            self.assertEqual(
                os.environ["NO_PROXY"],
                "example.com,localhost,127.0.0.1",
            )
            self.assertEqual(os.environ["no_proxy"], os.environ["NO_PROXY"])

    def test_local_runner_env_can_preserve_proxy_when_requested(self) -> None:
        module = self._import_play_local_module()
        with patch.dict(
            os.environ,
            {
                "AGI_PLAY_LOCAL_PRESERVE_PROXY": "1",
                "HTTP_PROXY": "http://127.0.0.1:9",
            },
            clear=False,
        ):
            module._configure_local_runner_env("local")
            self.assertEqual(os.environ["HTTP_PROXY"], "http://127.0.0.1:9")


if __name__ == "__main__":
    unittest.main()
