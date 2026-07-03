from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agi_core.config import RuntimeConfig
from agi_core.llm.llama_cpp_backend import LlamaCppBackend


class FakeTensor:
    def __init__(self, values) -> None:
        self._values = values
        self.shape = (1, len(values))

    def __getitem__(self, item):
        if isinstance(item, tuple):
            _, token_slice = item
            return self._values[token_slice]
        return self._values[item]

    def to(self, device: str):
        return self


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    last_prompt = None
    last_chat_messages = None

    @classmethod
    def from_pretrained(cls, path: str, local_files_only: bool = True):
        return cls()

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        type(self).last_chat_messages = messages
        return "<chat-prompt>"

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        type(self).last_prompt = prompt
        return {"input_ids": FakeTensor([11, 12, 13])}

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return '{"subgoal":"advance-level","confidence":0.8,"reason":"dir model","action_candidates":[{"action":"ACTION1"}]}'


class FakeTransformersModel:
    last_from_pretrained_kwargs = None

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        local_files_only: bool = True,
        low_cpu_mem_usage: bool = False,
        torch_dtype=None,
    ):
        cls.last_from_pretrained_kwargs = {
            "path": path,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "torch_dtype": torch_dtype,
        }
        return cls()

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [FakeTensor([11, 12, 13, 21, 22])]


class LLMBackendTests(unittest.TestCase):
    def test_transformers_directory_backend_loads_and_generates_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("stub\n", encoding="utf-8")

            transformers_module = types.SimpleNamespace(
                AutoTokenizer=FakeTokenizer,
                AutoModelForCausalLM=FakeTransformersModel,
            )
            torch_module = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: True),
                inference_mode=lambda: FakeInferenceMode(),
                float16="float16",
            )

            with patch.dict(
                sys.modules,
                {"transformers": transformers_module, "torch": torch_module},
                clear=False,
            ):
                backend = LlamaCppBackend(model_dir)

            self.assertTrue(backend.available)
            self.assertTrue(
                FakeTransformersModel.last_from_pretrained_kwargs["low_cpu_mem_usage"]
            )
            self.assertEqual(
                FakeTransformersModel.last_from_pretrained_kwargs["torch_dtype"],
                "float16",
            )
            payload = backend.generate_json("prompt", max_tokens=32, temperature=0.1)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["subgoal"], "advance-level")
            self.assertEqual(FakeTokenizer.last_prompt, "<chat-prompt>")
            self.assertEqual(
                FakeTokenizer.last_chat_messages,
                [{"role": "user", "content": "prompt"}],
            )

    def test_transformers_directory_requires_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("stub\n", encoding="utf-8")

            transformers_module = types.SimpleNamespace(
                AutoTokenizer=FakeTokenizer,
                AutoModelForCausalLM=FakeTransformersModel,
            )
            torch_module = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False),
                inference_mode=lambda: FakeInferenceMode(),
                float16="float16",
            )

            with patch.dict(
                sys.modules,
                {"transformers": transformers_module, "torch": torch_module},
                clear=False,
            ):
                backend = LlamaCppBackend(model_dir)

            self.assertFalse(backend.available)
            self.assertEqual(backend.last_error, "CUDA device required for transformers backend")

    def test_gguf_backend_requires_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.gguf"
            model_path.write_text("stub\n", encoding="utf-8")

            with patch.dict(
                sys.modules,
                {"llama_cpp": types.SimpleNamespace(Llama=object, supports_cuda=False)},
                clear=False,
            ):
                backend = LlamaCppBackend(model_path)

            self.assertFalse(backend.available)
            self.assertEqual(backend.last_error, "CUDA device required for llama.cpp backend")

    def test_gguf_backend_can_load_without_torch_when_llama_cpp_reports_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.gguf"
            model_path.write_text("stub\n", encoding="utf-8")

            class FakeLlama:
                last_init_kwargs = None

                def __init__(self, **kwargs) -> None:
                    type(self).last_init_kwargs = dict(kwargs)

            with patch.dict(
                sys.modules,
                {"llama_cpp": types.SimpleNamespace(Llama=FakeLlama, supports_cuda=True)},
                clear=False,
            ):
                with patch("importlib.import_module") as import_module:
                    def side_effect(name: str):
                        if name == "llama_cpp":
                            return sys.modules["llama_cpp"]
                        if name == "torch":
                            raise ImportError("torch not installed")
                        raise ImportError(name)

                    import_module.side_effect = side_effect
                    backend = LlamaCppBackend(model_path)

            self.assertTrue(backend.available)
            self.assertEqual(FakeLlama.last_init_kwargs["n_gpu_layers"], -1)

    def test_gguf_backend_enables_full_gpu_offload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.gguf"
            model_path.write_text("stub\n", encoding="utf-8")

            class FakeLlama:
                last_init_kwargs = None

                def __init__(self, **kwargs) -> None:
                    type(self).last_init_kwargs = dict(kwargs)

            llama_cpp_module = types.SimpleNamespace(Llama=FakeLlama)
            torch_module = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: True),
            )

            with patch.dict(
                sys.modules,
                {"llama_cpp": llama_cpp_module, "torch": torch_module},
                clear=False,
            ):
                backend = LlamaCppBackend(model_path)

            self.assertTrue(backend.available)
            self.assertEqual(FakeLlama.last_init_kwargs["n_gpu_layers"], -1)

    def test_effective_model_path_prefers_transformers_directory_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "models" / "gemma-dir"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")

            with patch.dict(os.environ, {"AGI_LLM_MODEL_PATH": str(model_dir)}, clear=False):
                config = RuntimeConfig.from_env(project_root=root)

            self.assertEqual(config.effective_model_path(), model_dir)

    def test_coerce_json_recovers_fenced_fragment_missing_final_brace(self) -> None:
        backend = LlamaCppBackend(model_path=None)
        payload = backend._coerce_json(
            "```json\n"
            "{\n"
            '  "subgoal": "advance",\n'
            '  "action_candidates": [{"action": "ACTION1"}]\n'
            "```"
        )

        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["subgoal"], "advance")


if __name__ == "__main__":
    unittest.main()
