from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

from .json_utils import coerce_json_dict


class LlamaCppBackend:
    """Optional local inference backend for GGUF or Transformers checkpoints."""

    def __init__(
        self,
        model_path: str | Path | None,
        n_ctx: int = 2048,
        device: str = "cuda:0",
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.n_ctx = n_ctx
        self.device = device
        self.available = False
        self.last_error: str | None = None
        self._llm: Any = None
        self._backend_kind: str | None = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._transformers_dtype: Any = None
        self._load()

    def generate_json(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any] | None:
        if not self.available or self._llm is None:
            return None
        try:
            if self._backend_kind == "transformers":
                return self._generate_with_transformers(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            if hasattr(self._llm, "create_chat_completion"):
                response = self._llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                text = response["choices"][0]["message"]["content"]
            else:
                response = self._llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response["choices"][0]["text"]
            payload = self._coerce_json(text)
            if payload is None:
                payload = self._coerce_plain_action_response(text)
            if payload is None:
                self.last_error = self._invalid_json_error(text)
            else:
                self.last_error = None
            return payload
        except Exception as exc:
            self.last_error = repr(exc)
            return None

    def _load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            self.last_error = "model path missing"
            return
        if self.model_path.is_dir():
            self._load_transformers()
            return
        if self.model_path.suffix.lower() == ".gguf":
            self._load_llama_cpp()
            return
        self.last_error = f"unsupported model path: {self.model_path}"

    def _load_llama_cpp(self) -> None:
        try:
            module = importlib.import_module("llama_cpp")
        except Exception as exc:
            self.last_error = repr(exc)
            return
        if not self._llama_cpp_cuda_available(module):
            self.last_error = "CUDA device required for llama.cpp backend"
            return
        llama_cls = getattr(module, "Llama", None)
        if llama_cls is None:
            self.last_error = "llama_cpp.Llama not found"
            return
        try:
            self._llm = llama_cls(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=-1,
                verbose=False,
            )
        except Exception as exc:
            self.last_error = repr(exc)
            self._llm = None
            return
        self._backend_kind = "llama_cpp"
        self.available = True

    def _load_transformers(self) -> None:
        try:
            transformers = importlib.import_module("transformers")
        except Exception as exc:
            self.last_error = repr(exc)
            return
        try:
            torch = importlib.import_module("torch")
        except Exception as exc:
            self.last_error = repr(exc)
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            self.last_error = "CUDA device required for transformers backend"
            return
        tokenizer_cls = getattr(transformers, "AutoTokenizer", None)
        model_cls = getattr(transformers, "AutoModelForCausalLM", None)
        if tokenizer_cls is None or model_cls is None:
            self.last_error = "transformers auto classes not found"
            return
        model_dtype = self._select_transformers_dtype(torch)
        try:
            self._tokenizer = tokenizer_cls.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._llm = model_cls.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                low_cpu_mem_usage=True,
                torch_dtype=model_dtype,
            )
            if hasattr(self._llm, "to"):
                self._llm = self._llm.to(self.device)
            if hasattr(self._llm, "eval"):
                self._llm.eval()
        except Exception as exc:
            self.last_error = repr(exc)
            self._llm = None
            self._tokenizer = None
            return
        self._torch = torch
        self._transformers_dtype = model_dtype
        self._backend_kind = "transformers"
        self.available = True

    def _llama_cpp_cuda_available(self, module: Any) -> bool:
        explicit_support = self._probe_llama_cpp_cuda_support(module)
        if explicit_support is not None:
            return explicit_support
        try:
            torch = importlib.import_module("torch")
        except Exception:
            torch = None
        if torch is not None:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and hasattr(cuda, "is_available"):
                try:
                    if bool(cuda.is_available()):
                        return True
                except Exception:
                    pass
        return False

    def _probe_llama_cpp_cuda_support(self, module: Any) -> bool | None:
        for attr_name in (
            "supports_cuda",
            "gpu_offload_supported",
            "GGML_USE_CUDA",
            "LLAMA_CPP_USE_CUDA",
        ):
            attr_value = getattr(module, attr_name, None)
            if isinstance(attr_value, bool):
                return attr_value
            if callable(attr_value):
                try:
                    return bool(attr_value())
                except Exception:
                    continue
        nested = getattr(module, "llama_cpp", None)
        if nested is not None:
            for attr_name in ("GGML_USE_CUDA", "LLAMA_CPP_USE_CUDA"):
                attr_value = getattr(nested, attr_name, None)
                if isinstance(attr_value, bool):
                    return attr_value
        return None

    def _generate_with_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any] | None:
        if self._llm is None or self._tokenizer is None:
            return None
        tokenizer = self._tokenizer
        model = self._llm
        inputs = self._prepare_transformers_inputs(prompt)
        if self._torch is None:
            self.last_error = "torch runtime missing"
            return None
        input_ids = inputs.get("input_ids")
        max_prompt_tokens = max(1, self.n_ctx - max_tokens)
        if hasattr(input_ids, "shape") and input_ids.shape[-1] > max_prompt_tokens:
            inputs["input_ids"] = input_ids[:, -max_prompt_tokens:]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask[:, -max_prompt_tokens:]
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": pad_token_id,
            "use_cache": False,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
        try:
            with self._torch.inference_mode():
                output = model.generate(**inputs, **generate_kwargs)
            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = output[0][input_length:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            payload = self._coerce_json(text)
            if payload is None:
                payload = self._coerce_plain_action_response(text)
            if payload is None:
                self.last_error = self._invalid_json_error(text)
            else:
                self.last_error = None
            return payload
        except Exception:
            self._clear_cuda_cache()
            raise

    def _prepare_transformers_inputs(self, prompt: str) -> dict[str, Any]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return {}
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            rendered_prompt = apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return tokenizer(rendered_prompt, return_tensors="pt")
        return tokenizer(prompt, return_tensors="pt")

    def _select_transformers_dtype(self, torch: Any) -> Any:
        cuda = getattr(torch, "cuda", None)
        bf16_supported = False
        if cuda is not None and hasattr(cuda, "is_bf16_supported"):
            try:
                bf16_supported = bool(cuda.is_bf16_supported())
            except Exception:
                bf16_supported = False
        if bf16_supported and hasattr(torch, "bfloat16"):
            return torch.bfloat16
        return getattr(torch, "float16", None)

    def _clear_cuda_cache(self) -> None:
        if self._torch is None:
            return
        cuda = getattr(self._torch, "cuda", None)
        if cuda is None or not hasattr(cuda, "empty_cache"):
            return
        try:
            cuda.empty_cache()
        except Exception:
            return

    def _coerce_json(self, text: str) -> dict[str, Any] | None:
        return coerce_json_dict(text)

    def _repair_json_fragment(self, text: str) -> str | None:
        from .json_utils import repair_json_fragment

        return repair_json_fragment(text)

    def _invalid_json_error(self, text: str) -> str:
        snippet = " ".join(str(text or "").split())
        if len(snippet) > 160:
            snippet = f"{snippet[:157]}..."
        if not snippet:
            return "response did not contain a valid JSON object"
        return f"response did not contain a valid JSON object: {snippet}"

    def _coerce_plain_action_response(self, text: str) -> dict[str, Any] | None:
        normalized = str(text or "").strip().upper()
        if not normalized:
            return None
        match = re.search(r"ACTION(?:ACTION)?([1-9][0-9]*)\b", normalized)
        if match is None:
            return None
        action_name = f"ACTION{match.group(1)}"
        return {
            "subgoal": "replan",
            "confidence": 0.35,
            "reason": "salvaged plain action response",
            "action_candidates": [
                {
                    "action": action_name,
                    "confidence": 0.35,
                    "reason": "salvaged plain action response",
                }
            ],
        }
