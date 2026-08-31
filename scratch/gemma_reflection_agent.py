%%writefile /kaggle/working/my_agent.py
# =====================================================================
# vLLM-driven ARC-AGI-3 submission agent
# The policy is served locally through vLLM's OpenAI-compatible API.
# =====================================================================
import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import subprocess
import textwrap
import threading
import time
import traceback
from typing import Any

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from agents.agent import Agent

logger = logging.getLogger(__name__)

# All MyAgent instances share one submission budget.  Swarm creates one agent
# per game concurrently, so an instance-local timer would allow the aggregate
# run to exceed Kaggle's wall-clock limit.
_SUBMISSION_STARTED_AT = time.monotonic()


class MyAgent(Agent):
    """vLLM-powered ARC agent that emits one JSON action per step."""

    MODEL_TO_GAME_ACTION = {
        "up": "ACTION1",
        "down": "ACTION2",
        "left": "ACTION3",
        "right": "ACTION4",
        "spacebar": "ACTION5",
        "click": "ACTION6",
        "undo": "ACTION7",
        "reset": "RESET",
    }
    GAME_TO_MODEL_ACTION = {
        game_name: model_name for model_name, game_name in MODEL_TO_GAME_ACTION.items()
    }
    MAX_ACTIONS = 200
    # Submission safety limits, matching the official GPT-OSS template style.
    # Swarm runs one thread per game; each thread must finish so the scorecard can close.
    GAME_TIME_LIMIT_S = 8 * 60 * 60
    FIRST_ACTION_DEADLINE_S = 14 * 60
    LLM_REQUEST_TIMEOUT_S = 400
    GLOBAL_TIME_LIMIT_SECONDS = 9 * 60 * 60
    GLOBAL_SHUTDOWN_RESERVE_SECONDS = 20 * 60
    MODEL_PATH = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"
    MAX_HISTORY = 12
    MAX_FRAME_MEMORY = 11
    ACTION_CONTEXT_FRAMES = 4
    REFLECTION_INTERVAL = 10
    MAX_REFLECTION_CHARS = 1800
    MAX_PLAN_ACTIONS = 4
    FRAME_BORDER_IGNORE = 3
    MAX_NEW_TOKENS = 1024
    REPAIR_MAX_NEW_TOKENS = 256
    REFLECTION_MAX_NEW_TOKENS = 10000
    FRAME_IMAGE_SCALE = 8
    DEFAULT_TRACE_PATH = "/kaggle/working/llm_inference_trace.jsonl"
    VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
    VLLM_SERVED_MODEL_NAME = "vllm-model"
    VLLM_LOG_PATH = "/kaggle/working/vllm_server.log"
    ARC_PALETTE = [
        (0, 0, 0),
        (0, 116, 217),
        (255, 65, 54),
        (46, 204, 64),
        (255, 220, 0),
        (170, 170, 170),
        (240, 18, 190),
        (255, 133, 27),
        (127, 219, 255),
        (135, 12, 37),
        (57, 204, 204),
        (177, 13, 201),
        (1, 255, 112),
        (133, 20, 75),
        (61, 153, 112),
        (221, 221, 221),
    ]
    _client: OpenAI | None = None
    _served_model: str | None = None
    _server_process: subprocess.Popen[bytes] | None = None
    _server_log: Any = None
    _server_lock = threading.Lock()
    _vllm_startup_error: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        self.history: list[dict[str, Any]] = []
        self.frame_memory: list[dict[str, Any]] = []
        self.pending_actions: list[dict[str, Any]] = []
        self.last_plan_summary = ""
        self.reflection_buffer: list[dict[str, Any]] = []
        self.reflection_memory_path = self._reflection_memory_path()
        self.reflection_memory = self._load_reflection_memory()
        self.reflections_completed = 0
        self.current_level_number = 1
        self.failed_state_actions: dict[str, set[str]] = {}
        self._game_started_monotonic = time.monotonic()
        self._deadline_hit = False

    def _reflection_memory_path(self) -> str:
        default_dir = (
            "/kaggle/working/agent_memory"
            if os.path.isdir("/kaggle/working")
            else os.path.join(os.getcwd(), "agent_memory")
        )
        base_dir = os.getenv("LLM_MEMORY_DIR", default_dir)
        safe_game_id = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in self.game_id
        )
        return os.path.join(base_dir, f"{safe_game_id}.md")

    def _load_reflection_memory(self) -> str:
        try:
            with open(self.reflection_memory_path, "r", encoding="utf-8") as memory_file:
                memory = memory_file.read().strip()
            if memory:
                return memory[: self.MAX_REFLECTION_CHARS]
        except OSError:
            pass
        return "# Agent Memory\n\nNo reflection has been completed yet."

    @classmethod
    def _global_deadline(cls) -> float:
        try:
            limit = float(
                os.getenv(
                    "AGENT_GLOBAL_TIME_LIMIT_SECONDS",
                    str(cls.GLOBAL_TIME_LIMIT_SECONDS),
                )
            )
        except ValueError:
            limit = float(cls.GLOBAL_TIME_LIMIT_SECONDS)
        try:
            reserve = float(
                os.getenv(
                    "AGENT_GLOBAL_SHUTDOWN_RESERVE_SECONDS",
                    str(cls.GLOBAL_SHUTDOWN_RESERVE_SECONDS),
                )
            )
        except ValueError:
            reserve = float(cls.GLOBAL_SHUTDOWN_RESERVE_SECONDS)
        return _SUBMISSION_STARTED_AT + max(0.0, limit - max(0.0, reserve))

    @classmethod
    def _remaining_global_seconds(cls) -> float:
        return max(0.0, cls._global_deadline() - time.monotonic())

    @classmethod
    def _load_vllm_once(cls) -> None:
        if cls._client is not None and cls._served_model is not None:
            return

        with cls._server_lock:
            if cls._client is not None and cls._served_model is not None:
                return

            port = os.getenv("VLLM_PORT", "8000")
            default_base_url = f"http://127.0.0.1:{port}/v1"
            base_url = os.getenv("VLLM_BASE_URL", default_base_url).rstrip("/")
            remaining = cls._remaining_global_seconds()
            if remaining <= 0:
                raise TimeoutError("Global submission time budget exhausted before vLLM startup")
            request_timeout = min(
                float(os.getenv("VLLM_REQUEST_TIMEOUT", "1200")), remaining
            )
            client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("VLLM_API_KEY", "local-server-key"),
                timeout=max(1.0, request_timeout),
                max_retries=0,
            )

            try:
                models = client.models.list()
            except Exception:
                if os.getenv("VLLM_START_SERVER", "1").lower() in {"0", "false", "no"}:
                    raise RuntimeError(f"No vLLM server is reachable at {base_url}")
                cls._start_vllm_server()
                models = cls._wait_for_vllm(client)

            if not models.data:
                raise RuntimeError("vLLM reported no served models")
            requested_model = os.getenv(
                "VLLM_SERVED_MODEL_NAME", cls.VLLM_SERVED_MODEL_NAME
            )
            model_ids = {item.id for item in models.data}
            cls._served_model = (
                requested_model if requested_model in model_ids else models.data[0].id
            )
            cls._client = client
            logger.info("vLLM ready at %s with model %s", base_url, cls._served_model)

    @classmethod
    def _ensure_vllm_available(cls) -> None:
        if cls._vllm_startup_error is not None:
            raise RuntimeError(
                f"vLLM disabled after startup failure: {cls._vllm_startup_error}"
            )
        try:
            cls._load_vllm_once()
        except Exception as exc:
            cls._vllm_startup_error = f"{type(exc).__name__}: {exc}"
            raise

    @classmethod
    def _start_vllm_server(cls) -> None:
        if cls._server_process is not None and cls._server_process.poll() is None:
            return

        model_path = os.getenv("VLLM_MODEL_PATH", cls.MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"vLLM model path not found: {model_path}. Attach the Kaggle model asset "
                "or set VLLM_MODEL_PATH."
            )

        served_name = os.getenv("VLLM_SERVED_MODEL_NAME", cls.VLLM_SERVED_MODEL_NAME)
        port = os.getenv("VLLM_PORT", "8000")
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--served-model-name",
            served_name,
            "--tensor-parallel-size",
            os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"),
            "--max-num-seqs",
            os.getenv("VLLM_MAX_NUM_SEQS", "20"),
            "--gpu-memory-utilization",
            os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.94"),
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--dtype",
            os.getenv("VLLM_DTYPE", "auto"),
            "--max-model-len",
            os.getenv("VLLM_MAX_MODEL_LEN", "32768"),
            "--enable-prefix-caching",
            "--trust-remote-code",
        ]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        log_path = os.getenv("VLLM_LOG_PATH", cls.VLLM_LOG_PATH)
        cls._server_log = open(log_path, "wb", buffering=0)
        logger.info("Starting vLLM server for %s; log: %s", model_path, log_path)
        cls._server_process = subprocess.Popen(
            command,
            stdout=cls._server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @classmethod
    def _wait_for_vllm(cls, client: OpenAI) -> Any:
        timeout = min(
            float(os.getenv("VLLM_STARTUP_TIMEOUT", "1000")),
            cls._remaining_global_seconds(),
        )
        if timeout <= 0:
            raise TimeoutError("Global submission time budget exhausted during vLLM startup")
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if cls._server_process is not None and cls._server_process.poll() is not None:
                log_path = os.getenv("VLLM_LOG_PATH", cls.VLLM_LOG_PATH)
                log_tail = cls._read_log_tail(log_path)
                raise RuntimeError(
                    f"vLLM server exited with code {cls._server_process.returncode}; "
                    f"see {log_path}\nLast server log lines:\n{log_tail}"
                )
            try:
                return client.models.list()
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        log_path = os.getenv("VLLM_LOG_PATH", cls.VLLM_LOG_PATH)
        log_tail = cls._read_log_tail(log_path)
        raise RuntimeError(
            f"vLLM server did not become ready within {timeout:.0f}s: {last_error}\n"
            f"Last server log lines:\n{log_tail}"
        )

    @staticmethod
    def _read_log_tail(path: str, max_bytes: int = 12000) -> str:
        try:
            with open(path, "rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - max_bytes))
                return log_file.read().decode("utf-8", errors="replace").strip()
        except OSError as exc:
            return f"Unable to read vLLM log: {exc}"

    @property
    def game_elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self._game_started_monotonic)

    @property
    def game_time_remaining_s(self) -> float:
        limit = float(os.getenv("GAME_TIME_LIMIT_S", str(self.GAME_TIME_LIMIT_S)))
        if limit <= 0:
            return self._remaining_global_seconds()
        return max(0.0, min(limit - self.game_elapsed_s, self._remaining_global_seconds()))

    def _mark_deadline_hit(self, reason: str) -> None:
        if not self._deadline_hit:
            logger.info(
                "%s for %s after %s actions and %.2fs",
                reason,
                self.game_id,
                self.action_counter,
                self.game_elapsed_s,
            )
        self._deadline_hit = True

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if self._remaining_global_seconds() <= 0:
            self._mark_deadline_hit("Global submission time budget exhausted")
            return True
        if self._deadline_hit:
            return True
        # Always allow the first RESET action; the gateway needs early activity.
        if self.action_counter == 0:
            return False
        if self.game_time_remaining_s <= 0:
            self._mark_deadline_hit("Per-game time limit reached")
            return True
        return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        prompt = ""
        response_text = ""
        timings: dict[str, float] = {}
        turn_start = time.perf_counter()
        try:
            if self._remaining_global_seconds() <= 0:
                action = GameAction.RESET
                action.reasoning = "Global submission time budget exhausted."
                return action
            if self.action_counter == 0:
                startup_elapsed_s = time.monotonic() - _SUBMISSION_STARTED_AT
                first_action_deadline_s = float(
                    os.getenv("FIRST_ACTION_DEADLINE_S", str(self.FIRST_ACTION_DEADLINE_S))
                )
                if startup_elapsed_s > first_action_deadline_s:
                    logger.warning(
                        "First action for %s selected after %.2fs, past %.2fs target",
                        self.game_id,
                        startup_elapsed_s,
                        first_action_deadline_s,
                    )
                action = GameAction.RESET
                action.reasoning = "Initial RESET before slow model startup."
                return action
            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.pending_actions = []
                action = GameAction.RESET
                action.reasoning = "Environment requires RESET before play."
                return action
            if self.game_time_remaining_s <= 0:
                self._mark_deadline_hit("Per-game time limit reached before LLM action")
                return self._fallback_action(latest_frame, "Per-game time limit reached.")

            # The gateway receives RESET before potentially slow model startup.
            self._ensure_vllm_available()

            stage_start = time.perf_counter()
            self._observe_frame(latest_frame)
            timings["observe_frame"] = time.perf_counter() - stage_start

            if len(self.reflection_buffer) >= self.REFLECTION_INTERVAL:
                stage_start = time.perf_counter()
                self._run_reflection(latest_frame)
                timings["reflection"] = time.perf_counter() - stage_start

            if self.pending_actions:
                action = self._dequeue_action(latest_frame)
                timings["total_choose_action"] = time.perf_counter() - turn_start
                logger.info(
                    "Agent timing step=%s dequeued_action=%s remaining=%s total_choose_action=%.3fs",
                    self.action_counter,
                    action.name,
                    len(self.pending_actions),
                    timings["total_choose_action"],
                )
                return action

            stage_start = time.perf_counter()
            prompt = self._build_prompt(frames, latest_frame)
            timings["build_prompt"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            frame_images = self._build_context_images(latest_frame.frame)
            timings["build_context_images"] = time.perf_counter() - stage_start
            try:
                stage_start = time.perf_counter()
                response_text = self._generate_response(
                    prompt,
                    frame_images,
                    enable_thinking=self._action_thinking_enabled(),
                )
                try:
                    parsed = self._extract_action_json(response_text)
                except Exception:
                    repair_response = self._generate_response(
                        self._build_json_repair_prompt(prompt, response_text),
                        frame_images,
                        enable_thinking=False,
                        max_new_tokens=self.REPAIR_MAX_NEW_TOKENS,
                    )
                    response_text += "\n\nJSON_REPAIR_OUTPUT:\n" + repair_response
                    parsed = self._extract_action_json(repair_response)
                timings["generate_response"] = time.perf_counter() - stage_start
            except Exception as exc:
                self._write_llm_trace(latest_frame, prompt, response_text, context_images=frame_images, error=repr(exc))
                raise

            if parsed is None:
                raise ValueError("Model loop produced no JSON payload")
            if not ("actions" in parsed or "action" in parsed):
                exc = ValueError(f"Model did not finish with an action payload: {parsed}")
                repair_prompt = self._build_json_repair_prompt(prompt, response_text)
                try:
                    stage_start = time.perf_counter()
                    repair_text = self._generate_response(
                        repair_prompt,
                        frame_images,
                        enable_thinking=False,
                        max_new_tokens=self.REPAIR_MAX_NEW_TOKENS,
                    )
                    timings["repair_generate_response"] = time.perf_counter() - stage_start
                    stage_start = time.perf_counter()
                    parsed = self._extract_action_json(repair_text)
                    timings["repair_extract_json"] = time.perf_counter() - stage_start
                    response_text = response_text + "\n\nJSON_REPAIR_OUTPUT:\n" + repair_text
                except Exception:
                    self._write_llm_trace(
                        latest_frame,
                        prompt,
                        response_text,
                        context_images=frame_images,
                        error=repr(exc),
                    )
                    raise exc

            stage_start = time.perf_counter()
            planned_actions = self._normalize_action_specs(parsed, latest_frame)
            if not planned_actions:
                logger.warning("Model returned no usable actions, using ordered fallback: %s", parsed)
                action = self._fallback_action(latest_frame, "Model returned no usable actions.")
                self._write_llm_trace(
                    latest_frame,
                    prompt,
                    response_text,
                    parsed=parsed,
                    chosen_action=action,
                    context_images=frame_images,
                    error="empty_or_unusable_action_plan",
                )
                self._remember_step(latest_frame, action, response_text, parsed)
                timings["plan_to_action"] = time.perf_counter() - stage_start
                timings["total_choose_action"] = time.perf_counter() - turn_start
                self._log_timing(latest_frame, frame_images, timings)
                return action
            self.pending_actions = planned_actions
            self.last_plan_summary = str(parsed.get("plan_summary", "")).strip()
            action = self._dequeue_action(
                latest_frame,
                {
                    "raw_plan": parsed,
                    "plan_length": len(planned_actions),
                    "plan_summary": self.last_plan_summary,
                },
                remember=False,
            )
            timings["plan_to_action"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            self._write_llm_trace(
                latest_frame,
                prompt,
                response_text,
                parsed=parsed,
                chosen_action=action,
                context_images=frame_images,
            )
            timings["write_trace"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            self._remember_step(latest_frame, action, response_text, parsed)
            timings["remember_step"] = time.perf_counter() - stage_start
            timings["total_choose_action"] = time.perf_counter() - turn_start
            self._log_timing(latest_frame, frame_images, timings)
            return action
        except Exception as exc:
            logger.warning("vLLM action generation failed: %s", exc)
            traceback.print_exc()
            action = self._fallback_action(latest_frame, f"vLLM failure: {exc}")
            if not self.history or self.history[-1].get("step") != self.action_counter:
                self._remember_step(
                    latest_frame,
                    action,
                    response_text or "FALLBACK_AFTER_VLLM_FAILURE",
                    {
                        "reasoning": "Fallback after model or JSON failure.",
                        "plan_summary": f"Fallback action after error: {exc}",
                    },
                )
            return action

    def _build_prompt(self, frames: list[FrameData], latest_frame: FrameData) -> str:
        available_actions = self._available_model_action_names(latest_frame)
        recent_history = json.dumps(self._prompt_history()[-4:], ensure_ascii=True)
        thinking_directive = "/think" if self._action_thinking_enabled() else "/no_think"
        example_action: dict[str, Any] = {"name": available_actions[0]}
        if "click" in available_actions:
            example_action = {"name": "click", "x": 12, "y": 34}
        output_example = json.dumps(
            {
                "board_change_assessment": "central-board evidence from the latest transition",
                "plan_summary": "test one rule or pursue the current subgoal",
                "actions": [example_action],
            },
            ensure_ascii=True,
        )
        ineffective_actions = self._ineffective_actions_for_current_state(latest_frame)
        legal_action_instructions = self._legal_action_instructions(available_actions)

        return textwrap.dedent(
            f"""
            You are the action agent for an interactive ARC-AGI-3 visual game.
            The images are chronological; the last is current. Red STEP labels are added
            chronology, not game UI. Ignore the outer {self._border_ignore_pixels()} pixels
            when judging progress. Trust numeric transitions over visual guesses.

            Infer the controllable object, causal action effects, and current objective.
            Prefer purposeful new states. A repeated state is not progress. Do not invent
            counters, bars, or goals without evidence.

            Legal actions for this exact state: {available_actions}
            Action format rules for this state only:
            {legal_action_instructions}
            Do not output any action name outside Legal actions for this exact state.
            Ineffective in this exact state: {ineffective_actions}

            Reflection memory (authoritative but revisable):
            {self.reflection_memory}

            Recent transitions:
            {recent_history}

            Return exactly one JSON object, no tools or markdown. Include 1 to
            {self.MAX_PLAN_ACTIONS} actions; use one exploratory action if uncertain.
            Example: {output_example}
            {thinking_directive}
            """
        ).strip()

    def _generate_response(
        self,
        prompt: str,
        frame_images: list[Image.Image],
        enable_thinking: bool,
        max_new_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str:
        if self._client is None or self._served_model is None:
            raise RuntimeError("vLLM client is not initialized")
        response_start = time.perf_counter()
        token_budget = max_new_tokens or int(
            os.getenv("LLM_MAX_NEW_TOKENS", str(self.MAX_NEW_TOKENS))
        )
        content: list[dict[str, Any]] = []
        for frame_image in frame_images:
            image_buffer = io.BytesIO()
            frame_image.save(image_buffer, format="PNG")
            encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
            image_url = f"data:image/png;base64,{encoded_image}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        request_kwargs: dict[str, Any] = {
            "model": self._served_model,
            "messages": messages,
            "max_tokens": token_budget,
            "temperature": float(
                os.getenv("LLM_TEMPERATURE", "0.6" if enable_thinking else "0.2")
            ),
            "top_p": float(os.getenv("LLM_TOP_P", "0.95")),
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "top_k": int(os.getenv("LLM_TOP_K", "20")),
                "repetition_penalty": float(
                    os.getenv("LLM_REPETITION_PENALTY", "1.08")
                ),
            },
        }
        if json_mode and os.getenv("VLLM_JSON_MODE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            request_kwargs["response_format"] = {"type": "json_object"}
        remaining = self.game_time_remaining_s
        if remaining <= 0:
            raise TimeoutError("Per-game or global time budget exhausted before inference")
        configured_timeout = float(
            os.getenv("VLLM_REQUEST_TIMEOUT", str(self.LLM_REQUEST_TIMEOUT_S))
        )
        request_client = self._client.with_options(
            timeout=max(1.0, min(configured_timeout, remaining))
        )
        response = request_client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        if not isinstance(content, str):
            content = "".join(str(part) for part in content)
        completion_tokens = (
            response.usage.completion_tokens if response.usage is not None else None
        )
        if choice.finish_reason == "length":
            logger.warning(
                "vLLM output reached token budget=%s without a stop token",
                token_budget,
            )
        if self._timing_enabled():
            logger.info(
                "vLLM timing step=%s images=%s thinking=%s finish=%s "
                "completion_tokens=%s budget=%s total=%.3fs",
                self.action_counter,
                [f"{image.width}x{image.height}" for image in frame_images],
                enable_thinking,
                choice.finish_reason,
                completion_tokens,
                token_budget,
                time.perf_counter() - response_start,
            )
        return content.strip()

    def _build_json_repair_prompt(self, original_prompt: str, bad_output: str) -> str:
        return textwrap.dedent(
            f"""
            The previous answer did not contain a valid JSON object.

            Original task:
            {original_prompt}

            Previous non-JSON answer:
            {bad_output[:3000]}

            Return exactly one JSON object now. Do not include thought, markdown, prose, or code fences.
            Required final shape:
            {{"actions": [{{"name": "up"}}, {{"name": "click", "x": 12, "y": 34}}]}}
            /no_think
            """
        ).strip()

    def _extract_action_json(self, text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        errors: list[str] = []
        command_payloads: list[tuple[int, int, dict[str, Any]]] = []
        other_payloads: list[tuple[int, int, dict[str, Any]]] = []
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, length = decoder.raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if isinstance(payload, dict):
                candidate = (start, start + length, payload)
                if "actions" in payload or "action" in payload:
                    command_payloads.append(candidate)
                else:
                    other_payloads.append(candidate)
        if command_payloads:
            return max(command_payloads, key=lambda item: (item[1], -item[0]))[2]
        if other_payloads:
            return max(other_payloads, key=lambda item: (item[1], -item[0]))[2]
        raise ValueError(f"No JSON object found in model output: {text!r}; parse_errors={errors[:3]}")

    def _normalize_action_specs(
        self,
        payload: dict[str, Any],
        latest_frame: FrameData,
    ) -> list[dict[str, Any]]:
        raw_actions = payload.get("actions")
        if raw_actions is None and payload.get("action"):
            raw_actions = [payload]
        elif raw_actions is not None and not isinstance(raw_actions, list):
            raw_actions = [raw_actions]
        if not isinstance(raw_actions, list):
            return []

        normalized: list[dict[str, Any]] = []
        ineffective_actions = set(self._ineffective_actions_for_current_state(latest_frame))
        for item in raw_actions[: self.MAX_PLAN_ACTIONS]:
            action_payload = item if isinstance(item, dict) else {}
            raw_name = self._coerce_action_name(
                action_payload.get("name") or action_payload.get("action") or item
            )
            if raw_name == "RESET":
                action = GameAction.RESET
            else:
                try:
                    action = GameAction.from_name(raw_name)
                except ValueError:
                    logger.info("Skipping unknown planned action %s", raw_name)
                    continue
            if action is not GameAction.RESET and not self._is_action_available(latest_frame, action):
                logger.info("Skipping unavailable planned action %s", raw_name)
                continue
            spec: dict[str, Any] = {"name": action.name}
            if action.is_complex():
                spec["x"] = self._clamp_coordinate(action_payload.get("x", 0))
                spec["y"] = self._clamp_coordinate(action_payload.get("y", 0))
            ineffective_key = self._action_failure_key_from_spec(spec)
            if ineffective_key in ineffective_actions:
                logger.info("Skipping action proven ineffective in current state: %s", ineffective_key)
                continue
            normalized.append(spec)
        return normalized

    def _coerce_action_name(self, raw_name: Any) -> str:
        raw_text = str(raw_name or "").strip()
        semantic_name = raw_text.lower().replace("-", "_").replace(" ", "_")
        semantic_aliases = {
            "move_up": "up",
            "move_down": "down",
            "move_left": "left",
            "move_right": "right",
        }
        semantic_name = semantic_aliases.get(semantic_name, semantic_name)
        if semantic_name in self.MODEL_TO_GAME_ACTION:
            return self.MODEL_TO_GAME_ACTION[semantic_name]

        text = raw_text.upper()
        if not text:
            return ""
        if text.isdigit():
            try:
                return GameAction.from_id(int(text)).name
            except ValueError:
                return ""
        digit_match = re.fullmatch(r"ACTION[_\s-]*(\d+)", text)
        if digit_match:
            return f"ACTION{digit_match.group(1)}"
        return text

    def _dequeue_action(
        self,
        latest_frame: FrameData,
        extra_reasoning: dict[str, Any] | None = None,
        remember: bool = True,
    ) -> GameAction:
        spec = self.pending_actions.pop(0)
        action = self._materialize_action(spec, latest_frame)
        reasoning: dict[str, Any] = {
            "driver": "vllm-openai-compatible",
            "model": self._served_model,
            "thinking_enabled": self._action_thinking_enabled(),
            "from_plan_queue": True,
            "remaining_planned_actions": len(self.pending_actions),
            "plan_summary": self.last_plan_summary,
            "raw_plan_action": spec,
            "available_actions": list(latest_frame.available_actions or []),
        }
        if extra_reasoning:
            reasoning.update(extra_reasoning)
        action.reasoning = reasoning
        if remember:
            self._remember_step(latest_frame, action, "DEQUEUED_FROM_PLAN", self._action_to_payload(action))
        logger.info(
            "Dequeued planned action %s for %s (%s remaining)",
            action.name,
            self.game_id,
            len(self.pending_actions),
        )
        return action

    def _materialize_action(self, spec: dict[str, Any], latest_frame: FrameData) -> GameAction:
        raw_name = str(spec.get("name", "")).upper().strip()
        action = GameAction.RESET if raw_name == "RESET" else GameAction.from_name(raw_name)
        if action is not GameAction.RESET and not self._is_action_available(latest_frame, action):
            self.pending_actions = []
            return self._fallback_action(latest_frame, f"Planned action {action.name} no longer available.")
        if self._model_action_name(action) in self._ineffective_actions_for_current_state(latest_frame):
            self.pending_actions = []
            return self._fallback_action(
                latest_frame, f"Planned action {action.name} already failed in this state."
            )
        if action.is_complex():
            action.set_data(
                {
                    "x": self._clamp_coordinate(spec.get("x", 0)),
                    "y": self._clamp_coordinate(spec.get("y", 0)),
                }
            )
        return action

    def _action_to_payload(self, action: GameAction) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": action.name}
        payload.update(self._action_data_dict(action))
        return payload

    def _action_data_dict(self, action: GameAction | None) -> dict[str, Any]:
        if action is None:
            return {}
        action_data = getattr(action, "action_data", None)
        if hasattr(action_data, "model_dump"):
            raw_data = action_data.model_dump()
        elif isinstance(action_data, dict):
            raw_data = action_data
        else:
            raw_data = {}
        return {
            key: self._clamp_coordinate(raw_data[key])
            for key in ("x", "y")
            if key in raw_data
        }

    def _action_failure_key_from_spec(self, spec: dict[str, Any]) -> str:
        action_name = str(spec.get("name", "")).upper().strip()
        if action_name == "ACTION6":
            x = self._clamp_coordinate(spec.get("x", 0))
            y = self._clamp_coordinate(spec.get("y", 0))
            return f"click@{x},{y}"
        try:
            return self._model_action_name(GameAction.from_name(action_name))
        except ValueError:
            return action_name.lower()

    def _action_failure_key(self, action: GameAction) -> str:
        if action.is_complex():
            data = self._action_data_dict(action)
            return f"click@{data.get('x', 0)},{data.get('y', 0)}"
        return self._model_action_name(action)

    def _ineffective_actions_for_current_state(
        self, latest_frame: FrameData
    ) -> list[str]:
        failed = getattr(self, "failed_state_actions", {})
        return sorted(failed.get(self._frame_hash(latest_frame.frame), set()))

    def _build_reflection_prompt(self, latest_frame: FrameData) -> str:
        transitions = json.dumps(
            self.reflection_buffer[-self.REFLECTION_INTERVAL :], ensure_ascii=True
        )
        return textwrap.dedent(
            f"""
            You are the reflection agent for an ARC-AGI-3 game. Review the previous
            memory, the last {self.REFLECTION_INTERVAL} completed transitions, and the
            chronological images. The final image is current; red STEP labels are added
            chronology. Pixel changes may be movement, transformation, collection,
            animation, or UI, so do not assume translation.

            Keep only evidence-supported, useful conclusions. Correct stale beliefs.
            Distinguish confirmed rules from hypotheses and state a concrete next goal.
            Return only a compact Markdown document under {self.MAX_REFLECTION_CHARS}
            characters with exactly these headings:

            # Agent Memory
            ## Rules
            ## Goal
            ## Progress
            ## Avoid

            Previous memory:
            {self.reflection_memory}

            Current level: {int(latest_frame.levels_completed) + 1}
            Completed transitions:
            {transitions}
            /no_think
            """
        ).strip()

    def _clean_reflection_markdown(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        if not cleaned:
            return self.reflection_memory
        if not cleaned.startswith("# Agent Memory"):
            cleaned = "# Agent Memory\n\n" + cleaned
        return cleaned[: self.MAX_REFLECTION_CHARS].rstrip()

    def _save_reflection_memory(self) -> None:
        try:
            memory_dir = os.path.dirname(self.reflection_memory_path)
            if memory_dir:
                os.makedirs(memory_dir, exist_ok=True)
            temp_path = self.reflection_memory_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as memory_file:
                memory_file.write(self.reflection_memory + "\n")
            os.replace(temp_path, self.reflection_memory_path)
        except OSError as exc:
            logger.warning("Failed to save reflection memory: %s", exc)

    def _run_reflection(self, latest_frame: FrameData) -> None:
        if len(self.reflection_buffer) < self.REFLECTION_INTERVAL:
            return
        # A reflection may revise the goal, so discard any stale queued plan.
        self.pending_actions = []
        prompt = self._build_reflection_prompt(latest_frame)
        images = self._build_context_images(
            latest_frame.frame, limit=self.REFLECTION_INTERVAL + 1
        )
        try:
            response = self._generate_response(
                prompt,
                images,
                enable_thinking=False,
                max_new_tokens=self.REFLECTION_MAX_NEW_TOKENS,
                json_mode=False,
            )
            self.reflection_memory = self._clean_reflection_markdown(response)
            self._save_reflection_memory()
            self.reflections_completed += 1
            logger.info(
                "Reflection completed for %s after %s transitions; memory=%s",
                self.game_id,
                self.REFLECTION_INTERVAL,
                self.reflection_memory_path,
            )
        except Exception as exc:
            logger.warning("Reflection failed for %s: %s", self.game_id, exc)
        finally:
            del self.reflection_buffer[: self.REFLECTION_INTERVAL]

    def _remember_step(
        self,
        latest_frame: FrameData,
        action: GameAction,
        raw_text: str,
        parsed: dict[str, Any],
    ) -> None:
        item = {
            "step": self.action_counter,
            "state": latest_frame.state.name,
            "levels_completed": latest_frame.levels_completed,
            "available_actions": self._available_model_action_names(latest_frame),
            "chosen_action": self._model_action_name(action),
            "action_data": self._action_data_dict(action) or None,
            "failure_key": self._action_failure_key(action),
            "raw_model_output": raw_text[:400],
            "parsed_output": parsed,
            "reasoning": parsed.get("reasoning", ""),
            "plan_before_action": parsed.get("plan_summary", ""),
            "frame_signature": self._frame_signature(latest_frame.frame),
        }
        self.history.append(item)
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY :]

    def _write_llm_trace(
        self,
        latest_frame: FrameData,
        prompt: str,
        response_text: str,
        parsed: dict[str, Any] | None = None,
        chosen_action: GameAction | None = None,
        context_images: list[Image.Image] | None = None,
        error: str | None = None,
    ) -> None:
        trace_path = os.getenv("LLM_TRACE_PATH", self.DEFAULT_TRACE_PATH)
        action_data = (self._action_data_dict(chosen_action) or None) if chosen_action else None
        context_image_paths = self._save_trace_images(trace_path, context_images)
        record = {
            "timestamp": time.time(),
            "game_id": self.game_id,
            "step": self.action_counter,
            "state": latest_frame.state.name,
            "levels_completed": latest_frame.levels_completed,
            "available_actions": self._available_action_names(latest_frame),
            "frame_signature": self._frame_signature(latest_frame.frame),
            "reflection_memory_path": self.reflection_memory_path,
            "reflection_memory": self.reflection_memory,
            "reflections_completed": self.reflections_completed,
            "input": {
                "prompt": prompt,
                "images": {
                    "source": "separate chronological observation frames, oldest first",
                    "paths": context_image_paths,
                    "format": "ordered PIL RGB images passed directly to the multimodal processor",
                    "scale": self.FRAME_IMAGE_SCALE,
                },
            },
            "output": {
                "raw_text": response_text,
                "parsed_json": parsed,
                "plan_summary": parsed.get("plan_summary", "") if parsed else "",
                "planned_actions": parsed.get("actions") if parsed else None,
                "chosen_action": (
                    self._model_action_name(chosen_action) if chosen_action else None
                ),
                "game_action": chosen_action.name if chosen_action else None,
                "action_data": action_data,
                "pending_actions_after_choice": self.pending_actions,
            },
            "error": error,
        }
        try:
            trace_dir = os.path.dirname(trace_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write LLM trace JSON: %s", exc)

    def _timing_enabled(self) -> bool:
        value = os.getenv("LLM_TIMING", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _action_thinking_enabled(self) -> bool:
        value = os.getenv("LLM_ACTION_THINKING", "0").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _log_timing(
        self,
        latest_frame: FrameData,
        context_images: list[Image.Image],
        timings: dict[str, float],
    ) -> None:
        if not self._timing_enabled():
            return
        ordered = [
            "observe_frame",
            "build_prompt",
            "build_context_images",
            "generate_response",
            "extract_json",
            "repair_generate_response",
            "repair_extract_json",
            "plan_to_action",
            "write_trace",
            "remember_step",
            "total_choose_action",
        ]
        timing_text = " ".join(
            f"{name}={timings[name]:.3f}s" for name in ordered if name in timings
        )
        logger.info(
            "Agent timing step=%s state=%s levels=%s images=%s %s",
            self.action_counter,
            latest_frame.state.name,
            latest_frame.levels_completed,
            [f"{image.width}x{image.height}" for image in context_images],
            timing_text,
        )

    def _observe_frame(self, latest_frame: FrameData) -> None:
        level_number = int(latest_frame.levels_completed) + 1
        if level_number != self.current_level_number:
            self.current_level_number = level_number
            self.failed_state_actions = {}
            self.pending_actions = []
        current_hash = self._frame_hash(latest_frame.frame)
        current_entry = {
            "step": self.action_counter,
            "state": latest_frame.state.name,
            "levels_completed": latest_frame.levels_completed,
            "frame": latest_frame.frame,
            "frame_hash": current_hash,
            "frame_signature": self._frame_signature(latest_frame.frame),
        }

        if self.frame_memory and self.frame_memory[-1]["step"] == self.action_counter:
            self.frame_memory[-1] = current_entry
        else:
            self.frame_memory.append(current_entry)
            if len(self.frame_memory) > self.MAX_FRAME_MEMORY:
                self.frame_memory = self.frame_memory[-self.MAX_FRAME_MEMORY :]

        if not self.history:
            return
        previous_action = self.history[-1]
        if "after_frame_signature" in previous_action:
            return
        if previous_action["step"] >= self.action_counter:
            return

        before_entry = None
        for item in reversed(self.frame_memory[:-1]):
            if item["step"] == previous_action["step"]:
                before_entry = item
                break
        if before_entry is None and len(self.frame_memory) >= 2:
            before_entry = self.frame_memory[-2]
        if before_entry is None:
            return

        changed_pixels = self._changed_pixels(before_entry["frame"], latest_frame.frame)
        levels_delta = latest_frame.levels_completed - previous_action["levels_completed"]
        if changed_pixels == 0 and levels_delta == 0:
            failed_actions = getattr(self, "failed_state_actions", None)
            if failed_actions is None:
                self.failed_state_actions = {}
                failed_actions = self.failed_state_actions
            failed_actions.setdefault(before_entry["frame_hash"], set()).add(
                previous_action.get("failure_key") or previous_action["chosen_action"]
            )
        previous_action.update(
            {
                "after_step": self.action_counter,
                "after_state": latest_frame.state.name,
                "after_levels_completed": latest_frame.levels_completed,
                "after_frame_signature": self._frame_signature(latest_frame.frame),
                "after_frame_hash": current_hash,
                "changed_pixels": changed_pixels,
                "levels_delta": levels_delta,
                "state_changed": before_entry["frame_hash"] != current_hash,
                "repeated_state": any(
                    item["frame_hash"] == current_hash for item in self.frame_memory[:-1]
                ),
            }
        )
        self.reflection_buffer.append(self._compact_history_item(previous_action))

    def _compact_history_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": item.get("step"),
            "action": item.get("chosen_action"),
            "action_data": item.get("action_data"),
            "failure_key": item.get("failure_key"),
            "levels_before": item.get("levels_completed"),
            "levels_after": item.get("after_levels_completed"),
            "levels_delta": item.get("levels_delta"),
            "changed_pixels": item.get("changed_pixels"),
            "state_changed": item.get("state_changed"),
            "repeated_state": item.get("repeated_state"),
            "plan_before_action": item.get("plan_before_action", ""),
            "frame_before": item.get("frame_signature"),
            "frame_after": item.get("after_frame_signature"),
        }

    def _prompt_history(self) -> list[dict[str, Any]]:
        return [
            self._compact_history_item(item)
            for item in self.history[-self.MAX_HISTORY :]
        ]

    def _save_trace_images(
        self,
        trace_path: str,
        context_images: list[Image.Image] | None,
    ) -> list[str]:
        if not context_images:
            return []
        try:
            base_dir = os.getenv("LLM_TRACE_IMAGE_DIR")
            if not base_dir:
                trace_dir = os.path.dirname(trace_path) or "."
                base_dir = os.path.join(trace_dir, "llm_trace_images")
            os.makedirs(base_dir, exist_ok=True)
            safe_game_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.game_id)
            image_paths = []
            for index, context_image in enumerate(context_images):
                image_path = os.path.join(
                    base_dir,
                    f"{safe_game_id}_step_{self.action_counter:04d}_frame_{index:02d}.png",
                )
                context_image.save(image_path)
                image_paths.append(image_path)
            return image_paths
        except Exception as exc:
            logger.warning("Failed to save LLM trace images: %s", exc)
            return []

    def _fallback_action(self, latest_frame: FrameData, reason: str) -> GameAction:
        ineffective_actions = set(self._ineffective_actions_for_current_state(latest_frame))
        available = [
            action
            for action in [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.ACTION5,
                GameAction.ACTION6,
                GameAction.ACTION7,
            ]
            if self._is_action_available(latest_frame, action)
            and self._model_action_name(action) not in ineffective_actions
        ]
        if not available:
            available = [
                action
                for action in [
                    GameAction.ACTION1,
                    GameAction.ACTION2,
                    GameAction.ACTION3,
                    GameAction.ACTION4,
                    GameAction.ACTION5,
                    GameAction.ACTION6,
                    GameAction.ACTION7,
                ]
                if self._is_action_available(latest_frame, action)
            ]
        if not available:
            action = GameAction.ACTION5
            action.reasoning = {"fallback": True, "reason": reason, "note": "No availability metadata"}
            return action

        action = available[self.action_counter % len(available)]
        if action.is_complex():
            x, y = self._pick_interesting_coordinate(latest_frame.frame)
            action.set_data({"x": x, "y": y})
        action.reasoning = {
            "fallback": True,
            "reason": reason,
            "strategy": "ordered_legal_action_cycle",
        }
        return action

    def _pick_interesting_coordinate(self, frame_3d: list[list[list[Any]]]) -> tuple[int, int]:
        last_grid = frame_3d[-1] if frame_3d else []
        non_zero = []
        for y, row in enumerate(last_grid[:64]):
            for x, value in enumerate(row[:64]):
                if int(value) != 0:
                    non_zero.append((x, y))
        if non_zero:
            return random.choice(non_zero)
        return random.randint(0, 63), random.randint(0, 63)

    def _available_action_names(self, latest_frame: FrameData) -> list[str]:
        available_actions = latest_frame.available_actions or []
        if not available_actions:
            return [
                "ACTION1",
                "ACTION2",
                "ACTION3",
                "ACTION4",
                "ACTION5",
                "ACTION6",
                "ACTION7",
            ]
        names = []
        for item in available_actions:
            value = int(item.value) if hasattr(item, "value") else int(item)
            names.append(f"ACTION{value}")
        return names

    def _available_model_action_names(self, latest_frame: FrameData) -> list[str]:
        return [
            self.GAME_TO_MODEL_ACTION.get(name, name.lower())
            for name in self._available_action_names(latest_frame)
        ]

    def _model_action_name(self, action: GameAction) -> str:
        return self.GAME_TO_MODEL_ACTION.get(action.name, action.name.lower())

    def _legal_action_instructions(self, available_actions: list[str]) -> str:
        descriptions = {
            "up": '- {"name":"up"}: move up',
            "down": '- {"name":"down"}: move down',
            "left": '- {"name":"left"}: move left',
            "right": '- {"name":"right"}: move right',
            "spacebar": '- {"name":"spacebar"}: activate/confirm',
            "click": '- {"name":"click","x":12,"y":34}: click a coordinate, x/y integers in [0,63]',
            "undo": '- {"name":"undo"}: undo/reverse',
        }
        return "\n".join(descriptions[action] for action in available_actions if action in descriptions)

    def _is_action_available(self, latest_frame: FrameData, action: GameAction) -> bool:
        available_actions = latest_frame.available_actions or []
        if not available_actions:
            return action is not GameAction.RESET
        available_ids = {
            int(item.value) if hasattr(item, "value") else int(item)
            for item in available_actions
        }
        return int(action.value) in available_ids

    def _clamp_coordinate(self, value: Any) -> int:
        try:
            coord = int(value)
        except (TypeError, ValueError):
            coord = 0
        return max(0, min(63, coord))

    def _frame_signature(self, frame_3d: list[list[list[Any]]]) -> dict[str, Any]:
        last_grid = frame_3d[-1] if frame_3d else []
        if not last_grid:
            return {"height": 0, "width": 0, "non_zero": 0}
        height = len(last_grid)
        width = len(last_grid[0]) if last_grid[0] else 0
        non_zero = sum(1 for row in last_grid for value in row if int(value) != 0)
        return {"height": height, "width": width, "non_zero": non_zero}

    def _frame_hash(self, frame_3d: list[list[list[Any]]]) -> str:
        grid = frame_3d[-1] if frame_3d else []
        payload = json.dumps(
            self._comparison_grid(grid), separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _border_ignore_pixels(self) -> int:
        raw_value = os.getenv("LLM_FRAME_BORDER_IGNORE", str(self.FRAME_BORDER_IGNORE))
        try:
            return max(0, int(raw_value))
        except ValueError:
            return self.FRAME_BORDER_IGNORE

    def _comparison_grid(self, grid: list[list[Any]]) -> list[list[Any]]:
        border = self._border_ignore_pixels()
        if border == 0 or len(grid) <= border * 2:
            return grid
        trimmed = []
        for row in grid[border:-border]:
            if len(row) <= border * 2:
                trimmed.append(row)
            else:
                trimmed.append(row[border:-border])
        return trimmed

    def _changed_pixels(
        self,
        before_3d: list[list[list[Any]]],
        after_3d: list[list[list[Any]]],
    ) -> int:
        before = self._comparison_grid(before_3d[-1] if before_3d else [])
        after = self._comparison_grid(after_3d[-1] if after_3d else [])
        height = max(len(before), len(after))
        width = max(
            max((len(row) for row in before), default=0),
            max((len(row) for row in after), default=0),
        )
        changed = 0
        for y in range(height):
            before_row = before[y] if y < len(before) else []
            after_row = after[y] if y < len(after) else []
            for x in range(width):
                before_value = before_row[x] if x < len(before_row) else 0
                after_value = after_row[x] if x < len(after_row) else 0
                if before_value != after_value:
                    changed += 1
        return changed

    def _frame_to_image(self, frame_3d: list[list[list[Any]]]) -> Image.Image:
        last_grid = frame_3d[-1] if frame_3d else []
        if not last_grid:
            last_grid = [[0 for _ in range(64)] for _ in range(64)]

        height = max(len(last_grid), 1)
        width = max(max((len(row) for row in last_grid), default=0), 1)
        image = Image.new("RGB", (width, height), self.ARC_PALETTE[0])
        pixels = []
        for y in range(height):
            row = last_grid[y] if y < len(last_grid) else []
            for x in range(width):
                value = row[x] if x < len(row) else 0
                try:
                    color_index = int(value) % len(self.ARC_PALETTE)
                except (TypeError, ValueError):
                    color_index = 0
                pixels.append(self.ARC_PALETTE[color_index])
        image.putdata(pixels)

        if self.FRAME_IMAGE_SCALE > 1:
            resampling = getattr(Image, "Resampling", Image).NEAREST
            image = image.resize(
                (image.width * self.FRAME_IMAGE_SCALE, image.height * self.FRAME_IMAGE_SCALE),
                resampling,
            )
        return image

    def _build_context_images(
        self,
        latest_frame_3d: list[list[list[Any]]],
        limit: int | None = None,
    ) -> list[Image.Image]:
        frame_limit = limit or self.ACTION_CONTEXT_FRAMES
        recent_entries = self.frame_memory[-frame_limit:] or [
            {"step": self.action_counter, "frame": latest_frame_3d}
        ]
        return [
            self._label_image(
                self._frame_to_image(item["frame"]),
                f"STEP {item['step']}",
            )
            for item in recent_entries
        ]

    def _label_font(self) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        except OSError:
            try:
                return ImageFont.load_default(size=32)
            except TypeError:
                return ImageFont.load_default()

    def _label_image(self, image: Image.Image, label: str) -> Image.Image:
        labeled = image.copy()
        draw = ImageDraw.Draw(labeled)
        draw.text(
            (8, 6),
            label,
            font=self._label_font(),
            fill=(255, 32, 32),
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        return labeled

    def _pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
        return "\n".join(lines)
