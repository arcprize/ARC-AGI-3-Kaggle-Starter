# === Code Cell 1 ===
import os
import subprocess
from pathlib import Path

PROFILE_ENV = {"ARC_AGENT_NAME": "forge_v46_gemma31b_public_single", "ARC_MODEL_PROFILE": "gemma31b_public_single", "LLM_ACTION_CANDIDATES": "1", "LLM_ACTION_CONTEXT_FRAMES": "4", "LLM_CANDIDATE_ARBITER": "0", "LLM_CLICK_FAILURE_RADIUS": "0", "LLM_CONFIDENCE_PROMPT": "0", "LLM_INCLUDE_FRAME_DESCRIPTOR": "0", "LLM_MAX_NEW_TOKENS": "1024", "LLM_MAX_PLAN_ACTIONS": "4", "LLM_REFLECTION_INTERVAL": "10", "LLM_REFLECTION_MAX_NEW_TOKENS": "10000", "LLM_TRACE_IMAGES": "0", "LOCAL_VALIDATION_GAME_IDS": "", "LOCAL_VALIDATION_GAME_TIME_LIMIT_S": "1200", "RUN_ARC_LOCAL_VALIDATION": "1", "VLLM_GENERATION_CONFIG": "", "VLLM_GPU_MEMORY_UTILIZATION": "0.94", "VLLM_LIMIT_MM_PER_PROMPT": "{\"image\": 4}", "VLLM_MAX_MODEL_LEN": "32768", "VLLM_MAX_NUM_SEQS": "20", "VLLM_MODEL_PATH": "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1", "VLLM_QUANTIZATION": ""}
for key, value in PROFILE_ENV.items():
    os.environ[key] = str(value)
print(f"ARC model profile: {os.environ['ARC_MODEL_PROFILE']}")
print(f"vLLM model path: {os.environ['VLLM_MODEL_PATH']}")

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_STARTUP_TIMEOUT"] = "1000"

extract_root = Path("/tmp/vllm_0230_offline")
archive_candidates = [
    Path("/kaggle/input/vllm-0-23-0-tf5-wheelhouse/wheels.tar.gz"),
    Path("/kaggle/input/vllm-0-23-0-tf5/wheels.tar.gz"),
    Path("/kaggle/input/vllm-0-23-0/wheels.tar.gz"),
]
archive_candidates.extend(Path("/kaggle/input").glob("**/wheels.tar.gz"))
for archive_path in archive_candidates:
    if archive_path.exists():
        import tarfile

        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_root)
        print(f"Extracted vLLM wheelhouse from {archive_path}")
        break

wheel_candidates = [
    Path("/kaggle/input/datasets/ko0kip/vllm-0230-offline/vllm_0230_offline/wheels"),
    extract_root / "wheels",
    Path("/kaggle/input/vllm-deps/wheels"),
]
wheel_candidates.extend(
    path for path in Path("/kaggle/input").glob("**/wheels")
    if any(path.glob("vllm*0.23.0*.whl"))
)
VLLM_WHEELS = next((path for path in wheel_candidates if path.exists()), None)
if VLLM_WHEELS is None:
    raise FileNotFoundError("Could not find the offline vLLM 0.23.0 wheelhouse")

subprocess.check_call([
    "uv", "pip", "install",
    "--no-index",
    f"--find-links={VLLM_WHEELS}",
    "vllm==0.23.0",
    "transformers==5.12.1",
])
print(f"Installed vLLM from {VLLM_WHEELS}")


# === Code Cell 2 ===
import shutil
import site
import subprocess
import sys
from pathlib import Path

# vLLM's wheel stack can leave an older Pillow tree behind. Remove it
# before installing the competition wheel's Pillow 12.2.0 dependency.
for base in site.getsitepackages():
    base_path = Path(base)
    shutil.rmtree(base_path / "PIL", ignore_errors=True)
    for dist_info in base_path.glob("pillow*dist-info"):
        shutil.rmtree(dist_info, ignore_errors=True)

arc_wheels = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels")
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-index",
    "--find-links", str(arc_wheels),
    "arc-agi",
    "python-dotenv",
])

from PIL import Image
print(f"Pillow import OK: {Image.__version__}")


# === Code Cell 4 ===
%%writefile /tmp/my_agent.py
from __future__ import annotations

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
from PIL import Image

from agents.agent import Agent

logger = logging.getLogger(__name__)

# All MyAgent instances share one submission budget.  Swarm creates one agent
# per game concurrently, so an instance-local timer would allow the aggregate
# run to exceed Kaggle's wall-clock limit.
_SUBMISSION_STARTED_AT = time.monotonic()


class MyAgent(Agent):
    """vLLM-powered ARC agent that emits one JSON action per step."""

    name = os.getenv("ARC_AGENT_NAME", "forge_v37_gemma31b_multicandidate_arbiter")

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
    ACTION_CANDIDATES = 3
    ARBITER_MAX_NEW_TOKENS = 512
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
    LABEL_GLYPHS = {
        " ": ["000", "000", "000", "000", "000"],
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"],
        "7": ["111", "001", "010", "010", "010"],
        "8": ["111", "101", "111", "101", "111"],
        "9": ["111", "101", "111", "001", "111"],
        "E": ["111", "100", "111", "100", "111"],
        "P": ["111", "101", "111", "100", "100"],
        "S": ["111", "100", "111", "001", "111"],
        "T": ["111", "010", "010", "010", "010"],
    }
    _client: OpenAI | None = None
    _served_model: str | None = None
    _server_process: subprocess.Popen[bytes] | None = None
    _server_log: Any = None
    _server_lock = threading.Lock()
    _vllm_startup_error: str | None = None
    # --- PATCH: bounded vLLM retry-with-backoff state (replaces permanent poison-pill) ---
    _retry_lock = threading.Lock()
    _vllm_failure_count: int = 0
    _vllm_next_retry_monotonic: float = 0.0
    VLLM_MAX_STARTUP_ATTEMPTS = 5
    VLLM_RETRY_COOLDOWN_S = 90.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed_material = ":".join(
            [
                os.getenv("ARC_AGENT_NAME", self.name),
                str(self.game_id),
                os.getenv("AGENT_RANDOM_SEED", "0"),
            ]
        )
        seed = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:12], 16)
        self.rng = random.Random(seed)
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
        # --- PATCH: persistent state-transition graph for graph-guided fallback ---
        # node key -> action key -> {"tried": bool, "dest_hash": str|None,
        #                             "levels_delta": int, "changed": bool}
        # Informed by Rudakov et al. 2025 "Graph-Based Exploration for
        # ARC-AGI-3" (arXiv:2512.24156): explicit state-graph tracking with
        # frontier-driven navigation beats blind/random fallback action
        # selection. We mark an action "tried" the moment it is *issued* by
        # the fallback (not only once its outcome is observed) - this closes
        # the reset-loop bug that paper reports (untested action pointing at
        # a RESET kept getting re-selected because it was only marked tried
        # after seeing the post-reset frame, which looked "new" again).
        self.exploration_graph: dict[str, dict[str, dict[str, Any]]] = {}
        self.current_state_hash: str | None = None
        # PATCH: node key -> number of candidate actions _graph_action_candidates()
        # saw available the *last* time we were physically at that state (i.e.
        # had real frame data to enumerate click targets etc). Lets frontier BFS
        # tell "fully explored" apart from "just under-recorded" for nodes we've
        # actually visited, instead of only ever using the fixed-size proxy.
        self.exploration_graph_candidate_totals: dict[str, int] = {}

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
        # PATCH: bounded retry-with-backoff, replacing the old permanent
        # poison-pill (which set _vllm_startup_error once and never retried,
        # silently degrading the rest of the 8-9hr run to the memoryless
        # fallback after any single transient GPU/model hiccup).
        #
        # PATCH 2 (liveness check): a prior success only proves _load_vllm_once
        # worked THEN. It says nothing about whether the server process is
        # still alive NOW. Without this, a mid-run crash (OOM, driver fault,
        # etc.) after any earlier success was invisible -- _client stayed
        # non-None forever, so this method kept returning instantly as if
        # nothing were wrong, while every real request to the dead server
        # timed out and fell back to blind cycling for the rest of the run.
        if cls._client is not None and cls._served_model is not None:
            with cls._server_lock:
                if cls._server_process is not None and cls._server_process.poll() is not None:
                    logger.warning(
                        "vLLM server process died (exit code %s) after a prior "
                        "success; resetting so the retry logic below restarts it.",
                        cls._server_process.returncode,
                    )
                    cls._client = None
                    cls._served_model = None
            if cls._client is not None and cls._served_model is not None:
                return

        with cls._retry_lock:
            attempts_exhausted = cls._vllm_failure_count >= cls.VLLM_MAX_STARTUP_ATTEMPTS
            now = time.monotonic()
            in_cooldown = now < cls._vllm_next_retry_monotonic

        if attempts_exhausted:
            raise RuntimeError(
                f"vLLM disabled after {cls._vllm_failure_count} failed startup "
                f"attempts: {cls._vllm_startup_error}"
            )
        if in_cooldown:
            raise RuntimeError(
                f"vLLM temporarily unavailable, retrying after cooldown "
                f"(attempt {cls._vllm_failure_count}/{cls.VLLM_MAX_STARTUP_ATTEMPTS}): "
                f"{cls._vllm_startup_error}"
            )

        try:
            cls._load_vllm_once()
            with cls._retry_lock:
                cls._vllm_failure_count = 0
                cls._vllm_startup_error = None
        except Exception as exc:
            with cls._retry_lock:
                cls._vllm_failure_count += 1
                cls._vllm_startup_error = f"{type(exc).__name__}: {exc}"
                cls._vllm_next_retry_monotonic = time.monotonic() + cls.VLLM_RETRY_COOLDOWN_S
                remaining_attempts = cls.VLLM_MAX_STARTUP_ATTEMPTS - cls._vllm_failure_count
            logger.warning(
                "vLLM startup attempt %s/%s failed (%s remaining, %.0fs cooldown): %s",
                cls._vllm_failure_count,
                cls.VLLM_MAX_STARTUP_ATTEMPTS,
                max(0, remaining_attempts),
                cls.VLLM_RETRY_COOLDOWN_S,
                exc,
            )
            raise

    @classmethod
    def _start_vllm_server(cls) -> None:
        if cls._server_process is not None and cls._server_process.poll() is None:
            return

        cls._configure_cuda_library_path()

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
        limit_mm_per_prompt = os.getenv("VLLM_LIMIT_MM_PER_PROMPT", "").strip()
        if limit_mm_per_prompt:
            command.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])
        quantization = os.getenv("VLLM_QUANTIZATION", "").strip()
        if quantization:
            command.extend(["--quantization", quantization])
        generation_config = os.getenv("VLLM_GENERATION_CONFIG", "").strip()
        if generation_config:
            command.extend(["--generation-config", generation_config])
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
    def _configure_cuda_library_path(cls) -> None:
        paths: list[str] = []
        try:
            import site

            for base in site.getsitepackages():
                nvidia_root = os.path.join(base, "nvidia")
                if os.path.isdir(nvidia_root):
                    for root, dirs, _files in os.walk(nvidia_root):
                        if os.path.basename(root) in {"lib", "lib64"}:
                            paths.append(root)
                        dirs[:] = [name for name in dirs if name not in {"__pycache__"}]
                torch_lib = os.path.join(base, "torch", "lib")
                if os.path.isdir(torch_lib):
                    paths.append(torch_lib)
        except Exception as exc:
            logger.warning("Failed to discover CUDA wheel library paths: %s", exc)
        existing = [
            item
            for item in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep)
            if item
        ]
        unique: list[str] = []
        seen = set()
        for path in paths + existing:
            if path and path not in seen and os.path.isdir(path):
                unique.append(path)
                seen.add(path)
        if unique:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(unique)

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

            reflection_interval = self._reflection_interval()
            if reflection_interval and len(self.reflection_buffer) >= reflection_interval:
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
                response_text, parsed = self._generate_action_response(
                    prompt,
                    frame_images,
                    latest_frame,
                )
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
        example_payload: dict[str, Any] = {
            "board_change_assessment": "central-board evidence from the latest transition",
            "plan_summary": "test one rule or pursue the current subgoal",
            "actions": [example_action],
        }
        if self._confidence_prompt_enabled():
            example_payload = {
                "confidence": 0.72,
                "board_change_assessment": "central-board evidence from the latest transition",
                "controllable_object": "object or cursor inferred from transitions",
                "goal_hypothesis": "what must change to advance the level",
                "plan_summary": "test one rule or pursue the current subgoal",
                "actions": [example_action],
            }
        output_example = json.dumps(example_payload, ensure_ascii=True)
        ineffective_actions = self._ineffective_actions_for_current_state(latest_frame)
        legal_action_instructions = self._legal_action_instructions(available_actions)
        frame_descriptor_block = ""
        if self._include_frame_descriptor():
            frame_descriptor = json.dumps(
                self._frame_descriptor(latest_frame.frame),
                ensure_ascii=True,
                separators=(",", ":"),
            )
            frame_descriptor_block = f"\n\nCurrent frame descriptor:\n{frame_descriptor}"
        confidence_instruction = ""
        if self._confidence_prompt_enabled():
            confidence_instruction = (
                "\nInclude confidence from 0.0 to 1.0. If confidence is below 0.55,"
                "\nchoose a reversible diagnostic action instead of a long plan."
            )

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
            {recent_history}{frame_descriptor_block}

            Return exactly one JSON object, no tools or markdown. Include 1 to
            {self._max_plan_actions()} actions; use one exploratory action if uncertain.{confidence_instruction}
            Example: {output_example}
            {thinking_directive}
            """
        ).strip()

    def _env_flag(self, name: str, default: str) -> bool:
        value = os.getenv(name, default).strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _include_frame_descriptor(self) -> bool:
        return self._env_flag("LLM_INCLUDE_FRAME_DESCRIPTOR", "1")

    def _confidence_prompt_enabled(self) -> bool:
        return self._env_flag("LLM_CONFIDENCE_PROMPT", "1")

    def _reflection_interval(self) -> int:
        try:
            return max(0, int(os.getenv("LLM_REFLECTION_INTERVAL", str(self.REFLECTION_INTERVAL))))
        except ValueError:
            return self.REFLECTION_INTERVAL

    def _reflection_max_new_tokens(self) -> int:
        try:
            return max(
                64,
                int(os.getenv("LLM_REFLECTION_MAX_NEW_TOKENS", str(self.REFLECTION_MAX_NEW_TOKENS))),
            )
        except ValueError:
            return self.REFLECTION_MAX_NEW_TOKENS

    def _action_context_frames(self) -> int:
        try:
            return max(1, int(os.getenv("LLM_ACTION_CONTEXT_FRAMES", str(self.ACTION_CONTEXT_FRAMES))))
        except ValueError:
            return self.ACTION_CONTEXT_FRAMES

    def _max_plan_actions(self) -> int:
        try:
            return max(1, min(8, int(os.getenv("LLM_MAX_PLAN_ACTIONS", str(self.MAX_PLAN_ACTIONS)))))
        except ValueError:
            return self.MAX_PLAN_ACTIONS

    def _generate_action_response(
        self,
        prompt: str,
        frame_images: list[Image.Image],
        latest_frame: FrameData,
    ) -> tuple[str, dict[str, Any]]:
        candidate_count = self._action_candidate_count()
        if candidate_count <= 1:
            response_text = self._generate_response(
                prompt,
                frame_images,
                enable_thinking=self._action_thinking_enabled(),
            )
            parsed, response_text = self._parse_or_repair_action_json(
                prompt,
                response_text,
                frame_images,
            )
            return response_text, parsed

        responses = self._generate_responses(
            prompt,
            frame_images,
            enable_thinking=self._action_thinking_enabled(),
            choice_count=candidate_count,
        )
        candidates: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, response_text in enumerate(responses):
            try:
                parsed, full_text = self._parse_or_repair_action_json(
                    prompt,
                    response_text,
                    frame_images,
                )
                normalized = self._normalize_action_specs(parsed, latest_frame)
                if not normalized:
                    errors.append(f"candidate {index}: no usable actions")
                    continue
                candidates.append(
                    {
                        "index": index,
                        "response_text": full_text,
                        "parsed": parsed,
                        "actions": normalized,
                        "score": self._candidate_static_score(
                            parsed,
                            normalized,
                            latest_frame,
                            index,
                        ),
                    }
                )
            except Exception as exc:
                errors.append(f"candidate {index}: {type(exc).__name__}: {exc}")

        if not candidates:
            raise ValueError(
                f"All {len(responses)} model candidates failed: {errors[:4]}"
            )

        selected = max(candidates, key=lambda item: item["score"])
        if len(candidates) > 1 and self._candidate_arbiter_enabled():
            selected = self._select_candidate_with_arbiter(
                frame_images,
                latest_frame,
                candidates,
                selected,
            )
        parsed = dict(selected["parsed"])
        parsed["_candidate_selection"] = {
            "candidate_count": len(responses),
            "valid_candidates": len(candidates),
            "selected_index": selected["index"],
            "static_score": selected["score"],
            "errors": errors[:4],
        }
        return selected["response_text"], parsed

    def _parse_or_repair_action_json(
        self,
        prompt: str,
        response_text: str,
        frame_images: list[Image.Image],
    ) -> tuple[dict[str, Any], str]:
        try:
            return self._extract_action_json(response_text), response_text
        except Exception:
            repair_response = self._generate_response(
                self._build_json_repair_prompt(prompt, response_text),
                frame_images,
                enable_thinking=False,
                max_new_tokens=self.REPAIR_MAX_NEW_TOKENS,
            )
            full_text = response_text + "\n\nJSON_REPAIR_OUTPUT:\n" + repair_response
            return self._extract_action_json(repair_response), full_text

    def _action_candidate_count(self) -> int:
        try:
            count = int(os.getenv("LLM_ACTION_CANDIDATES", str(self.ACTION_CANDIDATES)))
        except ValueError:
            count = self.ACTION_CANDIDATES
        if self.game_time_remaining_s < float(os.getenv("LLM_CANDIDATE_MIN_SECONDS", "900")):
            return 1
        return max(1, min(5, count))

    def _candidate_arbiter_enabled(self) -> bool:
        value = os.getenv("LLM_CANDIDATE_ARBITER", "1").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _candidate_static_score(
        self,
        parsed: dict[str, Any],
        actions: list[dict[str, Any]],
        latest_frame: FrameData,
        index: int,
    ) -> float:
        score = 100.0 - index * 0.25
        confidence = self._coerce_confidence(parsed.get("confidence"))
        score += confidence * 8.0
        if len(actions) > 1:
            score += min(len(actions), self.MAX_PLAN_ACTIONS) * 0.35
        if actions and actions[0].get("name") == "RESET":
            score -= 25.0
        if actions and actions[0].get("name") == "ACTION6":
            score += self._click_action_score(actions[0], latest_frame)
        summary = str(parsed.get("plan_summary", "")).lower()
        assessment = str(parsed.get("board_change_assessment", "")).lower()
        useful_words = (
            "goal",
            "win",
            "complete",
            "match",
            "move",
            "collect",
            "open",
            "unlock",
            "test",
            "progress",
        )
        score += sum(0.15 for word in useful_words if word in summary or word in assessment)
        risky_words = ("random", "guess", "uncertain", "stuck", "repeat")
        score -= sum(0.5 for word in risky_words if word in summary or word in assessment)
        return score

    def _coerce_confidence(self, raw_value: Any) -> float:
        if isinstance(raw_value, (int, float)):
            return max(0.0, min(1.0, float(raw_value)))
        text = str(raw_value or "").strip().lower()
        if not text:
            return 0.5
        if text.endswith("%"):
            try:
                return max(0.0, min(1.0, float(text[:-1]) / 100.0))
            except ValueError:
                return 0.5
        labels = {
            "low": 0.25,
            "medium": 0.55,
            "med": 0.55,
            "high": 0.8,
            "certain": 0.95,
        }
        if text in labels:
            return labels[text]
        try:
            value = float(text)
            return max(0.0, min(1.0, value if value <= 1.0 else value / 100.0))
        except ValueError:
            return 0.5

    def _click_action_score(self, action_spec: dict[str, Any], latest_frame: FrameData) -> float:
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        x = self._clamp_coordinate(action_spec.get("x", 0))
        y = self._clamp_coordinate(action_spec.get("y", 0))
        if not grid or y >= len(grid) or x >= len(grid[y]):
            return -2.0
        value = int(grid[y][x])
        if value == 0:
            return -0.4
        non_zero = [
            (xx, yy)
            for yy, row in enumerate(grid)
            for xx, cell in enumerate(row)
            if int(cell) != 0
        ]
        if not non_zero:
            return 0.0
        min_x = min(xx for xx, _ in non_zero)
        max_x = max(xx for xx, _ in non_zero)
        min_y = min(yy for _, yy in non_zero)
        max_y = max(yy for _, yy in non_zero)
        margin = 2
        inside_content = min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin
        return 1.2 if inside_content else 0.2

    def _select_candidate_with_arbiter(
        self,
        frame_images: list[Image.Image],
        latest_frame: FrameData,
        candidates: list[dict[str, Any]],
        default_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        arbiter_payload = []
        for item in candidates:
            parsed = item["parsed"]
            arbiter_payload.append(
                {
                    "id": item["index"],
                    "actions": item["actions"],
                    "confidence": parsed.get("confidence"),
                    "plan_summary": str(parsed.get("plan_summary", ""))[:500],
                    "board_change_assessment": str(parsed.get("board_change_assessment", ""))[:500],
                    "static_score": round(float(item["score"]), 3),
                }
            )
        prompt = textwrap.dedent(
            f"""
            You are choosing among candidate action plans for the same ARC-AGI-3 state.
            The images are chronological with red STEP labels; the last image is current.
            Pick the plan most likely to complete the current level with few actions.
            Penalize repeated-state guesses, unsupported clicks, and reset unless necessary.

            State: {latest_frame.state.name}
            Levels completed: {latest_frame.levels_completed}
            Available actions: {self._available_model_action_names(latest_frame)}
            Ineffective in this exact state: {self._ineffective_actions_for_current_state(latest_frame)}
            Recent transitions: {json.dumps(self._prompt_history()[-4:], ensure_ascii=True)}

            Candidate plans:
            {json.dumps(arbiter_payload, ensure_ascii=True)}

            Return exactly one JSON object: {{"choice": <candidate id>, "reason": "short evidence"}}
            /no_think
            """
        ).strip()
        try:
            response_text = self._generate_response(
                prompt,
                frame_images,
                enable_thinking=False,
                max_new_tokens=self.ARBITER_MAX_NEW_TOKENS,
                json_mode=True,
            )
            parsed = self._extract_action_json(response_text)
            choice = int(parsed.get("choice"))
            for item in candidates:
                if item["index"] == choice:
                    item["parsed"] = dict(item["parsed"])
                    item["parsed"]["_arbiter"] = {
                        "choice": choice,
                        "reason": str(parsed.get("reason", ""))[:500],
                    }
                    return item
        except Exception as exc:
            logger.info("Candidate arbiter failed; using static score: %s", exc)
        return default_candidate

    def _generate_response(
        self,
        prompt: str,
        frame_images: list[Image.Image],
        enable_thinking: bool,
        max_new_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str:
        return self._generate_responses(
            prompt,
            frame_images,
            enable_thinking,
            max_new_tokens=max_new_tokens,
            json_mode=json_mode,
            choice_count=1,
        )[0]

    def _generate_responses(
        self,
        prompt: str,
        frame_images: list[Image.Image],
        enable_thinking: bool,
        max_new_tokens: int | None = None,
        json_mode: bool = True,
        choice_count: int = 1,
    ) -> list[str]:
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
        if choice_count > 1:
            request_kwargs["n"] = max(1, min(5, int(choice_count)))
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
        completion_tokens = (
            response.usage.completion_tokens if response.usage is not None else None
        )
        texts: list[str] = []
        finish_reasons = []
        for choice in response.choices:
            content = choice.message.content or ""
            if not isinstance(content, str):
                content = "".join(str(part) for part in content)
            texts.append(content.strip())
            finish_reasons.append(choice.finish_reason)
            if choice.finish_reason == "length":
                logger.warning(
                    "vLLM output reached token budget=%s without a stop token",
                    token_budget,
                )
        if self._timing_enabled():
            logger.info(
                "vLLM timing step=%s images=%s thinking=%s choices=%s finish=%s "
                "completion_tokens=%s budget=%s total=%.3fs",
                self.action_counter,
                [f"{image.width}x{image.height}" for image in frame_images],
                enable_thinking,
                len(texts),
                finish_reasons,
                completion_tokens,
                token_budget,
                time.perf_counter() - response_start,
            )
        return texts or [""]

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
        for item in raw_actions[: self._max_plan_actions()]:
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
            if action is GameAction.ACTION6 and self._click_near_failed(spec, latest_frame):
                logger.info("Skipping click near failed point in current state: %s", spec)
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
            if self._click_near_failed(spec, latest_frame):
                self.pending_actions = []
                return self._fallback_action(
                    latest_frame,
                    f"Planned click near failed point in this state: {spec}.",
                )
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
        reflection_interval = self._reflection_interval()
        transitions = json.dumps(
            self.reflection_buffer[-reflection_interval :], ensure_ascii=True
        )
        return textwrap.dedent(
            f"""
            You are the reflection agent for an ARC-AGI-3 game. Review the previous
            memory, the last {reflection_interval} completed transitions, and the
            chronological images. The final image is current. Pixel changes may be movement, transformation, collection,
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
        reflection_interval = self._reflection_interval()
        if not reflection_interval or len(self.reflection_buffer) < reflection_interval:
            return
        # A reflection may revise the goal, so discard any stale queued plan.
        self.pending_actions = []
        prompt = self._build_reflection_prompt(latest_frame)
        # BUGFIX (Jul 5, 2026): reflection_interval + 1 images (11 by default)
        # exceeds the vLLM server's hard per-prompt image cap
        # (VLLM_LIMIT_MM_PER_PROMPT, default 4). Every reflection call was
        # failing with a 400 error and being silently swallowed below --
        # reflection has never actually succeeded. Cap at the same safe
        # budget action calls already use.
        images = self._build_context_images(
            latest_frame.frame,
            limit=min(reflection_interval + 1, self._action_context_frames()),
        )
        try:
            response = self._generate_response(
                prompt,
                images,
                enable_thinking=False,
                max_new_tokens=self._reflection_max_new_tokens(),
                json_mode=False,
            )
            self.reflection_memory = self._clean_reflection_markdown(response)
            self._save_reflection_memory()
            self.reflections_completed += 1
            logger.info(
                "Reflection completed for %s after %s transitions; memory=%s",
                self.game_id,
                reflection_interval,
                self.reflection_memory_path,
            )
        except Exception as exc:
            logger.warning("Reflection failed for %s: %s", self.game_id, exc)
        finally:
            del self.reflection_buffer[:reflection_interval]

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
                    "format": "ordered PIL RGB images with red STEP labels",
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
        self.current_state_hash = current_hash
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
        repeated_state = any(
            item["frame_hash"] == current_hash for item in self.frame_memory[:-1]
        )
        # PATCH: record the observed transition as an edge in the exploration
        # graph, keyed by the same failure_key/chosen_action used elsewhere so
        # click actions at different coordinates are distinct edges.
        graph_action_key = previous_action.get("failure_key") or previous_action.get(
            "chosen_action"
        )
        if graph_action_key:
            node = self.exploration_graph.setdefault(before_entry["frame_hash"], {})
            node[graph_action_key] = {
                "tried": True,
                "dest_hash": current_hash,
                "levels_delta": levels_delta,
                "changed": changed_pixels > 0 or current_hash != before_entry["frame_hash"],
            }
        if changed_pixels == 0 and levels_delta == 0:
            failed_actions = getattr(self, "failed_state_actions", None)
            if failed_actions is None:
                self.failed_state_actions = {}
                failed_actions = self.failed_state_actions
            failed_actions.setdefault(before_entry["frame_hash"], set()).add(
                previous_action.get("failure_key") or previous_action["chosen_action"]
            )
            self.pending_actions = []
        elif repeated_state and levels_delta == 0:
            self.pending_actions = []
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
                "repeated_state": repeated_state,
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
        trace_images = os.getenv("LLM_TRACE_IMAGES", "0").strip().lower()
        if trace_images not in {"1", "true", "yes", "on"}:
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

    # PATCH: fallback action selection now tries graph-guided exploration
    # first (informed by Rudakov et al. 2025, arXiv:2512.24156 - "Graph-Based
    # Exploration for ARC-AGI-3"), falling back to the original blind
    # round-robin cycle only if the graph has nothing actionable yet (e.g.
    # very first fallback call, or an isolated/unreachable frontier). This
    # means a long vLLM outage degrades to *directed* exploration instead of
    # memoryless cycling, while never being worse than the old behavior.
    SIMPLE_FALLBACK_ACTIONS = ("up", "down", "left", "right", "spacebar", "undo")
    GRAPH_CLICK_CANDIDATES = 6

    def _fallback_action(self, latest_frame: FrameData, reason: str) -> GameAction:
        try:
            action = self._graph_guided_fallback_action(latest_frame, reason)
            if action is not None:
                return action
        except Exception as exc:  # never let the fallback-of-last-resort itself throw
            logger.warning("Graph-guided fallback failed (%s), using legacy cycle: %s", reason, exc)
        return self._legacy_cycle_fallback_action(latest_frame, reason)

    def _graph_action_candidates(
        self, latest_frame: FrameData
    ) -> list[tuple[str, int, GameAction, dict[str, int] | None]]:
        """Return (action_key, priority, GameAction, click_data_or_None) tuples
        for every action available in the current state, lowest priority
        number = tried first (mirrors the paper's tiered salience heuristic:
        cheap/simple actions before expensive click enumeration)."""
        ineffective_actions = set(self._ineffective_actions_for_current_state(latest_frame))
        candidates: list[tuple[str, int, GameAction, dict[str, int] | None]] = []
        simple_lookup = {
            "up": GameAction.ACTION1,
            "down": GameAction.ACTION2,
            "left": GameAction.ACTION3,
            "right": GameAction.ACTION4,
            "spacebar": GameAction.ACTION5,
            "undo": GameAction.ACTION7,
        }
        for name in self.SIMPLE_FALLBACK_ACTIONS:
            action = simple_lookup[name]
            if self._is_action_available(latest_frame, action) and name not in ineffective_actions:
                candidates.append((name, 0, action, None))

        if self._is_action_available(latest_frame, GameAction.ACTION6):
            non_zero = {
                (x, y)
                for y, row in enumerate((latest_frame.frame[-1] if latest_frame.frame else [])[:64])
                for x, value in enumerate(row[:64])
                if int(value) != 0
            }
            for x, y in self._component_centers(non_zero)[: self.GRAPH_CLICK_CANDIDATES]:
                key = f"click@{x},{y}"
                if key in ineffective_actions:
                    continue
                candidates.append((key, 1, GameAction.ACTION6, {"x": x, "y": y}))

        if self._is_action_available(latest_frame, GameAction.RESET) and "reset" not in ineffective_actions:
            candidates.append(("reset", 2, GameAction.RESET, None))
        return candidates

    def _graph_guided_fallback_action(
        self, latest_frame: FrameData, reason: str
    ) -> GameAction | None:
        current_hash = self.current_state_hash or self._frame_hash(latest_frame.frame)
        candidates = self._graph_action_candidates(latest_frame)
        if not candidates:
            return None
        # PATCH: we have real frame data right now, so we know exactly how many
        # actions are available at this node -- record it for frontier BFS.
        self.exploration_graph_candidate_totals[current_hash] = len(candidates)

        node = self.exploration_graph.get(current_hash, {})
        untested_here = [c for c in candidates if not node.get(c[0], {}).get("tried")]
        if untested_here:
            key, _priority, action, click_data = min(untested_here, key=lambda c: c[1])
            return self._materialize_graph_action(
                key, action, click_data, latest_frame, reason, "graph_untested_here"
            )

        # Nothing untested from the current state: BFS the known graph for
        # the nearest node that still has an untested action, and take the
        # first step toward it. "Has untested actions" for a *remote* node is
        # a heuristic proxy (we only know the actions we've already recorded
        # edges for at that node, not its true action space) - documented
        # limitation, consistent with the paper's own frontier-distance
        # approximation.
        path = self._graph_bfs_path_to_frontier(current_hash)
        if path:
            first_action_key = path[0]
            edge = node.get(first_action_key)
            action, click_data = self._graph_key_to_action(first_action_key)
            if action is not None:
                return self._materialize_graph_action(
                    first_action_key, action, click_data, latest_frame, reason, "graph_bfs_to_frontier"
                )
        return None

    def _graph_bfs_path_to_frontier(self, start_hash: str) -> list[str] | None:
        """BFS over recorded (tried, dest known) edges to the nearest node
        that still has at least one candidate action we haven't recorded an
        edge for yet. Returns the list of action_keys along the shortest
        path, or None."""
        from collections import deque

        visited = {start_hash}
        queue: deque[tuple[str, list[str]]] = deque([(start_hash, [])])
        while queue:
            state_hash, path = queue.popleft()
            node = self.exploration_graph.get(state_hash, {})
            if state_hash != start_hash:
                # PATCH: every recorded edge is marked tried the instant it is
                # created (see _observe_frame / _materialize_graph_action), so
                # "any recorded-but-untested edge" can never actually occur --
                # that check was dead code. What we actually know: if we've
                # physically been at this node before, exploration_graph_candidate_totals
                # tells us exactly how many candidates existed there, so we can
                # tell "fully explored" from "still has untried options". For a
                # node we've only ever heard about secondhand (a neighbor's
                # edge names it as a destination, but we've never stood there
                # ourselves), fall back to the thin-node proxy.
                known_total = self.exploration_graph_candidate_totals.get(state_hash)
                is_frontier = len(node) < known_total if known_total is not None else len(node) < 3
                if is_frontier:
                    return path
            for action_key, edge in node.items():
                dest = edge.get("dest_hash")
                if dest and dest not in visited:
                    visited.add(dest)
                    queue.append((dest, path + [action_key]))
        return None

    def _graph_key_to_action(
        self, action_key: str
    ) -> tuple[GameAction | None, dict[str, int] | None]:
        match = re.fullmatch(r"click@(\d+),(\d+)", action_key)
        if match:
            return GameAction.ACTION6, {"x": int(match.group(1)), "y": int(match.group(2))}
        if action_key == "reset":
            return GameAction.RESET, None
        game_name = self.MODEL_TO_GAME_ACTION.get(action_key)
        if game_name is None:
            return None, None
        return getattr(GameAction, game_name, None), None

    def _materialize_graph_action(
        self,
        action_key: str,
        action: GameAction,
        click_data: dict[str, int] | None,
        latest_frame: FrameData,
        reason: str,
        strategy: str,
    ) -> GameAction:
        # Mark tried the instant we *issue* the action, not only once we
        # later observe its outcome in _observe_frame. This closes the
        # reset-loop bug reported in Rudakov et al. 2025: without this,
        # an untested edge pointing at RESET (or any action whose outcome
        # we fail to observe before the run ends) keeps getting re-selected
        # as "still untested" forever.
        current_hash = self.current_state_hash or self._frame_hash(latest_frame.frame)
        node = self.exploration_graph.setdefault(current_hash, {})
        node.setdefault(action_key, {"tried": True, "dest_hash": None, "levels_delta": 0, "changed": False})
        node[action_key]["tried"] = True

        if action.is_complex():
            if click_data is not None:
                action.set_data(click_data)
            else:
                x, y = self._pick_interesting_coordinate(latest_frame.frame, latest_frame)
                action.set_data({"x": x, "y": y})
        action.reasoning = {
            "fallback": True,
            "reason": reason,
            "strategy": strategy,
            "action_key": action_key,
        }
        return action

    def _legacy_cycle_fallback_action(self, latest_frame: FrameData, reason: str) -> GameAction:
        """Original blind round-robin fallback, kept as a safety net for
        when the exploration graph is empty or offers no reachable frontier
        (e.g. the very first fallback call of a game)."""
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
            x, y = self._pick_interesting_coordinate(latest_frame.frame, latest_frame)
            action.set_data({"x": x, "y": y})
        action.reasoning = {
            "fallback": True,
            "reason": reason,
            "strategy": "ordered_legal_action_cycle",
        }
        return action

    def _pick_interesting_coordinate(
        self,
        frame_3d: list[list[list[Any]]],
        latest_frame: FrameData | None = None,
    ) -> tuple[int, int]:
        last_grid = frame_3d[-1] if frame_3d else []
        non_zero = []
        failed_points = (
            self._failed_click_points_for_current_state(latest_frame)
            if latest_frame is not None
            else []
        )
        radius = self._click_failure_radius()
        for y, row in enumerate(last_grid[:64]):
            for x, value in enumerate(row[:64]):
                if int(value) != 0:
                    non_zero.append((x, y))
        if failed_points and radius > 0:
            filtered = [
                (x, y)
                for x, y in non_zero
                if all(abs(x - fx) > radius or abs(y - fy) > radius for fx, fy in failed_points)
            ]
            if filtered:
                non_zero = filtered
        if non_zero:
            component_centers = self._component_centers(set(non_zero))
            if component_centers:
                index = self.action_counter % len(component_centers)
                return component_centers[index]
            return non_zero[self.action_counter % len(non_zero)]
        return self.rng.randint(0, 63), self.rng.randint(0, 63)

    def _component_centers(
        self,
        allowed_points: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        components: list[tuple[int, int, int, int, int, int]] = []
        for start in sorted(allowed_points):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            cells: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    point = (nx, ny)
                    if point in seen or point not in allowed_points:
                        continue
                    seen.add(point)
                    stack.append(point)
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            area = len(cells)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            center_x = round(sum(xs) / area)
            center_y = round(sum(ys) / area)
            components.append((area, (max_x - min_x + 1) * (max_y - min_y + 1), center_x, center_y, min_x, min_y))
        components.sort()
        return [(center_x, center_y) for _area, _bbox_area, center_x, center_y, _min_x, _min_y in components]

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
            "click": '- {"name":"click","x":12,"y":34}: click original grid coordinates, x/y integers in [0,63], not scaled image pixels or STEP-label pixels',
            "undo": '- {"name":"undo"}: undo/reverse',
        }
        return "\n".join(descriptions[action] for action in available_actions if action in descriptions)

    def _frame_descriptor(self, frame_3d: list[list[list[Any]]]) -> dict[str, Any]:
        grid = frame_3d[-1] if frame_3d else []
        if not grid:
            return {"height": 0, "width": 0, "colors": {}}
        height = len(grid)
        width = max((len(row) for row in grid), default=0)
        colors: dict[str, dict[str, Any]] = {}
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                color = int(value)
                if color == 0:
                    continue
                key = str(color)
                item = colors.setdefault(
                    key,
                    {"count": 0, "bbox": [x, y, x, y], "sample": []},
                )
                item["count"] += 1
                bbox = item["bbox"]
                bbox[0] = min(bbox[0], x)
                bbox[1] = min(bbox[1], y)
                bbox[2] = max(bbox[2], x)
                bbox[3] = max(bbox[3], y)
                if len(item["sample"]) < 6:
                    item["sample"].append([x, y])
        top_colors = sorted(colors.items(), key=lambda pair: pair[1]["count"], reverse=True)[:10]
        return {
            "height": height,
            "width": width,
            "nonzero_colors": {key: value for key, value in top_colors},
        }

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

    def _click_failure_radius(self) -> int:
        try:
            return max(0, int(os.getenv("LLM_CLICK_FAILURE_RADIUS", "0")))
        except ValueError:
            return 0

    def _failed_click_points_for_current_state(
        self, latest_frame: FrameData | None
    ) -> list[tuple[int, int]]:
        if latest_frame is None:
            return []
        points: list[tuple[int, int]] = []
        for key in self._ineffective_actions_for_current_state(latest_frame):
            match = re.fullmatch(r"click@(\d+),(\d+)", key)
            if match:
                points.append((int(match.group(1)), int(match.group(2))))
        return points

    def _click_near_failed(
        self,
        spec: dict[str, Any],
        latest_frame: FrameData,
    ) -> bool:
        radius = self._click_failure_radius()
        if radius <= 0:
            return False
        x = self._clamp_coordinate(spec.get("x", 0))
        y = self._clamp_coordinate(spec.get("y", 0))
        for failed_x, failed_y in self._failed_click_points_for_current_state(latest_frame):
            if abs(x - failed_x) <= radius and abs(y - failed_y) <= radius:
                return True
        return False

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
        frame_limit = limit or self._action_context_frames()
        recent_entries = self.frame_memory[-frame_limit:] or [
            {"step": self.action_counter, "frame": latest_frame_3d}
        ]
        return [
            self._label_image(self._frame_to_image(item["frame"]), f"STEP {item['step']}")
            for item in recent_entries
        ]

    def _label_font(self, image_font: Any) -> Any:
        try:
            return image_font.truetype("DejaVuSans-Bold.ttf", 32)
        except OSError:
            try:
                return image_font.load_default(size=32)
            except TypeError:
                return image_font.load_default()

    def _label_image(self, image: Image.Image, label: str) -> Image.Image:
        labeled = image.copy()
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(labeled)
            draw.text(
                (8, 6),
                label,
                font=self._label_font(ImageFont),
                fill=(255, 32, 32),
                stroke_width=3,
                stroke_fill=(0, 0, 0),
            )
            return labeled
        except Exception as exc:
            logger.warning("Falling back to bitmap STEP label: %s", exc)
            self._draw_bitmap_text(labeled, label.upper(), 4, 4, 2)
            return labeled

    def _draw_bitmap_text(
        self,
        image: Image.Image,
        text: str,
        x: int,
        y: int,
        scale: int,
        color: tuple[int, int, int] = (255, 65, 54),
    ) -> None:
        cursor = x
        for char in text:
            glyph = self.LABEL_GLYPHS.get(char, self.LABEL_GLYPHS[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit != "1":
                        continue
                    left = cursor + gx * scale
                    top = y + gy * scale
                    for py in range(scale):
                        for px in range(scale):
                            xx = left + px
                            yy = top + py
                            if 0 <= xx < image.width and 0 <= yy < image.height:
                                image.putpixel((xx, yy), color)
            cursor += 4 * scale
            if cursor >= image.width:
                break

    def _pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
        return "\n".join(lines)


# === Code Cell 5 ===
import base64
import io
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from PIL import Image

PROFILE_ENV = {"ARC_AGENT_NAME": "forge_v46_gemma31b_public_single", "ARC_MODEL_PROFILE": "gemma31b_public_single", "LLM_ACTION_CANDIDATES": "1", "LLM_ACTION_CONTEXT_FRAMES": "4", "LLM_CANDIDATE_ARBITER": "0", "LLM_CLICK_FAILURE_RADIUS": "0", "LLM_CONFIDENCE_PROMPT": "0", "LLM_INCLUDE_FRAME_DESCRIPTOR": "0", "LLM_MAX_NEW_TOKENS": "1024", "LLM_MAX_PLAN_ACTIONS": "4", "LLM_REFLECTION_INTERVAL": "10", "LLM_REFLECTION_MAX_NEW_TOKENS": "10000", "LLM_TRACE_IMAGES": "0", "LOCAL_VALIDATION_GAME_IDS": "", "LOCAL_VALIDATION_GAME_TIME_LIMIT_S": "1200", "RUN_ARC_LOCAL_VALIDATION": "1", "VLLM_GENERATION_CONFIG": "", "VLLM_GPU_MEMORY_UTILIZATION": "0.94", "VLLM_LIMIT_MM_PER_PROMPT": "{\"image\": 4}", "VLLM_MAX_MODEL_LEN": "32768", "VLLM_MAX_NUM_SEQS": "20", "VLLM_MODEL_PATH": "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1", "VLLM_QUANTIZATION": ""}
for key, value in PROFILE_ENV.items():
    os.environ[key] = str(value)

# Commit-mode guardrail: prove the final single-file agent imports
# under Kaggle's offline framework before we risk a competition rerun.
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    framework_src = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents')
    smoke_dir = Path('/kaggle/working/ARC-AGI-3-Agents-smoke')
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    try:
        shutil.copytree(framework_src, smoke_dir)
        (smoke_dir / 'agents' / '__init__.py').write_text(
            "from .agent import Agent, Playback\n"
            "from .swarm import Swarm\n",
            encoding='utf-8',
        )
        sys.path.insert(0, str(smoke_dir))

        spec = importlib.util.spec_from_file_location('my_agent_smoke', '/tmp/my_agent.py')
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        print(f'Smoke import OK: {module.MyAgent.__name__}')

        from importlib.metadata import version
        print(f"vLLM package: {version('vllm')}")
        print(f"Transformers package: {version('transformers')}")

        local_validation_requested = os.getenv('RUN_ARC_LOCAL_VALIDATION', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
        if local_validation_requested:
            print('Skipping vLLM startup smoke; RUN_ARC_LOCAL_VALIDATION will exercise the full server path.')
        elif os.getenv('RUN_VLLM_STARTUP_SMOKE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}:
            print('Starting vLLM startup smoke...')
            try:
                module.MyAgent._ensure_vllm_available()
                print(f'vLLM startup smoke OK: {module.MyAgent._served_model}')
                if os.getenv('RUN_VLLM_GENERATION_SMOKE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}:
                    image = Image.new('RGB', (64, 64), (0, 0, 0))
                    image.putpixel((31, 31), (255, 0, 0))
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    image_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
                    smoke_n = max(1, int(os.getenv('LLM_ACTION_CANDIDATES', '1')))
                    request = {
                        'model': module.MyAgent._served_model,
                        'messages': [{
                            'role': 'user',
                            'content': [
                                {'type': 'image_url', 'image_url': {'url': image_url}},
                                {'type': 'text', 'text': 'Return JSON only: {"ok": true, "color": "red"}'},
                            ],
                        }],
                        'max_tokens': 24,
                        'temperature': 0.2,
                        'response_format': {'type': 'json_object'},
                    }
                    if smoke_n > 1:
                        request['n'] = min(5, smoke_n)
                    smoke_response = module.MyAgent._client.chat.completions.create(**request)
                    print(f'vLLM image generation smoke choices: {len(smoke_response.choices)}')
            finally:
                proc = getattr(module.MyAgent, '_server_process', None)
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except Exception:
                        proc.kill()
                        proc.wait()
                log_handle = getattr(module.MyAgent, '_server_log', None)
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
    finally:
        try:
            sys.path.remove(str(smoke_dir))
        except ValueError:
            pass
        shutil.rmtree(smoke_dir, ignore_errors=True)


# === Code Cell 6 ===
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request

GATEWAY_GAMES_URL = 'http://gateway:8001/api/games'
ARC_API_KEY = os.getenv('ARC_API_KEY') or 'test-key-123'
PROFILE_ENV = {"ARC_AGENT_NAME": "forge_v46_gemma31b_public_single", "ARC_MODEL_PROFILE": "gemma31b_public_single", "LLM_ACTION_CANDIDATES": "1", "LLM_ACTION_CONTEXT_FRAMES": "4", "LLM_CANDIDATE_ARBITER": "0", "LLM_CLICK_FAILURE_RADIUS": "0", "LLM_CONFIDENCE_PROMPT": "0", "LLM_INCLUDE_FRAME_DESCRIPTOR": "0", "LLM_MAX_NEW_TOKENS": "1024", "LLM_MAX_PLAN_ACTIONS": "4", "LLM_REFLECTION_INTERVAL": "20", "LLM_REFLECTION_MAX_NEW_TOKENS": "3000", "LLM_TRACE_IMAGES": "0", "LOCAL_VALIDATION_GAME_IDS": "", "LOCAL_VALIDATION_GAME_LIMIT": "8", "LOCAL_VALIDATION_GAME_TIME_LIMIT_S": "1200", "RUN_ARC_LOCAL_VALIDATION": "1", "VLLM_GENERATION_CONFIG": "", "VLLM_GPU_MEMORY_UTILIZATION": "0.94", "VLLM_LIMIT_MM_PER_PROMPT": "{\"image\": 4}", "VLLM_MAX_MODEL_LEN": "32768", "VLLM_MAX_NUM_SEQS": "20", "VLLM_MODEL_PATH": "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1", "VLLM_QUANTIZATION": ""}
for key, value in PROFILE_ENV.items():
    os.environ[key] = str(value)

def profile_env_text() -> str:
    return ''.join(f'{key}={value}\n' for key, value in PROFILE_ENV.items())

def gateway_available() -> bool:
    request = urllib.request.Request(
        GATEWAY_GAMES_URL,
        headers={'X-API-Key': ARC_API_KEY, 'Accept': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            print(f'Gateway probe status: {response.status}')
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        print(f'Gateway probe HTTP error: {exc.code} {exc.reason}')
    except Exception as exc:
        print(f'Gateway not available in this run: {exc!r}')
    return False

def wait_for_gateway(max_wait_sec: int) -> bool:
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        if gateway_available():
            return True
        time.sleep(5)
    return False

is_competition_rerun = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
should_run_gateway = is_competition_rerun or gateway_available()
run_local_validation = (
    not is_competition_rerun
    and os.getenv('RUN_ARC_LOCAL_VALIDATION', '0').strip().lower()
    in {'1', 'true', 'yes', 'on'}
)

def prepare_framework() -> Path:
    agents_wd = Path('/kaggle/working/ARC-AGI-3-Agents')
    if agents_wd.exists():
        shutil.rmtree(agents_wd)
    shutil.copytree(
        '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents',
        agents_wd,
    )
    shutil.copyfile(
        '/tmp/my_agent.py',
        agents_wd / 'agents' / 'templates' / 'my_agent.py',
    )
    (agents_wd / 'agents' / '__init__.py').write_text(
        "from typing import Type\n"
        "from dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\n"
        "from .swarm import Swarm\n"
        "from .templates.random_agent import Random\n"
        "from .templates.my_agent import MyAgent\n\n"
        "load_dotenv()\n\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n"
        "    'random': Random,\n"
        "    'myagent': MyAgent,\n"
        "}\n",
        encoding='utf-8',
    )
    return agents_wd

def summarize_local_validation(log_path: Path) -> None:
    if not log_path.exists():
        print(f'Local validation summary unavailable; missing {log_path}')
        return
    pattern = re.compile(
        r'INFO \| ([a-z0-9]+-[a-f0-9]+) - ([A-Z0-9]+): count (\d+), levels completed (\d+)'
    )
    rows = {}
    for line in log_path.read_text(errors='ignore').splitlines():
        match = pattern.search(line)
        if not match:
            continue
        game_id, action_name, count_raw, levels_raw = match.groups()
        count = int(count_raw)
        levels = int(levels_raw)
        row = rows.setdefault(
            game_id,
            {'max_levels': 0, 'last_count': 0, 'last_levels': 0, 'last_action': ''},
        )
        row['max_levels'] = max(row['max_levels'], levels)
        if count >= row['last_count']:
            row.update(
                last_count=count,
                last_levels=levels,
                last_action=action_name,
            )
    print(
        f'LOCAL_VALIDATION games={len(rows)} total_max_levels='
        f'{sum(row["max_levels"] for row in rows.values())}'
    )
    for game_id, row in sorted(
        rows.items(), key=lambda item: (-item[1]['max_levels'], item[0])
    ):
        print(
            f'LOCAL_VALIDATION {game_id} max={row["max_levels"]} '
            f'last={row["last_levels"]} steps={row["last_count"]} '
            f'last_action={row["last_action"]}'
        )

if should_run_gateway:
    agents_wd = prepare_framework()

    # Point the framework at the gateway sidecar.
    with open(agents_wd / '.env', 'w') as f:
        f.write(
            "SCHEME=http\n"
            "HOST=gateway\n"
            "PORT=8001\n"
            f"ARC_API_KEY={ARC_API_KEY}\n"
            "ARC_BASE_URL=http://gateway:8001/\n"
            "OPERATION_MODE=online\n"
            "ENVIRONMENTS_DIR=\n"
            "RECORDINGS_DIR=/kaggle/working/server_recording\n"
            + profile_env_text()
        )

    # Wait briefly with auth, but never hang the notebook for 10 minutes.
    # main.py will print API details if the gateway is still unavailable.
    wait_for_gateway(60 if is_competition_rerun else 10)

    # Run it. The gateway records every action and emits submission.parquet.
    run_env = os.environ.copy()
    run_env.update(PROFILE_ENV)
    run_env.update({
        'MPLBACKEND': 'agg',
        'VLLM_STARTUP_TIMEOUT': '1000',
        'VLLM_USE_FLASHINFER_SAMPLER': '0',
    })
    subprocess.run(
        [sys.executable, 'main.py', '--agent', 'myagent'],
        cwd=agents_wd,
        check=True,
        env=run_env,
    )
elif run_local_validation:
    print('Running local validation mode')
    agents_wd = prepare_framework()
    recordings_dir = Path('/kaggle/working/server_recording')
    recordings_dir.mkdir(parents=True, exist_ok=True)
    env_dir = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
    if not env_dir.exists():
        raise FileNotFoundError(f'Local environment_files not found: {env_dir}')
    local_game_ids = [
        item.strip()
        for item in os.getenv('LOCAL_VALIDATION_GAME_IDS', '').split(',')
        if item.strip()
    ]
    if not local_game_ids:
        # DIAGNOSTIC PATCH (Jul 5, 2026): if no explicit game IDs were given,
        # auto-select the first N (sorted, deterministic) from environment_files
        # instead of silently validating against the entire public suite.
        # Explicit LOCAL_VALIDATION_GAME_IDS always takes precedence over this.
        game_limit_raw = os.getenv('LOCAL_VALIDATION_GAME_LIMIT', '3').strip()
        try:
            game_limit = int(game_limit_raw)
        except ValueError:
            game_limit = 3
        if game_limit > 0:
            available_games = sorted(p.name for p in env_dir.iterdir() if p.is_dir())
            local_game_ids = available_games[:game_limit]
            if local_game_ids:
                print(
                    f'No LOCAL_VALIDATION_GAME_IDS set; auto-selected first '
                    f'{len(local_game_ids)} games: {",".join(local_game_ids)}'
                )
    if local_game_ids:
        subset_dir = Path('/kaggle/working/local_validation_environment_files')
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
        subset_dir.mkdir(parents=True, exist_ok=True)
        missing_games = []
        for game_id in local_game_ids:
            source_dir = env_dir / game_id
            if not source_dir.exists():
                missing_games.append(game_id)
                continue
            shutil.copytree(source_dir, subset_dir / game_id)
        if missing_games:
            raise FileNotFoundError(
                f'Local validation games missing from environment_files: {missing_games}'
            )
        env_dir = subset_dir
        print(f'Local validation subset: {",".join(local_game_ids)}')
    with open(agents_wd / '.env', 'w') as f:
        f.write(
            "SCHEME=http\n"
            "HOST=127.0.0.1\n"
            "PORT=8001\n"
            f"ARC_API_KEY={ARC_API_KEY}\n"
            "ARC_BASE_URL=http://127.0.0.1:8001/\n"
            "OPERATION_MODE=online\n"
            "ENVIRONMENTS_DIR=\n"
            f"RECORDINGS_DIR={recordings_dir}\n"
            + profile_env_text()
        )
    server_log = recordings_dir / 'arc_server.log'
    server_process = subprocess.Popen(
        [
            sys.executable,
            '-c',
            textwrap.dedent(
                f"""
                import os
                from pathlib import Path
                from arc_agi import Arcade, OperationMode

                os.environ['OPERATION_MODE'] = 'OFFLINE'
                os.environ['ENVIRONMENTS_DIR'] = {str(env_dir)!r}
                os.environ['RECORDINGS_DIR'] = {str(recordings_dir)!r}
                Path(os.environ['RECORDINGS_DIR']).mkdir(parents=True, exist_ok=True)
                Arcade(
                    operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.environ['ENVIRONMENTS_DIR'],
                ).listen_and_serve(
                    host='0.0.0.0',
                    port=8001,
                    competition_mode=True,
                    save_all_recordings=True,
                )
                """
            ),
        ],
        stdout=open(server_log, 'w'),
        stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(5)
        env = os.environ.copy()
        env.update(PROFILE_ENV)
        env.update(
            {
                'MPLBACKEND': 'agg',
                'VLLM_STARTUP_TIMEOUT': '1000',
                'VLLM_USE_FLASHINFER_SAMPLER': '0',
                'GAME_TIME_LIMIT_S': os.getenv('LOCAL_VALIDATION_GAME_TIME_LIMIT_S', str(60 * 60)),
            }
        )
        subprocess.run(
            [sys.executable, 'main.py', '--agent', 'myagent'],
            cwd=agents_wd,
            check=True,
            env=env,
        )
    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
    summarize_local_validation(agents_wd / 'logs.log')
else:
    print('Skipping gateway/local validation; set RUN_ARC_LOCAL_VALIDATION=1 for public-suite validation.')


# === Code Cell 7 ===
from pathlib import Path
if not Path('/kaggle/working/submission.parquet').exists():
    # Save-and-run-all (commit) mode: emit a dummy submission so the
    # commit succeeds. The real submission.parquet is produced by the
    # gateway whenever it is reachable.
    import pandas as pd
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
    submission.head()


