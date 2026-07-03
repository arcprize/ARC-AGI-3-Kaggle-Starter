from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXTERNAL_LLM_MODEL_PATH = Path(
    r"E:\.models\gemma-4-transformers-gemma-4-e2b-it-v1"
)


def _resolve_path(root: Path, value: str | Path | None, default_relative: str) -> Path:
    if value is None:
        return root / default_relative
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


@dataclass(slots=True)
class RuntimeConfig:
    project_root: Path
    hyperon_rules_path: Path
    hyperon_learning_path: Path
    trained_priors_db_path: Path
    hyperon_wheel_dir: Path
    extra_wheel_dir: Path
    hyperon_enabled: bool
    hyperon_skip_probe: bool
    hyperon_probe_timeout_seconds: float
    llm_model_dir: Path
    llm_model_path: Path
    llm_max_calls_per_episode: int = 2
    llm_max_tokens: int = 192
    llm_temperature: float = 0.1
    symbolic_confidence_gate: float = 0.70
    deterministic_fallback_action: str = "ACTION1"
    reset_state_names: tuple[str, ...] = ("NOT_PLAYED", "GAME_OVER")
    win_state_name: str = "WIN"
    default_legal_actions: tuple[str, ...] = (
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
        "ACTION5",
        "ACTION6",
    )
    source_bias: dict[str, float] = field(
        default_factory=lambda: {
            "reactive": 0.25,
            "symbolic": 0.15,
            "llm": 0.05,
        }
    )

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "RuntimeConfig":
        root = project_root or Path(__file__).resolve().parents[1]
        env = os.environ
        llm_model_dir = _resolve_path(root, env.get("AGI_LLM_MODEL_DIR"), "models")
        llm_model_path = _resolve_path(
            root,
            env.get("AGI_LLM_MODEL_PATH"),
            "models/executive.gguf",
        )
        return cls(
            project_root=root,
            hyperon_rules_path=_resolve_path(
                root,
                env.get("AGI_HYPERON_RULES_PATH"),
                "agi_core/knowledge/rules/arc.metta",
            ),
            hyperon_learning_path=_resolve_path(
                root,
                env.get("AGI_HYPERON_LEARNING_PATH"),
                "agi_core/knowledge/rules/arc_learned.json",
            ),
            trained_priors_db_path=_resolve_path(
                root,
                env.get("AGI_TRAINED_PRIORS_DB_PATH"),
                "training_artifacts/arc_demo_action_priors.sqlite3",
            ),
            hyperon_wheel_dir=_resolve_path(
                root,
                env.get("AGI_HYPERON_WHEEL_DIR"),
                "third_party/hyperon",
            ),
            extra_wheel_dir=_resolve_path(
                root,
                env.get("AGI_EXTRA_WHEEL_DIR"),
                "third_party/wheels",
            ),
            hyperon_enabled=env.get("AGI_HYPERON_ENABLE", "1").strip().lower()
            not in {"0", "false", "no", "off"},
            hyperon_skip_probe=env.get("AGI_HYPERON_SKIP_PROBE", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            hyperon_probe_timeout_seconds=float(
                env.get("AGI_HYPERON_PROBE_TIMEOUT", "5.0")
            ),
            llm_model_dir=llm_model_dir,
            llm_model_path=llm_model_path,
            llm_max_calls_per_episode=int(env.get("AGI_LLM_MAX_CALLS", "2")),
            llm_max_tokens=int(env.get("AGI_LLM_MAX_TOKENS", "192")),
            llm_temperature=float(env.get("AGI_LLM_TEMPERATURE", "0.1")),
            symbolic_confidence_gate=float(
                env.get("AGI_SYMBOLIC_GATE", "0.70")
            ),
            deterministic_fallback_action=env.get(
                "AGI_FALLBACK_ACTION",
                "ACTION1",
            ).upper(),
        )

    def effective_model_path(self) -> Path:
        if self.llm_model_path.exists():
            return self.llm_model_path
        if DEFAULT_EXTERNAL_LLM_MODEL_PATH.exists():
            return DEFAULT_EXTERNAL_LLM_MODEL_PATH
        matches = sorted(self.llm_model_dir.glob("*.gguf"))
        if matches:
            return matches[0]
        for child in sorted(self.llm_model_dir.iterdir()) if self.llm_model_dir.exists() else []:
            if child.is_dir() and (child / "config.json").exists():
                return child
        return self.llm_model_path
