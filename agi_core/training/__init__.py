"""Training data pipeline for ARC-AGI-3.

Processes human-demo recordings into SQLite databases with action priors,
episodic memory records, abstract memories, and board-policy parameters.
"""

from .demo_zip import (
    RecordingMeta,
    TrainingStep,
    build_action_prior_artifact,
    compute_frame_signature,
    default_worker_count,
    iter_recordings,
    iter_steps,
    normalize_action_id,
    normalize_action_name,
    runtime_frame_signature,
)

__all__ = [
    "RecordingMeta",
    "TrainingStep",
    "build_action_prior_artifact",
    "compute_frame_signature",
    "default_worker_count",
    "iter_recordings",
    "iter_steps",
    "normalize_action_id",
    "normalize_action_name",
    "runtime_frame_signature",
]
