from __future__ import annotations

from typing import Any, Callable


def normalize_state_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name).upper()
    return str(value).strip().upper()


def format_decision_debug(
    decision: Any,
    metadata: dict[str, Any],
    *,
    memory_summary_loader: Callable[[], dict[str, Any]] | None = None,
) -> str:
    top_proposals = [
        f"{proposal.action_name}:{proposal.score:.3f}/{proposal.source}"
        for proposal in list(getattr(decision, "proposals", None) or [])[:3]
    ]
    subgoal = f" subgoal={decision.subgoal}" if getattr(decision, "subgoal", None) else ""
    reason = f" reason={decision.reason}" if getattr(decision, "reason", None) else ""
    proposals = f" top=[{', '.join(top_proposals)}]" if top_proposals else ""
    extras = _format_metadata(metadata, memory_summary_loader=memory_summary_loader)
    return (
        f"decision action={decision.action_name} source={decision.source}"
        f" conf={decision.confidence:.3f}{subgoal}{reason}{proposals}{extras}"
    )


def _format_metadata(
    metadata: dict[str, Any],
    *,
    memory_summary_loader: Callable[[], dict[str, Any]] | None,
) -> str:
    extras: list[str] = []
    target = _format_target(metadata)
    if target:
        extras.append(target)
    board_context = _format_board_context(memory_summary_loader)
    if board_context:
        extras.append(board_context)
    executive_debug = dict(metadata.get("executive_debug") or {})
    if executive_debug:
        extras.append(_format_executive_debug(executive_debug))
    score_floor_debug = dict(metadata.get("score_floor_debug") or {})
    if not score_floor_debug:
        return "".join(extras)
    extras.append(_format_score_floor_debug("score_floor", score_floor_debug, include_frames=True))
    active_reopen = dict(score_floor_debug.get("active_reopen") or {})
    if active_reopen:
        extras.append(
            _format_score_floor_debug(
                "score_floor_active",
                active_reopen,
                include_frames=False,
            )
        )
    return "".join(extras)


def _format_target(metadata: dict[str, Any]) -> str:
    target_anchor = metadata.get("target_anchor")
    if isinstance(target_anchor, (list, tuple)) and len(target_anchor) == 2:
        try:
            stage = metadata.get("shape_stage")
            return (
                f" meta[target_anchor={int(target_anchor[0])},{int(target_anchor[1])}"
                f" stage={int(stage) if stage is not None else '-'}]"
            )
        except (TypeError, ValueError):
            pass
    if "x" not in metadata or "y" not in metadata:
        return ""
    try:
        x = int(metadata.get("x"))
        y = int(metadata.get("y"))
    except (TypeError, ValueError):
        return ""
    return f" meta[target={x},{y}]"


def _format_board_context(
    memory_summary_loader: Callable[[], dict[str, Any]] | None,
) -> str:
    if memory_summary_loader is None:
        return ""
    try:
        summary = dict(memory_summary_loader() or {})
    except Exception:
        return ""
    board_context = dict(summary.get("current_board_context") or {})
    board_height = int(board_context.get("board_height", 0) or 0)
    board_width = int(board_context.get("board_width", 0) or 0)
    board_active_cells = int(board_context.get("board_active_cells", 0) or 0)
    board_source_attr = str(board_context.get("board_source_attr", "") or "")
    board_default_probe = dict(board_context.get("board_default_probe") or {})
    if (
        board_height <= 0
        and board_width <= 0
        and board_active_cells <= 0
        and not board_source_attr
        and not board_default_probe
    ):
        return ""
    target = ""
    try:
        if "x" in board_default_probe and "y" in board_default_probe:
            target = f"{int(board_default_probe['x'])},{int(board_default_probe['y'])}"
    except (TypeError, ValueError):
        target = ""
    return (
        " meta[board"
        f" size={board_width}x{board_height}"
        f" active={board_active_cells}"
        f" source={board_source_attr or '-'}"
        f" default_probe={target or '-'}]"
    )


def _format_score_floor_debug(
    label: str,
    payload: dict[str, Any],
    *,
    include_frames: bool,
) -> str:
    candidate_actions = ",".join(
        str(action_name) for action_name in list(payload.get("candidate_actions") or [])
    )
    action_counts = dict(payload.get("action_counts") or {})
    count_pairs = ",".join(
        f"{action_name}:{action_counts[action_name]}"
        for action_name in sorted(action_counts)
    )
    frames = ""
    if include_frames:
        frame_signatures = ",".join(
            str(frame_signature)
            for frame_signature in list(payload.get("frame_signatures") or [])
        )
        frames = f" frames={frame_signatures or '-'}"
    return (
        f" meta[{label}"
        f" reopened={int(bool(payload.get('is_reopened')))}"
        f" matches={int(payload.get('recent_match_count', 0) or 0)}"
        f" candidates={candidate_actions or '-'}"
        f"{frames}"
        f" counts={count_pairs or '-'}]"
    )


def _format_executive_debug(payload: dict[str, Any]) -> str:
    strategy_mode = str(payload.get("strategy_mode", "") or "")
    fallback_trigger = str(payload.get("fallback_trigger", "") or "")
    selected_source_family = str(payload.get("selected_source_family", "") or "")
    breakdown = dict(payload.get("top_score_breakdown") or {})
    breakdown_summary = ",".join(
        f"{key}:{breakdown[key]}"
        for key in ("base_confidence", "source_bias", "recommended_bonus", "trained_prior_bonus", "working_memory_failure_penalty", "probe_exit_penalty", "dynamics_adjustment")
        if key in breakdown
    )
    return (
        " meta[exec"
        f" strategy={strategy_mode or '-'}"
        f" source_family={selected_source_family or '-'}"
        f" fallback={fallback_trigger or '-'}"
        f" breakdown={breakdown_summary or '-'}]"
    )
