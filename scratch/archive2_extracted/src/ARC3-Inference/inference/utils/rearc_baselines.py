from __future__ import annotations

try:
    from re_arc.dsl.precomputed_actions import metadata_baseline_actions
except Exception as exc:  # pragma: no cover - depends on the installed re_arc version.
    metadata_baseline_actions = None
    _METADATA_IMPORT_ERROR = exc
else:
    _METADATA_IMPORT_ERROR = None


def load_re_arc_baseline_actions(game_id: str, *, environments_dir: str | None = None) -> list[int]:
    if metadata_baseline_actions is None:
        raise RuntimeError(
            "Loading baseline_actions requires a re_arc version with metadata_baseline_actions "
            "from the default branch or a newer release."
        ) from _METADATA_IMPORT_ERROR

    try:
        raw_actions = metadata_baseline_actions(game_id, environments_dir=environments_dir)
    except Exception as exc:
        raise RuntimeError(f"{game_id}: failed to load re_arc metadata baseline_actions.") from exc

    baseline_actions = [int(value) for value in raw_actions]
    if not baseline_actions or any(value <= 0 for value in baseline_actions):
        raise RuntimeError(f"{game_id}: re_arc metadata baseline_actions must be a non-empty positive list.")
    return baseline_actions
