"""Multi-modal perception: combine pixel analysis with action history
to infer game mechanics that aren't visible from a single frame.

The ``ActionEffectLog`` records what happened after each action and uses
pattern-matching rules to infer gravity, sliding, entity switching, and
other mechanics from the sequence of observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ActionEffect:
    """What changed after a single action."""

    action: str
    player_pos_before: tuple[int, int] | None = None
    player_pos_after: tuple[int, int] | None = None
    player_color_before: int = -1
    player_color_after: int = -1
    entity_deltas: list[tuple[int, int, int]] = field(default_factory=list)
    board_changed: bool = False


@dataclass(slots=True)
class InferredMechanics:
    """Mechanics inferred from the action-effect history."""

    gravity_evidence: float = 0.0
    sliding_evidence: float = 0.0
    entity_switch_evidence: float = 0.0
    has_entity_switch: bool = False
    primary_flow: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "gravity_evidence": self.gravity_evidence,
            "sliding_evidence": self.sliding_evidence,
            "entity_switch_evidence": self.entity_switch_evidence,
            "has_entity_switch": self.has_entity_switch,
            "primary_flow": self.primary_flow,
        }


class ActionEffectLog:
    """Records action-effect pairs and infers game mechanics from patterns.

    Usage::

        log = ActionEffectLog()
        # After each action:
        log.record(action, prev_model, curr_model)
        mechanics = log.infer()
    """

    def __init__(self, max_len: int = 16) -> None:
        self._history: list[ActionEffect] = []
        self._max_len = max_len

    def record(
        self,
        action: str,
        player_pos_before: tuple[int, int] | None,
        player_pos_after: tuple[int, int] | None,
        player_color_before: int = -1,
        player_color_after: int = -1,
        entity_tracks_before: list | None = None,
        entity_tracks_after: list | None = None,
        board_changed: bool = False,
    ) -> None:
        """Record the effect of an action."""
        # Compute entity deltas (entity_id, dx, dy) from tracks
        entity_deltas: list[tuple[int, int, int]] = []
        if entity_tracks_before and entity_tracks_after:
            before_map = {t.entity_id: t for t in entity_tracks_before}
            after_map = {t.entity_id: t for t in entity_tracks_after}
            for eid in set(before_map) | set(after_map):
                bt = before_map.get(eid)
                at = after_map.get(eid)
                if bt and at and bt.positions and at.positions:
                    bp = bt.positions[-1]
                    ap = at.positions[-1]
                    dx = ap[0] - bp[0]
                    dy = ap[1] - bp[1]
                    if dx != 0 or dy != 0:
                        entity_deltas.append((eid, dx, dy))

        effect = ActionEffect(
            action=action,
            player_pos_before=player_pos_before,
            player_pos_after=player_pos_after,
            player_color_before=player_color_before,
            player_color_after=player_color_after,
            entity_deltas=entity_deltas,
            board_changed=board_changed,
        )
        self._history.append(effect)
        if len(self._history) > self._max_len:
            self._history.pop(0)

    def reset(self) -> None:
        self._history.clear()

    def infer(self) -> InferredMechanics:
        """Analyse the history for recurring patterns and return inferred mechanics."""
        m = InferredMechanics()

        if not self._history:
            return m

        # Rule 1: Entity switch — ACTION5 + player colour changed
        # (works even with a single entry)
        switch_count = 0
        for effect in self._history:
            if (effect.action == "ACTION5"
                    and effect.player_color_before != effect.player_color_after
                    and effect.player_color_before >= 0
                    and effect.player_color_after >= 0):
                switch_count += 1
        if switch_count >= 1:
            m.entity_switch_evidence = min(1.0, 0.5 + switch_count * 0.15)
            m.has_entity_switch = True

        # Rule 2: Gravity — entity with consistent +vy across multiple
        # non-player-directed frames
        gravity_frames = 0
        sliding_frames = 0
        flow_votes: dict[str, int] = {}

        for effect in self._history[-8:]:  # look at recent history
            if not effect.entity_deltas:
                continue
            for eid, dx, dy in effect.entity_deltas:
                if dy > 0:
                    gravity_frames += 1
                if dx != 0 and dy == 0:
                    sliding_frames += 1
                if abs(dx) >= abs(dy):
                    flow_votes["right" if dx > 0 else "left"] = \
                        flow_votes.get("right" if dx > 0 else "left", 0) + 1
                if abs(dy) > abs(dx):
                    flow_votes["down" if dy > 0 else "up"] = \
                        flow_votes.get("down" if dy > 0 else "up", 0) + 1

        if gravity_frames >= 2:
            m.gravity_evidence = min(1.0, gravity_frames * 0.15)
        if sliding_frames >= 2:
            m.sliding_evidence = min(1.0, sliding_frames * 0.15)

        if flow_votes:
            m.primary_flow = max(flow_votes, key=flow_votes.get)

        return m
