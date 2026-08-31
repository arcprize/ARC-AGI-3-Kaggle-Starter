"""Hard no-op guard (Experiment 2, Kandidat A: Known-Noop-Guard).

Experiment 1 (K1 auto-memory, reverted) only *mentioned* known no-ops in the
injected context -- the model was free to ignore that and repeat them anyway
(~12% no-op repeats remained). This module instead lets the harness actively
block re-executing an action already proven to have no effect in the exact
same board state, before it ever reaches the environment.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


def board_signature(grid: Any) -> str:
    """Stable, compact signature for a 2D integer grid."""
    rows = tuple(tuple(int(cell) for cell in row) for row in grid or ())
    digest = hashlib.blake2b(repr(rows).encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def normalize_action_signature(value: Any) -> str:
    return " ".join(str(value or "").split())


class NoopGuard:
    """Tracks ``(level, board_before_sig, action_sig)`` combos known to be no-ops.

    Bounded per level (oldest states/actions evicted first) so long runs can't
    grow this without limit.
    """

    def __init__(self, *, max_states_per_level: int = 512, max_actions_per_state: int = 16) -> None:
        self._max_states_per_level = max(1, int(max_states_per_level))
        self._max_actions_per_state = max(1, int(max_actions_per_state))
        self._levels: "OrderedDict[int, OrderedDict[str, OrderedDict[str, None]]]" = OrderedDict()

    def observe(
        self,
        *,
        level: Any,
        board_before_sig: str,
        action_sig: str,
        board_changed: bool,
        animated: bool = False,
    ) -> None:
        """Record one executed action and whether it had any effect.

        ``animated`` means the environment returned more than one frame for
        this action. Such an action is never a no-op, even when the visible
        board ends up identical to before: in a type-1 animation (ft09, sb26)
        the first and last frame match by design and the entire signal -- a
        rejected click, a consumed attempt -- lives in the frames in between.
        Recording those as no-ops made the guard hard-block actions that had
        clearly worked, on exactly the games with the most animations.
        """
        try:
            level_num = int(level)
        except (TypeError, ValueError):
            return
        action_sig = normalize_action_signature(action_sig)
        if not action_sig:
            return
        states = self._levels.setdefault(level_num, OrderedDict())

        if board_changed or animated:
            # Evidence now contradicts a previously recorded no-op for this
            # exact state+action: drop the stale entry so we never block a
            # combo that turned out to have an effect after all.
            entry = states.get(board_before_sig)
            if entry is not None and action_sig in entry:
                del entry[action_sig]
                if not entry:
                    del states[board_before_sig]
            return

        entry = states.get(board_before_sig)
        if entry is None:
            entry = OrderedDict()
            states[board_before_sig] = entry
            while len(states) > self._max_states_per_level:
                states.popitem(last=False)
        entry[action_sig] = None
        while len(entry) > self._max_actions_per_state:
            entry.popitem(last=False)

    def is_known_noop(self, level: Any, board_sig: str, action_sig: str) -> bool:
        """True if ``action_sig`` is on record as a no-op in this exact state."""
        try:
            level_num = int(level)
        except (TypeError, ValueError):
            return False
        states = self._levels.get(level_num)
        if not states:
            return False
        entry = states.get(board_sig)
        if not entry:
            return False
        return normalize_action_signature(action_sig) in entry
