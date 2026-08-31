"""Cross-clone replay session + scout scheduler — the headline transfer knob.

This graft turns the 110 competition runs (25 public games cloned
round-robin) into a cooperative pool: the first clone of a family to clear a
level *publishes* its pruned per-level action sequence to the process-global
:mod:`taaf_grafts.family_store`, and later clones of the SAME family *replay*
that sequence mechanically at the two clean, verifiable moments — play start
(the store's ``best_prefix``) and immediately after an auto-RESET restarts a
level (``segment_for_level``). A sibling that would otherwise cold-start can
skip straight to the deepest already-solved level for free.

WHY every path degrades to stock (the design's provability guarantee):

- The only clone signal is the initial-frame fingerprint. A miss (``None``)
  makes every store call a no-op, so a non-clone hidden set turns the entire
  stack into a measured no-op (gate phase D).
- Every replayed action is re-verified against the recorded state before the
  next one. Levels are checked always; the visible grid is checked only while
  the family verifies in ``'grid'`` mode. The FIRST divergence stops the
  replay and falls through to the normal LLM loop — the already-completed
  levels are kept, and the cost of a wrong guess is one wasted action.
- A consistent-levels grid divergence (cosmetic-RNG / lf52 class: the board
  animates but the logic is identical) flips the family to ``'level'`` mode
  so later siblings verify by level count only and still transfer.
- The preamble, every store interaction, adoption, and win publishing are
  each wrapped so any fault leaves the session running exactly as banking
  would. The scout scheduler is pure and total: any fault returns the
  original game order, so scheduling is at worst stock order.

TWO BINDING CONTRACTS honored here:

1. TIMING — no per-game budget is ever extended. The preamble runs before the
   normal loop and replay checks ``should_stop``/``stop_event`` between every
   action; it only ever *shrinks* wall clock by skipping LLM turns, never
   grows it.
2. PUBLISH TRIGGER — segments publish on every observed level advance and the
   full win publishes on a RECORDED win (``run.state == 'won'`` committed by
   ``game.execute_action``), never on banking's second-play replay success.

Observability is a single stdout line per event (``[transfer] ...``) for the
Kaggle commit-log gate. Transfer deliberately never writes ``run.solver_note``
or any other compared results field: the store-off / lazy-sibling
zero-regression proofs require every scored field to stay byte-identical to
stock, so stdout is the only externally visible signal.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import arcengine

from inference.framework.solver import _grid_from_state

import taaf.game

from taaf_grafts import family_store
from taaf_grafts.banking_solver import (
    BankingHarnessSolver,
    BankingPlanError,
    _BankingGameSession,
    prune_winning_trace,
)
from taaf_grafts.family_store import fingerprint, get_store
from taaf_grafts.trace_utils import prune_level_segments

Grid = tuple[tuple[int, ...], ...]

# Sentinel family for fingerprint misses in the scheduler: each such game is
# its own singleton group (it can be a clone of nothing), so it must never
# share a bucket with another miss.
_NO_FAMILY = object()


def _round_robin_by_family(
    games: list[Any], key_of: Callable[[Any], Any]
) -> list[Any]:
    """Round-robin ``games`` so each family's scout dispatches first.

    Games are grouped by ``key_of`` (a ``None`` key means "no family" — each
    such game is its own singleton group), preserving first-seen order across
    families and singletons alike. The result emits every group's first
    member, then every group's second member, and so on: one scout per family
    before any sibling. The output is asserted to be an exact permutation of
    ``games`` (same objects, none dropped, none duplicated); a caller that
    catches the assertion falls back to the original order.
    """
    groups: dict[Any, list[Any]] = {}
    order: list[Any] = []
    misses = 0
    for game in games:
        key = key_of(game)
        if key is None:
            key = (_NO_FAMILY, misses)
            misses += 1
        bucket = groups.get(key)
        if bucket is None:
            bucket = []
            groups[key] = bucket
            order.append(key)
        bucket.append(game)

    ordered_groups = [groups[key] for key in order]
    reordered: list[Any] = []
    depth = 0
    while True:
        added = False
        for bucket in ordered_groups:
            if depth < len(bucket):
                reordered.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1

    assert len(reordered) == len(games), "scout reorder changed the game count"
    assert Counter(map(id, reordered)) == Counter(
        map(id, games)
    ), "scout reorder is not an exact permutation"
    return reordered


@dataclass
class _TransferGameSession(_BankingGameSession):
    """Banking session that also transfers per-level solutions across clones.

    Adds four seams on top of banking, each a strict superset that degrades
    to the banking behavior on any fault: a play-start replay preamble, a
    post-auto-RESET adoption attempt, per-level-advance publishing, and
    win publishing before the banking finish path.
    """

    _fp_key: bytes | None = field(default=None, init=False, repr=False)
    _fp_ready: bool = field(default=False, init=False, repr=False)
    _replayed_actions: int = field(default=0, init=False, repr=False)
    _published_levels: set[int] = field(default_factory=set, init=False, repr=False)
    _adopted_levels: set[int] = field(default_factory=set, init=False, repr=False)
    _adoption_disabled: bool = field(default=False, init=False, repr=False)
    _win_published: bool = field(default=False, init=False, repr=False)

    # -- knobs / gates -----------------------------------------------------

    def _transfer_enabled(self) -> bool:
        return bool(getattr(self.solver, "transfer_enabled", True))

    def _store_active(self) -> bool:
        """Transfer produces an externally visible side effect (a publish, a
        replay, a stdout line) only when both the solver knob and the store's
        global kill switch are on. When either is off the session is a pure
        banking session — the store-off zero-regression proof."""
        return self._transfer_enabled() and family_store.ENABLED

    def _family_key(self) -> bytes | None:
        """Fingerprint the game once, from its pristine initial state, and
        cache it. The first call happens in the preamble before any action,
        so the cached key reflects the level-0 board; every later publish /
        adoption / win reuses it (the initial state is gone by then)."""
        if not self._fp_ready:
            try:
                self._fp_key = fingerprint(self.game)
            except Exception:
                self._fp_key = None
            self._fp_ready = True
        return self._fp_key

    # -- play-start replay preamble ---------------------------------------

    def play(self) -> None:
        try:
            self._transfer_preamble()
        except Exception:  # noqa: BLE001 — a broken preamble must not block play
            pass
        super().play()

    def _transfer_preamble(self) -> None:
        if not self._store_active():
            return
        key = self._family_key()
        prefix = get_store().best_prefix(key)
        if not prefix:
            return
        # Idempotent mirror of stock play()'s pre-loop init (solver.py:263-274)
        # so the replayed actions land on a real transcript + seeded history +
        # written runtime_state; super().play() repeats these harmlessly
        # (seed is a no-op once seeded, the rest overwrite), keeping the
        # model-visible state 1:1 with a normal run.
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.touch(exist_ok=True)
        self.seed_initial_history()
        self.write_runtime_state()

        verify_mode = get_store().get_verify_mode(key) or "grid"
        replayed_levels = 0
        for segment in prefix:
            if not self._replay_segment(segment, key, verify_mode):
                break
            replayed_levels += 1
        if replayed_levels:
            self._note(
                f"replayed levels 0..{replayed_levels - 1} "
                f"({self._replayed_actions} actions)"
            )

    # -- post-auto-RESET adoption -----------------------------------------

    def _execute_auto_reset(self) -> None:
        super()._execute_auto_reset()
        try:
            self._maybe_adopt_segment()
        except Exception:  # noqa: BLE001 — adoption must not break the run
            self._adoption_disabled = True

    def _maybe_adopt_segment(self) -> None:
        if not self._store_active() or self._adoption_disabled:
            return
        if not getattr(self.solver, "transfer_adopt_on_reset", True):
            return
        key = self._family_key()
        if key is None:
            return
        level = int(self.game.current_state.levels_completed)
        if level in self._adopted_levels:
            return  # at most one adoption attempt per level index
        self._adopted_levels.add(level)
        segment = get_store().segment_for_level(key, level)
        if segment is None:
            return
        # Gate on the fresh post-RESET board matching the stored level-start
        # board before sending ANY action: adoption only fires on a live
        # state the segment provably applies to.
        if segment.level_start_grid != _grid_from_state(self.game.current_state):
            return
        verify_mode = get_store().get_verify_mode(key) or "grid"
        if self._replay_segment(segment, key, verify_mode):
            self._note(f"adopted level {level} ({len(segment.steps)} actions)")
        else:
            self._adoption_disabled = True

    # -- shared per-step replay -------------------------------------------

    def _replay_segment(self, segment: Any, key: bytes | None, verify_mode: str) -> bool:
        """Replay one stored level segment from the current board via the
        normal ``_execute_action`` path, verifying each step. Returns True iff
        every step verified and the level advanced exactly as recorded. On any
        stop / cap / engine refusal / divergence it notes the reason (flipping
        the family to ``'level'`` verification on a consistent-levels grid
        divergence) and returns False. Never raises."""
        cap = getattr(self.solver, "transfer_max_prefix_actions", None)
        total = len(segment.steps)
        for index, step in enumerate(segment.steps, start=1):
            if self.should_stop() or self.stop_event.is_set():
                self._note(f"aborted at step {index}/{total}: stop requested")
                return False
            if cap is not None and self._replayed_actions >= int(cap):
                self._note(f"aborted at step {index}/{total}: prefix cap {cap}")
                return False
            action = arcengine.ActionInput(
                id=step.action_id, data=dict(step.action_data)
            )
            try:
                self._execute_action(
                    action, batch_index=0, batch_size=1, generated_tokens=0
                )
            except Exception as exc:  # noqa: BLE001 — an engine refusal aborts free
                self._note(
                    f"aborted at step {index}/{total}: {type(exc).__name__}: {exc}"
                )
                return False
            self._replayed_actions += 1
            state = self.game.current_state
            # GAME_OVER is unconditional divergence: the recorded trajectory
            # never died at this step, so a death here means the live board
            # left the recording. Abort AT the death rather than blind-firing
            # the rest of the segment onto the scored play — critical in
            # 'level' mode, where the level check below stays satisfied through
            # a death (levels_completed is unchanged) and would otherwise let a
            # misclassified stochastic family commit a whole wasted segment.
            if state.raw.state == arcengine.GameState.GAME_OVER:
                self._note(f"aborted at step {index}/{total}: GAME_OVER")
                return False
            if int(state.levels_completed) != int(step.post_levels):
                self._note(f"aborted at step {index}/{total}: level divergence")
                return False
            if verify_mode == "grid" and _grid_from_state(state) != step.post_grid:
                get_store().note_grid_divergence(key)
                self._note(
                    f"aborted at step {index}/{total}: grid divergence "
                    "(verify_mode -> level)"
                )
                return False
        return True

    # -- publish-as-you-go -------------------------------------------------

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
    ) -> dict[str, Any]:
        try:
            before = int(self.game.current_state.levels_completed)
        except Exception:  # noqa: BLE001 — degrade to no publish, still act
            before = None
        payload = super()._execute_action(
            action,
            batch_index=batch_index,
            batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )
        if before is not None:
            try:
                if int(self.game.current_state.levels_completed) > before:
                    self._publish_completed_levels()
            except Exception:  # noqa: BLE001 — publishing must not touch play
                pass
        return payload

    def _publish_completed_levels(self) -> None:
        if not self._store_active():
            return
        key = self._family_key()
        if key is None or not self.history_entries:
            return
        segments = prune_level_segments(
            self._trace, self.history_entries[0].frame.grid
        )
        store = get_store()
        for level in sorted(segments):
            if level in self._published_levels:
                continue  # a completed level's segment is stable — publish once
            segment = segments[level]
            store.publish_segment(key, level, segment)
            self._published_levels.add(level)
            self._note(f"published level {level} ({len(segment.steps)} actions)")

    # -- win publishing (strictly before the banking finish path) ---------

    def _finish_if_needed(self) -> None:
        try:
            self._maybe_publish_win()
        except Exception:  # noqa: BLE001 — publishing must never block completion
            pass
        super()._finish_if_needed()

    def _maybe_publish_win(self) -> None:
        if self._win_published or not self._store_active():
            return
        run = self.game.game_run
        # Publish strictly AFTER a fully recorded WIN and strictly BEFORE
        # finish_game sets final_score (contract 2: recorded win, not replay).
        if run is None or run.state != "won" or run.final_score is not None:
            return
        if not self.history_entries or len(self._trace) != len(run.history):
            return
        key = self._family_key()
        if key is None:
            return
        try:
            plan = prune_winning_trace(
                self._trace,
                self.history_entries[0].frame.grid,
                int(self.game.number_of_levels),
            )
        except BankingPlanError:
            return
        self._win_published = True
        get_store().publish_win(key, plan)
        self._note(f"published win ({len(plan)} actions)")

    # -- observability -----------------------------------------------------

    def _note(self, message: str) -> None:
        """Emit one ``[transfer] ...`` line to stdout for the Kaggle commit-log
        gate. Deliberately does NOT touch ``run.solver_note`` or any compared
        results field — the transfer stack must stay byte-identical to stock in
        every scored field (the store-off and lazy-sibling zero-regression
        proofs), so stdout is its only externally visible signal."""
        try:
            sys.stdout.write(f"[transfer] {message}\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 — logging must never break a run
            pass


@dataclass
class TransferHarnessSolver(BankingHarnessSolver):
    """Banking solver plus cross-clone transfer and the scout scheduler.

    The session graft rides the ``session_class`` seam (``SessionSeamMixin``,
    inherited through :class:`BankingHarnessSolver`); the scheduler is a single
    ``_run_games`` override. Every transfer knob is getattr-defaulted so the
    session code is safe even when attached to a plainer solver, and all knobs
    also exist as dataclass fields so ``from_solver`` and composite overrides
    can set them. ``transfer`` implies ``banking`` (the parent's banking is
    always on).

    NOTE ON BASES: the design text writes ``(SessionSeamMixin,
    BankingHarnessSolver)``, but ``BankingHarnessSolver`` already lists
    ``SessionSeamMixin`` first, so that tuple is a C3-MRO conflict. Inheriting
    ``BankingHarnessSolver`` alone yields the intended MRO
    (``... -> SessionSeamMixin -> HarnessSolver -> ...``) and the same
    ``_play_one`` seam, which constructs ``session_class``.
    """

    session_class = _TransferGameSession
    label: str = "TransferHarnessSolver"
    transfer_enabled: bool = True
    transfer_adopt_on_reset: bool = True
    transfer_max_prefix_actions: int | None = None

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        await super()._run_games(self._scout_reorder(games))

    def _scout_reorder(
        self, games: list[taaf.game.Game]
    ) -> list[taaf.game.Game]:
        """Round-robin the game list so each clone family's scout dispatches
        first (siblings then find its published segments). Pure and total: any
        fault returns the original list, so the schedule degrades to stock
        order. Score-neutral — per-game clocks start only after the dispatch
        semaphore is acquired, so ordering never changes any per-game budget."""
        try:
            if not getattr(self, "transfer_enabled", True):
                return games
            return _round_robin_by_family(list(games), fingerprint)
        except Exception:  # noqa: BLE001 — any scheduling fault -> stock order
            return games
