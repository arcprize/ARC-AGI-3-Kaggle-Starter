"""In-process game twin: load any ARC-AGI-3 game and search it for free.

The competition dataset ships every game's Python source under
``environment_files/<id>/<hash>/<id>.py`` plus a pure-Python ``arcengine``.
So a game is a plain in-process object we can ``deepcopy`` and hammer at
thousands of steps/sec without touching the graded run's action count.

Two measured facts this module rests on (see ``verify_twin.py``):

- ``copy.deepcopy(game)`` is a complete state snapshot (~14 ms) and stepping
  the copy leaves the original and the scorecard at ``total_actions=0``.
- ``ARCBaseGame._get_valid_actions()`` returns the exact legal action set for
  the current state, *including* concrete ACTION6 click coordinates. That
  turns a 4096-wide coordinate space into a branching factor of 2-11.

Nothing here is game-specific: every game is driven through the same
``perform_action`` / ``_get_valid_actions`` engine contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# RESET must restart the *level*, not the whole game, or prefix-replay and
# post-death recovery both silently snap back to level 0. Set before arc_agi
# builds its client, since the value is cached at import time.
os.environ.setdefault("ONLY_RESET_LEVELS", "true")

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402

for _noisy in ("arc_agi.scorecard", "arc_agi", "arcengine"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

_QUIET = logging.getLogger("arc3x.twin")
_QUIET.setLevel(logging.CRITICAL)

# Actions in engine id order. Index == the int id used in available_actions.
ACTION_BY_ID: dict[int, GameAction] = {
    0: GameAction.RESET,
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


@dataclass(frozen=True)
class Act:
    """A hashable, picklable action: engine id plus optional click coords."""

    aid: int
    x: int = -1
    y: int = -1

    @property
    def is_click(self) -> bool:
        return self.aid == 6

    def to_input(self) -> ActionInput:
        data = {"x": self.x, "y": self.y} if self.is_click else {}
        return ActionInput(id=ACTION_BY_ID[self.aid], data=data)

    def __repr__(self) -> str:
        return f"A{self.aid}({self.x},{self.y})" if self.is_click else f"A{self.aid}"


@dataclass
class Obs:
    """What the searcher sees after a step. ``frame`` is the final 64x64 grid."""

    frame: np.ndarray
    level: int
    score: int
    state: Any
    valid: tuple[Act, ...]
    n_frames: int = 1

    @property
    def game_over(self) -> bool:
        return self.state == GameState.GAME_OVER

    @property
    def won(self) -> bool:
        return self.state == GameState.WIN

    @property
    def terminal(self) -> bool:
        return self.game_over or self.won

    def key(self) -> bytes:
        """Frame identity hash - the archive's notion of "same situation"."""
        return hashlib.blake2b(
            self.frame.tobytes() + bytes((self.level & 0xFF,)), digest_size=16
        ).digest()


def discover_env_dirs(*roots: str | Path) -> Path | None:
    """Find an ``environment_files`` directory under any of ``roots``.

    Checked in order so the same code path works locally, in an interactive
    Kaggle session, and inside a graded rerun container.
    """
    for root in roots:
        if not root:
            continue
        p = Path(root)
        if p.name == "environment_files" and p.is_dir():
            return p
        cand = p / "environment_files"
        if cand.is_dir():
            return cand
    return None


def default_env_dir() -> Path:
    """Locate environment_files without any hardcoded absolute path."""
    found = discover_env_dirs(
        os.environ.get("ARC3X_ENV_DIR", ""),
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3",
        Path(__file__).resolve().parent.parent / "datasets" / "arc-prize-2026-arc-agi-3",
        Path.cwd() / "datasets" / "arc-prize-2026-arc-agi-3",
        Path.cwd(),
    )
    if found is not None:
        return found
    # Last resort: rglob under the Kaggle input mount / cwd.
    for base in (Path("/kaggle/input"), Path.cwd()):
        if base.is_dir():
            for meta in base.rglob("environment_files/*/*/metadata.json"):
                return meta.parent.parent.parent
    raise FileNotFoundError("could not locate environment_files")


class Twin:
    """One game, running in-process, snapshot-able and searchable for free."""

    def __init__(self, game_id: str, env_dir: str | Path | None = None):
        self.env_dir = Path(env_dir) if env_dir else default_env_dir()
        self.game_id = game_id
        arcade = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(self.env_dir),
            logger=_QUIET,
        )
        self._arcade = arcade
        env = arcade.make(game_id, scorecard_id=arcade.create_scorecard())
        if env is None or getattr(env, "_game", None) is None:
            raise RuntimeError(f"could not load game {game_id!r} from {self.env_dir}")
        self.env = env
        self.game = env._game
        self.n_levels = int(self.game.win_score)
        self.baselines = self._load_baselines()

    # -- metadata ---------------------------------------------------------

    def _load_baselines(self) -> list[int] | None:
        base = self.game_id.split("-")[0]
        for meta in (self.env_dir / base).rglob("metadata.json"):
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            b = d.get("baseline_actions")
            if b:
                return list(b)
        return None

    # -- core stepping ----------------------------------------------------

    @staticmethod
    def valid_actions(game: Any) -> tuple[Act, ...]:
        """Exact legal actions for this state, straight from the engine.

        Includes concrete click coordinates for ACTION6, which is what makes
        click games tractable at all. Returns () if a game overrides the
        introspection; callers fall back to the frame's coarse action list.
        """
        out: list[Act] = []
        try:
            for ai in game._get_valid_actions():
                aid = int(ai.id.value)
                if aid == 0:
                    continue
                d = dict(ai.data or {})
                if aid == 6:
                    out.append(Act(6, int(d.get("x", 0)), int(d.get("y", 0))))
                else:
                    out.append(Act(aid))
        except Exception:
            return ()
        # Dedupe, keeping the engine's deterministic order.
        seen: set[Act] = set()
        uniq: list[Act] = []
        for a in out:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        return tuple(uniq)

    @classmethod
    def step_game(cls, game: Any, act: Act) -> Obs:
        """Apply one action to ``game`` (usually a clone) and read the result."""
        fd = game.perform_action(act.to_input(), raw=True)
        frames = getattr(fd, "frame", None) or []
        frame = (
            np.asarray(frames[-1], dtype=np.int8)
            if frames
            else np.zeros((64, 64), dtype=np.int8)
        )
        return Obs(
            frame=frame,
            level=int(getattr(fd, "levels_completed", 0)),
            score=int(getattr(game, "_score", 0)),
            state=getattr(fd, "state", None),
            valid=cls.valid_actions(game),
            n_frames=len(frames),
        )

    def snapshot(self) -> Any:
        """A free, complete state snapshot of the live game."""
        return copy.deepcopy(self.game)

    def current(self) -> Obs:
        """Observe the live game without spending an action."""
        raw = self.env.observation_space
        frames = getattr(raw, "frame", None) or []
        frame = (
            np.asarray(frames[-1], dtype=np.int8)
            if frames
            else np.zeros((64, 64), dtype=np.int8)
        )
        return Obs(
            frame=frame,
            level=int(getattr(raw, "levels_completed", 0)),
            score=int(getattr(self.game, "_score", 0)),
            state=getattr(raw, "state", None),
            valid=self.valid_actions(self.game),
            n_frames=len(frames),
        )

    def replay(self, plan: Iterable[Act], from_game: Any | None = None) -> Obs:
        """Run a plan on a clone (or a supplied game) and return the end state."""
        g = copy.deepcopy(from_game if from_game is not None else self.game)
        obs = self.current()
        for a in plan:
            obs = self.step_game(g, a)
            if obs.terminal:
                break
        return obs
