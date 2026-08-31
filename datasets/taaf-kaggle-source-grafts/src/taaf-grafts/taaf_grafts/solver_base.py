"""Session seam for all harness-solver grafts (design module 1).

Stock ``HarnessSolver._play_one`` (solver.py:1208-1238) hard-codes the
session class it constructs (``_HarnessGameSession``). Every session graft
otherwise needs its own copy of that body just to swap that one class,
and each such copy silently rots the day the vendored body changes.

``SessionSeamMixin`` holds the ONE copy: its ``_play_one`` is byte-for-byte
the stock body except that it constructs ``self.session_class`` (a class
attribute, default ``_HarnessGameSession``). A session graft becomes a
one-line ``session_class = <MySession>`` on a ``(SessionSeamMixin, ...)``
subclass. ``STOCK_PLAY_ONE_SRC_HASH`` pins the blake2b of the stock source
region so a companion test fails loudly the day the vendored copy drifts —
the seam's correctness rests on the copy staying identical to stock.
"""

from __future__ import annotations

from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
from inference.framework.solver import (
    _HarnessGameSession,
    _LocalServerRuntime,
)

import taaf.game

# blake2b(inspect.getsource(HarnessSolver._play_one).encode('utf-8')) pinned
# at write time against vendored solver.py:1208-1238. A drift-guard test
# recomputes this and fails loudly if the vendored body ever changes.
STOCK_PLAY_ONE_SRC_HASH = (
    "e325541909010736ce7c1953208f3c8b252ed60b4216987b55f06c2337da08e2b"
    "05574ba4958a45e16fb1e7964620ad83d727629eced41996d591cdf1e84614c"
)


class SessionSeamMixin:
    """Mixin owning the single verbatim copy of stock ``_play_one``.

    Cooperative-MRO mixin for ``HarnessSolver`` subclasses: place it FIRST
    in the bases (``class X(SessionSeamMixin, HarnessSolver)``) so this
    ``_play_one`` wins over the vendored one, then set ``session_class``.
    The body below must stay identical to the stock body except for the
    ``self.session_class`` construction (drift-guarded by the pinned hash).
    """

    session_class: type = _HarnessGameSession

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        try:
            assert game.game_run is not None
            run = game.game_run
            run_stem = self._run_stem(run.game_id, pass_index)
            state_path = self._artifacts_dir() / f"{run_stem}_{RUNTIME_STATE_FILENAME}"
            viewer_data_path = self._artifacts_dir() / f"{run_stem}_viewer_data.json"
            transcript_path = self._transcripts_dir() / f"{run_stem}.txt"
            analysis_relpath = f"solver_analysis/{run_stem}.html"
            analyzer = self._make_analyzer(game, index, local_server)
            session = self.session_class(
                solver=self,
                game=game,
                analyzer=analyzer,
                game_index=index,
                pass_index=pass_index,
                state_path=state_path,
                transcript_path=transcript_path,
                analysis_html_relpath=analysis_relpath,
                stop_event=self._stop_event,
                viewer_data_path=viewer_data_path,
            )
            session.play()
        except Exception as exc:
            self._finish_after_error(game, exc)
