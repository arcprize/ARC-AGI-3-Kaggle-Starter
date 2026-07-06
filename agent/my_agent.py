"""Thin ARC wrapper around the reusable Hyperon + local-LLM executive core.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so local verification and submission stay in
lock-step:

    [edit my_agent.py] -> [make play-local] -> [make submit]

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import os
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent
from agi_core.adapters.arc import ArcAdapter
from agi_core.config import RuntimeConfig
from agi_core.executive import Executive, ExecutiveRuntime
from agi_core.knowledge.hyperon_bridge import HyperonBridge
from agi_core.llm import build_llm_backend
from agi_core.memory import EpisodicMemory
from agi_core.orch_or import ObjectiveReduction
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.specialists.critic import ProposalCritic
from agi_core.specialists.llm import LLMSpecialist
from agi_core.specialists.reactive import ReactiveSpecialist
from agi_core.specialists.symbolic import SymbolicSpecialist


class MyAgent(Agent):
    """ARC-facing wrapper around the reusable executive runtime."""

    MAX_ACTIONS = 80
    DEBUG_DECISIONS = os.environ.get("AGI_DEBUG_DECISIONS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = RuntimeConfig.from_env()
        adapter = ArcAdapter(
            game_action_cls=GameAction,
            default_game_id=self.game_id,
            deterministic_action=config.deterministic_fallback_action,
            reset_state_names=config.reset_state_names,
        )
        memory = EpisodicMemory(
            win_state_name=config.win_state_name,
            reset_state_names=config.reset_state_names,
        )
        hyperon = HyperonBridge(
            rules_path=config.hyperon_rules_path,
            learning_path=config.hyperon_learning_path,
            memory_db_path=config.memory_db_path,
            archive_memory_db_path=config.archive_memory_db_path,
            trained_priors_db_path=config.trained_priors_db_path,
            project_root=config.project_root,
            reset_state_names=config.reset_state_names,
            win_state_name=config.win_state_name,
            enabled=config.hyperon_enabled,
            skip_probe=config.hyperon_skip_probe,
            probe_timeout_seconds=config.hyperon_probe_timeout_seconds,
        )
        llm_backend = build_llm_backend(config)
        if config.planner_enabled and config.planner_parallel:
            ThreadPoolManager.set_max_workers(config.planner_workers)
        base_executive = Executive(
            config=config,
            reactive=ReactiveSpecialist(),
            symbolic=SymbolicSpecialist(),
            llm=LLMSpecialist(),
            critic=ProposalCritic(),
        )
        executive = (
            ObjectiveReduction(executive=base_executive, config=config)
            if config.planner_enabled
            else base_executive
        )
        self.runtime = ExecutiveRuntime(
            adapter=adapter,
            executive=executive,
            memory=memory,
            hyperon=hyperon,
            llm_backend=llm_backend,
            config=config,
        )
        self.runtime.bind_game(self.game_id)

    @property
    def name(self) -> str:
        return f"{super().name}.executive.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Do not stop on GAME_OVER because reset is a valid recovery action.
        if self._normalize_state_name(latest_frame.state) == self.runtime.config.win_state_name:
            self.runtime.bind_game(self.game_id)
            self.runtime.note_terminal_observation(frames, latest_frame)
            return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self.runtime.bind_game(self.game_id)
        decision = self.runtime.decide(frames, latest_frame)
        if self.DEBUG_DECISIONS:
            self._log_decision(decision)
        return self.runtime.adapter.to_game_action(decision)

    def _log_decision(self, decision) -> None:
        top_proposals = []
        for proposal in list(decision.proposals or [])[:3]:
            top_proposals.append(
                f"{proposal.action_name}:{proposal.score:.3f}/{proposal.source}"
            )
        subgoal = f" subgoal={decision.subgoal}" if decision.subgoal else ""
        reason = f" reason={decision.reason}" if decision.reason else ""
        proposals = f" top=[{', '.join(top_proposals)}]" if top_proposals else ""
        metadata = self._format_debug_metadata(getattr(decision, "metadata", {}) or {})
        print(
            f"decision action={decision.action_name} source={decision.source}"
            f" conf={decision.confidence:.3f}{subgoal}{reason}{proposals}{metadata}"
            ,
            flush=True,
        )

    def _normalize_state_name(self, value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "name"):
            return str(value.name).upper()
        return str(value).strip().upper()

    def _format_debug_metadata(self, metadata: dict[str, Any]) -> str:
        extras: list[str] = []
        target = self._format_debug_target(metadata)
        if target:
            extras.append(target)
        board_context = self._format_debug_board_context()
        if board_context:
            extras.append(board_context)
        score_floor_debug = dict(metadata.get("score_floor_debug") or {})
        if not score_floor_debug:
            return "".join(extras)
        candidate_actions = ",".join(
            str(action_name) for action_name in list(score_floor_debug.get("candidate_actions") or [])
        )
        frame_signatures = ",".join(
            str(frame_signature) for frame_signature in list(score_floor_debug.get("frame_signatures") or [])
        )
        action_counts = dict(score_floor_debug.get("action_counts") or {})
        count_pairs = ",".join(
            f"{action_name}:{action_counts[action_name]}"
            for action_name in sorted(action_counts)
        )
        extras.append(
            (
            " meta[score_floor"
            f" reopened={int(bool(score_floor_debug.get('is_reopened')))}"
            f" matches={int(score_floor_debug.get('recent_match_count', 0) or 0)}"
            f" candidates={candidate_actions or '-'}"
            f" frames={frame_signatures or '-'}"
            f" counts={count_pairs or '-'}]"
            )
        )
        active = self._format_active_score_floor_debug(score_floor_debug)
        if active:
            extras.append(active)
        return "".join(extras)

    def _format_active_score_floor_debug(self, score_floor_debug: dict[str, Any]) -> str:
        active_reopen = dict(score_floor_debug.get("active_reopen") or {})
        if not active_reopen:
            return ""
        active_candidates = ",".join(
            str(action_name) for action_name in list(active_reopen.get("candidate_actions") or [])
        )
        active_counts = dict(active_reopen.get("action_counts") or {})
        active_count_pairs = ",".join(
            f"{action_name}:{active_counts[action_name]}"
            for action_name in sorted(active_counts)
        )
        return (
            " meta[score_floor_active"
            f" reopened={int(bool(active_reopen.get('is_reopened')))}"
            f" matches={int(active_reopen.get('recent_match_count', 0) or 0)}"
            f" candidates={active_candidates or '-'}"
            f" counts={active_count_pairs or '-'}]"
        )

    def _format_debug_target(self, metadata: dict[str, Any]) -> str:
        if "x" not in metadata or "y" not in metadata:
            return ""
        try:
            x = int(metadata.get("x"))
            y = int(metadata.get("y"))
        except (TypeError, ValueError):
            return ""
        return f" meta[target={x},{y}]"

    def _format_debug_board_context(self) -> str:
        summary_fn = getattr(self.runtime.memory, "summarize_for_hyperon", None)
        if summary_fn is None:
            return ""
        try:
            summary = dict(summary_fn() or {})
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
