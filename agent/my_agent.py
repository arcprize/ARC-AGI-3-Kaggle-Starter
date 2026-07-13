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
from agi_core.agent_debug import format_decision_debug, normalize_state_name
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


def _resolve_parallel_thread_workers(config: RuntimeConfig) -> int:
    try:
        planner_workers = int(getattr(config, "planner_workers", 0) or 0)
    except Exception:
        planner_workers = 0
    try:
        multi_agent_workers = int(getattr(config, "multi_agent_workers", 0) or 0)
    except Exception:
        multi_agent_workers = 0
    if not getattr(config, "multi_agent_enabled", False):
        return planner_workers
    return max(planner_workers, multi_agent_workers)


class MyAgent(Agent):
    """ARC-facing wrapper around the reusable executive runtime."""

    MAX_ACTIONS = 300
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
            memory_db_path=config.gameplay_memory_db_path(),
            archive_memory_db_path=config.gameplay_archive_memory_db_path(),
            trained_priors_db_path=config.gameplay_trained_priors_db_path(),
            project_root=config.project_root,
            reset_state_names=config.reset_state_names,
            win_state_name=config.win_state_name,
            enabled=config.hyperon_enabled,
            skip_probe=config.hyperon_skip_probe,
            probe_timeout_seconds=config.hyperon_probe_timeout_seconds,
        )
        llm_backend = build_llm_backend(config) if config.llm_enabled else None
        from agi_core.orch_or.parallel import ProcessPoolManager, ThreadPoolManager, sync_pool_worker_counts
        sync_pool_worker_counts(
            config.planner_workers,
            thread_workers=_resolve_parallel_thread_workers(config),
        )
        # Pre-warm SQLite caches only when live gameplay explicitly allows disk I/O.
        if config.gameplay_disk_io_enabled:
            from agi_core.storage.memory_cache import get_cache
            for _db in (
                config.gameplay_memory_db_path(),
                config.gameplay_archive_memory_db_path(),
                config.gameplay_trained_priors_db_path(),
            ):
                if _db:
                    get_cache(_db)
        # Pre-warm both pools so worker threads/processes exist before first frame.
        ThreadPoolManager.executor()
        try:
            ProcessPoolManager.executor()
        except Exception:
            pass
        # Initialise CUDA if enabled
        if config.cuda_enabled:
            from agi_core.cuda import init_cuda
            init_cuda(config.cuda_device)

        # Human-like cognition modules
        from agi_core.cognition import (
            AttentionFocus, ChunkLibrary, ConsolidatedMemory,
            GoalStack, MetaCognition, WorkingMemory,
        )
        self._attention = AttentionFocus()
        self._working_memory = WorkingMemory()
        self._goal_stack = GoalStack()
        self._episodic_memory = ConsolidatedMemory()
        self._meta_cognition = MetaCognition()
        self._chunk_library = ChunkLibrary()
        # Also wire into hyperon bridge for rule processing
        hyperon.meta_cognition = self._meta_cognition  # type: ignore[attr-defined]

        base_executive = Executive(
            config=config,
            reactive=ReactiveSpecialist(),
            symbolic=SymbolicSpecialist(),
            llm=LLMSpecialist() if config.llm_enabled else None,
            critic=ProposalCritic(),
            attention=self._attention,
            working_memory=self._working_memory,
            goal_stack=self._goal_stack,
            episodic_memory=self._episodic_memory,
            meta_cognition=self._meta_cognition,
            chunk_library=self._chunk_library,
        )
        if config.multi_agent_enabled:
            from agi_core.multi_agent import AgentSupervisor as _Sup
            from agi_core.multi_agent.agents import (
                ExplorationAgent, InteractionAgent,
                MovementAgent, PuzzleAgent,
            )
            _agents = [MovementAgent(), ExplorationAgent(), InteractionAgent(), PuzzleAgent()]
            executive = _Sup(executive=base_executive, config=config, agents=_agents)
        else:
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
        if normalize_state_name(latest_frame.state) == self.runtime.config.win_state_name:
            self.runtime.bind_game(self.game_id)
            self.runtime.note_terminal_observation(frames, latest_frame)
            # Record episode in consolidated memory / extract chunks
            em = getattr(self, '_episodic_memory', None)
            cl = getattr(self, '_chunk_library', None)
            if em is not None:
                actions = [getattr(f, 'action', '?') for f in frames[-50:]]
                board = getattr(latest_frame, 'board', None) or []
                goal_count = len(getattr(latest_frame, 'goal_positions', []) or [])
                player_start = getattr(latest_frame, 'player_position', (0, 0)) or (0, 0)
                em.record(
                    dynamics=getattr(self, '_last_dynamics', "unknown"),
                    action_sequence=actions,
                    outcome="win",
                    goal_count=goal_count,
                    player_start=player_start,
                    board_hash=hash(str(board)[-256:]),
                )
                em.consolidate()
            if cl is not None and em is not None:
                last = em.last_episode()
                if last is not None:
                    cl.extract(last.action_sequence, last.dynamics, last.goal_count, "win")
            # Reset multi-agent cognition for next game
            if hasattr(self.runtime.executive, 'reset_agents'):
                self.runtime.executive.reset_agents()
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
        print(
            format_decision_debug(
                decision,
                getattr(decision, "metadata", {}) or {},
                memory_summary_loader=self._debug_memory_summary,
            ),
            flush=True,
        )

    def _debug_memory_summary(self) -> dict[str, Any]:
        summary_fn = getattr(self.runtime.memory, "summarize_for_hyperon", None)
        if summary_fn is None:
            return {}
        return dict(summary_fn() or {})
