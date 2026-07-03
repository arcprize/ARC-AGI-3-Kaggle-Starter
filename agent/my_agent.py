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
from agi_core.llm.llama_cpp_backend import LlamaCppBackend
from agi_core.memory import EpisodicMemory
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
            trained_priors_db_path=config.trained_priors_db_path,
            project_root=config.project_root,
            reset_state_names=config.reset_state_names,
            win_state_name=config.win_state_name,
            enabled=config.hyperon_enabled,
            skip_probe=config.hyperon_skip_probe,
            probe_timeout_seconds=config.hyperon_probe_timeout_seconds,
        )
        llm_backend = LlamaCppBackend(config.effective_model_path())
        executive = Executive(
            config=config,
            reactive=ReactiveSpecialist(),
            symbolic=SymbolicSpecialist(),
            llm=LLMSpecialist(),
            critic=ProposalCritic(),
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
        if latest_frame.state is GameState.WIN:
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
        print(
            f"decision action={decision.action_name} source={decision.source}"
            f" conf={decision.confidence:.3f}{subgoal}{reason}{proposals}"
        )
