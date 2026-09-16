"""Official-framework adapter for the HypothesisWorld research architecture."""

from __future__ import annotations

import json
import logging
import random
from hashlib import blake2b
from typing import Any

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

from .config import AgentConfig
from .exploration import (
    ActionCandidate,
    CandidateScore,
    ExperimentSelector,
    generate_candidates,
)
from .hypotheses import ActionKey, HypothesisEngine, HypothesisUpdate
from .memory import CompactMemory
from .perception import diff_states, perceive
from .planner import ShortHorizonPlanner
from .state import StateDiff, StructuredState
from .world_model import WorldModel

logger = logging.getLogger("hypothesisworld")


class MyAgent(Agent):
    """Learn by proposing, testing, and falsifying compact world rules."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = AgentConfig.from_env()
        self.MAX_ACTIONS = min(self.MAX_ACTIONS, self.config.max_actions)
        seed_bytes = blake2b(
            f"{self.game_id}:{self.config.random_seed}".encode(), digest_size=8
        ).digest()
        self.rng = random.Random(int.from_bytes(seed_bytes, "big"))
        self.memory = CompactMemory(self.config.max_transitions)
        self.hypotheses = HypothesisEngine(self.config.max_hypotheses_per_action)
        self.world_model = WorldModel()
        self.planner = ShortHorizonPlanner(
            self.world_model,
            self.memory,
            self.config.planning_horizon,
            self.config.planning_confidence,
        )
        self.selector = ExperimentSelector(
            self.config,
            self.memory,
            self.hypotheses,
            self.world_model,
            self.planner,
            self.rng,
        )
        self._previous_state: StructuredState | None = None
        self._previous_levels = 0
        self._pending_action: ActionKey | None = None
        self._previous_diff: StateDiff | None = None

    @property
    def name(self) -> str:
        return f"{super().name}.hypothesisworld-{self.config.ablation.value}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN or self.action_counter >= self.MAX_ACTIONS

    @staticmethod
    def _state_name(frame: FrameData) -> str:
        return getattr(frame.state, "name", str(frame.state).split(".")[-1])

    @staticmethod
    def _legal_action_ids(frame: FrameData) -> list[int]:
        ids: list[int] = []
        for raw in frame.available_actions or []:
            value = getattr(raw, "value", raw)
            try:
                action_id = int(value)
            except (TypeError, ValueError):
                continue
            if action_id not in ids and action_id != 0:
                ids.append(action_id)
        return ids

    def _observe_transition(
        self, current: StructuredState, latest_frame: FrameData
    ) -> HypothesisUpdate | None:
        self.memory.observe_state(current, self.action_counter)
        if self._previous_state is None or self._pending_action is None:
            return None
        progress_delta = int(latest_frame.levels_completed or 0) - self._previous_levels
        diff = diff_states(
            self._previous_state,
            current,
            progress_delta=progress_delta,
            terminal=self._state_name(latest_frame),
        )
        self._previous_diff = diff
        self.memory.record_transition(
            self.action_counter,
            self._previous_state,
            self._pending_action,
            current,
            diff,
        )
        if self.config.uses_world_model:
            self.world_model.update(self._previous_state, self._pending_action, current, diff)
        update = None
        if self.config.uses_hypotheses:
            update = self.hypotheses.update(self._pending_action, diff, self.action_counter)
            self.memory.remember_rejected(update.falsified, self.action_counter)
        logger.info(
            "HYPOTHESISWORLD %s",
            json.dumps(
                {
                    "event": "outcome",
                    "game_id": self.game_id,
                    "ablation": self.config.ablation.value,
                    "seed": self.config.random_seed,
                    "step": self.action_counter,
                    "action": self._pending_action.compact(),
                    "actual": diff.compact(),
                    "strengthened": list(update.strengthened) if update else [],
                    "falsified": list(update.falsified) if update else [],
                    "generated": list(update.generated) if update else [],
                    "progress": latest_frame.levels_completed,
                },
                separators=(",", ":"),
            ),
        )
        return update

    def _reasoning(
        self,
        state: StructuredState,
        ranking: list[CandidateScore],
        selected: ActionCandidate,
    ) -> dict[str, object]:
        top_hypotheses = [
            {
                "id": hypothesis.hypothesis_id,
                "belief": round(hypothesis.effective_belief, 3),
                "description": hypothesis.description,
            }
            for hypothesis in self.hypotheses.top(self.config.log_top_hypotheses)
        ]
        candidates = [
            {
                "action": score.candidate.compact(),
                "value": round(score.total, 3),
                "progress": round(score.progress, 3),
                "information": round(score.information, 3),
                "novelty": round(score.novelty, 3),
                "risk": round(score.risk, 3),
                "planning": round(score.planning, 3),
                "predictions": [
                    {
                        "outcome": prediction.label(),
                        "belief": round(weight, 3),
                        "hypothesis": hypothesis_id,
                    }
                    for prediction, weight, hypothesis_id in self.hypotheses.predict(
                        score.candidate.key
                    )[:4]
                ]
                if self.config.uses_hypotheses
                else [],
            }
            for score in ranking[:8]
        ]
        return {
            "agent": "HypothesisWorld",
            "game_id": self.game_id,
            "ablation": self.config.ablation.value,
            "seed": self.config.random_seed,
            "step": self.action_counter,
            "state": {
                "fingerprint": state.fingerprint,
                "raw_fingerprint": state.raw_fingerprint,
                "objects": len(state.objects),
                "visited": self.memory.visited_states[state.fingerprint],
                "unique_abstract_states": len(self.memory.visited_states),
                "unique_raw_states": len(self.memory.raw_states),
                "raw_aliases": len(self.memory.state_aliases[state.fingerprint]),
                "persistent_object_types": len(self.memory.persistent_object_signatures()),
                "transient_object_types": len(self.memory.transient_object_signatures),
            },
            "hypotheses": top_hypotheses,
            "candidates": candidates,
            "selected": selected.compact(),
        }

    def _reset(self, reason: str) -> GameAction:
        self._previous_state = None
        self._pending_action = None
        self._previous_diff = None
        action = GameAction.RESET
        action.reasoning = {
            "agent": "HypothesisWorld",
            "selected": "RESET",
            "why": reason,
        }
        return action

    def append_frame(self, frame: FrameData) -> None:
        """Consume outcomes immediately, including terminal WIN transitions."""
        super().append_frame(frame)
        if self._previous_state is None or self._pending_action is None:
            return
        current = perceive(frame)
        self._observe_transition(current, frame)
        self._previous_state = current
        self._previous_levels = int(frame.levels_completed or 0)
        self._pending_action = None

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        state_name = self._state_name(latest_frame)
        current = perceive(latest_frame)
        if self._previous_state is None and current.width and current.height:
            self.memory.observe_state(current, self.action_counter)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return self._reset(f"required by state {state_name}")

        legal_ids = self._legal_action_ids(latest_frame)
        if not legal_ids:
            # The interface requires an action. RESET is the only safe recovery
            # when a nonterminal frame unexpectedly exposes no legal actions.
            return self._reset("no available_actions were exposed")

        candidates = generate_candidates(
            legal_ids,
            current,
            self.memory,
            self._previous_diff,
            self.config.max_coordinate_candidates,
        )
        ranking = self.selector.rank(current, candidates)
        if not ranking:
            return self._reset("available_actions produced no candidates")
        selected = ranking[0].candidate
        reasoning = self._reasoning(current, ranking, selected)
        logger.info(
            "HYPOTHESISWORLD %s",
            json.dumps({"event": "selection", **reasoning}, separators=(",", ":")),
        )

        action = GameAction.from_id(selected.key.action_id)
        if selected.coordinates is not None:
            x, y = selected.coordinates
            action.set_data({"x": x, "y": y})
        action.reasoning = reasoning

        self._previous_state = current
        self._previous_levels = int(latest_frame.levels_completed or 0)
        self._pending_action = selected.key
        return action
