from __future__ import annotations

import unittest

from agi_core.config import RuntimeConfig
from agi_core.executive import Executive
from agi_core.knowledge.hyperon_bridge import HyperonBridge
from agi_core.memory import EpisodicMemory
from agi_core.specialists.critic import ProposalCritic
from agi_core.specialists.llm import LLMSpecialist
from agi_core.specialists.reactive import ReactiveSpecialist
from agi_core.specialists.symbolic import SymbolicSpecialist
from agi_core.types import ActionDecision, ActionProposal, ExecutiveContext, GoalState, Observation


class FakeAdapter:
    def bind_game(self, game_id: str) -> None:
        self.game_id = game_id

    def fallback_action_name(self, observation: Observation) -> str:
        if observation.legal_actions:
            return observation.legal_actions[0]
        return "RESET"

    def fallback_decision(self, observation: Observation) -> ActionDecision:
        return ActionDecision(
            action_name=self.fallback_action_name(observation),
            source="fallback",
            confidence=0.01,
            reason="fallback",
            used_fallback=True,
        )


class FakeLLMBackend:
    def __init__(self, payload, available: bool = True) -> None:
        self.payload = payload
        self.available = available
        self.calls = 0

    def generate_json(self, prompt: str, max_tokens: int, temperature: float):
        self.calls += 1
        return self.payload


def build_context(
    observation: Observation,
    memory: EpisodicMemory,
    llm_backend: FakeLLMBackend,
) -> ExecutiveContext:
    config = RuntimeConfig.from_env()
    adapter = FakeAdapter()
    adapter.bind_game(observation.game_id)
    hyperon = HyperonBridge()
    return ExecutiveContext(
        observation=observation,
        goals=[GoalState(name="finish-episode", target_state="WIN")],
        memory=memory,
        adapter=adapter,
        hyperon=hyperon,
        llm_backend=llm_backend,
        config=config,
        episode_id=observation.game_id,
    )


class ExecutiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=ReactiveSpecialist(),
            symbolic=SymbolicSpecialist(),
            llm=LLMSpecialist(),
            critic=ProposalCritic(),
        )

    def test_symbolic_high_confidence_blocks_llm(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(
                {
                    "subgoal": "ignored",
                    "confidence": 0.95,
                    "reason": "ignored",
                    "action_candidates": [{"action": "ACTION2"}],
                }
            ),
        )

        decision = self.executive.decide(context)

        self.assertEqual(context.llm_backend.calls, 0)
        self.assertEqual(decision.source, "symbolic")

    def test_illegal_llm_action_is_filtered(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1"],
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1"],
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="test",
            )
        )
        memory.note_observation(after)
        llm_backend = FakeLLMBackend(
            {
                "subgoal": "illegal-push",
                "confidence": 0.95,
                "reason": "try illegal action",
                "action_candidates": [{"action": "ACTION6"}],
            }
        )
        context = build_context(after, memory, llm_backend)

        decision = self.executive.decide(context)

        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.action_name, "ACTION1")
        self.assertNotEqual(decision.source, "llm")

    def test_fallback_is_used_when_no_specialist_can_propose(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=9,
            game_state="WIN",
            levels_completed=4,
            last_action="ACTION2",
            legal_actions=[],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        decision = self.executive.decide(context)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.action_name, "RESET")

    def test_loop_penalty_discourages_alternating_stall_cycle(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frames = [
            Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            )
            for index in range(5)
        ]
        for action_name, before, after in (
            ("ACTION1", frames[0], frames[1]),
            ("ACTION2", frames[1], frames[2]),
            ("ACTION1", frames[2], frames[3]),
            ("ACTION2", frames[3], frames[4]),
        ):
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stall-loop",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.80,
                    reason="loop-option",
                ),
                ActionProposal(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.79,
                    reason="loop-option",
                ),
                ActionProposal(
                    action_name="ACTION3",
                    source="symbolic",
                    confidence=0.70,
                    reason="fresh-option",
                ),
            ],
        )
        best = rescored[0]
        self.assertEqual(best.action_name, "ACTION3")
        by_name = {proposal.action_name: proposal for proposal in rescored}
        self.assertLess(by_name["ACTION1"].score, by_name["ACTION3"].score)
        self.assertLess(by_name["ACTION2"].score, by_name["ACTION3"].score)
