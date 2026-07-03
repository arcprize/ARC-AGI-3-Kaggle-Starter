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
from agi_core.types import (
    ActionDecision,
    ActionProposal,
    ExecutiveContext,
    GoalState,
    MemoryBundle,
    Observation,
)


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

    def default_metadata_for_action(self, action_name: str) -> dict[str, int]:
        if action_name == "ACTION6":
            return {"x": 32, "y": 32}
        return {}


class FakeLLMBackend:
    def __init__(self, payload, available: bool = True) -> None:
        self.payload = payload
        self.available = available
        self.calls = 0
        self.last_prompt = None

    def generate_json(self, prompt: str, max_tokens: int, temperature: float):
        self.calls += 1
        self.last_prompt = prompt
        return self.payload


class FakeContextualHyperon:
    def __init__(self) -> None:
        self.synced = False
        self.updated_proposals = []
        self.contextual_calls: list[tuple[str | None, list[str]]] = []
        self._current_bundle = MemoryBundle(recommended_actions=["ACTION1"])

    def sync(self, observation, memory, goals) -> None:
        self.synced = True

    def current_memory_bundle(self):
        return self._current_bundle

    def contextual_memory_bundle(self, observation, subgoal, legal_actions):
        self.contextual_calls.append((subgoal, list(legal_actions)))
        return MemoryBundle(
            recommended_actions=["ACTION2"],
            subgoal_action_stats={
                "ACTION2": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
            },
        )

    def update_proposals(self, proposals) -> None:
        self.updated_proposals = list(proposals)

    def run_rules(self, query: str):
        if query == "goal-hints":
            return [{"subgoal": "break-stall", "confidence": 0.8}]
        if query == "reset-needed":
            return []
        if query == "action-priors:break-stall":
            return [
                {
                    "action_name": "ACTION1",
                    "confidence": 0.62,
                    "reason": "symbolic prior",
                }
            ]
        if query == "action-priors":
            return [
                {
                    "action_name": "ACTION1",
                    "confidence": 0.62,
                    "reason": "symbolic prior",
                }
            ]
        return []


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

    def test_unknown_legality_fails_closed_to_reset(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=[],
            derived_features={"legal_actions_known": False},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        decision = self.executive.decide(context)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.action_name, "RESET")
        self.assertEqual(decision.reason, "fail-closed unknown legality reset")

    def test_fallback_prefers_memory_recommended_action_when_specialists_fail(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=9,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION2"],
            suppressed_actions=["ACTION1"],
        )

        decision = self.executive._memory_aware_fallback(context)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.action_name, "ACTION2")
        self.assertEqual(decision.reason, "memory-aware executive fallback")

    def test_fallback_resets_when_memory_suppresses_all_non_reset_legal_actions(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=9,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["RESET", "ACTION1", "ACTION2"],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1", "ACTION2"],
        )

        decision = self.executive._memory_aware_fallback(context)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.action_name, "RESET")
        self.assertEqual(decision.reason, "memory-aware exhausted fallback reset")

    def test_fallback_assigns_default_metadata_for_memory_recommended_complex_action(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=9,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION6"],
            suppressed_actions=["ACTION1"],
        )

        decision = self.executive._memory_aware_fallback(context)

        self.assertEqual(decision.action_name, "ACTION6")
        self.assertEqual(decision.metadata, {"x": 32, "y": 32})

    def test_fallback_avoids_adapter_deterministic_action_when_memory_suppresses_it(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=9,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1"],
        )

        decision = self.executive._memory_aware_fallback(context)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.action_name, "ACTION2")
        self.assertEqual(decision.reason, "memory-aware legal fallback")

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

    def test_frame_context_stall_penalty_discourages_repeating_same_stalled_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(3):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": "sig-loop"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": "sig-loop"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.5,
                    reason="stalled-frame-loop",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-loop"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.84,
                    reason="repeat-stall",
                ),
                ActionProposal(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.70,
                    reason="fresh-option",
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION2")
        by_name = {proposal.action_name: proposal for proposal in rescored}
        self.assertLess(by_name["ACTION1"].score, by_name["ACTION2"].score)

    def test_state_only_stall_penalty_applies_when_frame_signature_is_missing(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(3):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.5,
                    reason="state-only-stall",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.84,
                    reason="repeat-state-stall",
                ),
                ActionProposal(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.70,
                    reason="fresh-option",
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION2")

    def test_state_only_exhaustion_applies_without_frame_signature(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="state-only-basin",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={},
        )

        self.assertTrue(memory.is_basin_exhausted(observation))

    def test_reactive_stall_breaker_avoids_deterministic_baseline_loop(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION1")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-reactive"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-reactive"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stall-seed",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-reactive"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION3")
        self.assertTrue(all(proposal.source == "reactive" for proposal in proposals))
        self.assertTrue(all("baseline" not in proposal.reason for proposal in proposals))

    def test_reactive_stall_breaker_ranks_fresh_actions_ahead_of_stalled_context_actions(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION3", "ACTION4", "ACTION3")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-reactive-rank"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-reactive-rank"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stall-seed",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-reactive-rank"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        proposals = self.executive.reactive.propose(context)

        self.assertGreaterEqual(len(proposals), 2)
        self.assertEqual([proposal.action_name for proposal in proposals[:2]], ["ACTION1", "ACTION2"])

    def test_reactive_stall_breaker_confidence_decays_with_long_stagnation(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        sequence = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION1", "ACTION2", "ACTION3")
        for index, action_name in enumerate(sequence):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
                derived_features={"frame_signature": "sig-decay"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
                derived_features={"frame_signature": "sig-decay"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stall-seed",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
            derived_features={"frame_signature": "sig-decay"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertLess(proposals[0].confidence, 0.55)

    def test_reactive_yields_when_current_frame_context_is_exhausted(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION4")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-exhausted"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-exhausted"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stall-seed",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-exhausted"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        proposals = self.executive.reactive.propose(context)

        self.assertEqual(proposals, [])

    def test_reactive_yields_immediately_when_persistent_memory_marks_basin_exhausted(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-persistent-exhausted"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
        )

        proposals = self.executive.reactive.propose(context)

        self.assertEqual(proposals, [])

    def test_reactive_yields_when_persistent_memory_suppresses_all_non_reset_actions(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-persistent-suppressed"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1", "ACTION2", "ACTION3"],
        )

        proposals = self.executive.reactive.propose(context)

        self.assertEqual(proposals, [])

    def test_reactive_yields_from_persistent_suppression_without_frame_signature(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1", "ACTION2"],
        )

        proposals = self.executive.reactive.propose(context)

        self.assertEqual(proposals, [])

    def test_reactive_stall_breaker_uses_persistent_recommendations_on_revisit(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION3", "ACTION4")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-reactive-persistent"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-reactive-persistent"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stall-seed",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-reactive-persistent"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION2", "ACTION1"],
            suppressed_actions=["ACTION3", "ACTION4"],
        )

        proposals = self.executive.reactive.propose(context)

        self.assertGreaterEqual(len(proposals), 2)
        self.assertEqual([proposal.action_name for proposal in proposals[:2]], ["ACTION2", "ACTION1"])

    def test_reactive_uses_persistent_memory_baseline_before_stagnation_threshold(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-reactive-baseline-memory"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION3", "ACTION2"],
            suppressed_actions=["ACTION1"],
        )

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION3")
        self.assertEqual(proposals[0].reason, "memory-informed reactive baseline")

    def test_reactive_memory_baseline_assigns_default_metadata_for_complex_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            derived_features={"frame_signature": "sig-reactive-action6"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION6"],
            suppressed_actions=["ACTION1"],
        )

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 32, "y": 32})

    def test_reactive_deterministic_baseline_avoids_suppressed_adapter_choice(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1"],
        )

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION2")
        self.assertEqual(proposals[0].reason, "deterministic reactive baseline")

    def test_symbolic_assigns_adapter_metadata_for_complex_action(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION6"],
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )

        class Action6Hyperon(FakeContextualHyperon):
            def run_rules(self, query: str):
                if query == "goal-hints":
                    return []
                if query == "reset-needed":
                    return []
                if query == "action-priors":
                    return [{"action_name": "ACTION6", "confidence": 0.72}]
                return []

        context.hyperon = Action6Hyperon()

        proposals = self.executive.symbolic.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 32, "y": 32})

    def test_critic_uses_proposal_subgoal_bundle_instead_of_single_global_bundle(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION1"],
            subgoal_action_stats={
                "ACTION1": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
            },
        )
        context.contextual_memory_bundles["advance-level"] = context.memory_bundle
        context.contextual_memory_bundles["break-stall"] = MemoryBundle(
            recommended_actions=["ACTION2"],
            subgoal_action_stats={
                "ACTION2": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
            },
        )

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.60,
                    reason="advance",
                    subgoal="advance-level",
                ),
                ActionProposal(
                    action_name="ACTION2",
                    source="reactive",
                    confidence=0.59,
                    reason="break",
                    subgoal="break-stall",
                ),
            ],
        )

        by_name = {proposal.action_name: proposal for proposal in rescored}
        self.assertGreater(by_name["ACTION1"].score, 0.60)
        self.assertGreater(by_name["ACTION2"].score, 0.59)

    def test_executive_replan_mode_suppresses_reactive_under_deep_stagnation(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(12):
            action_name = "ACTION1" if index % 2 == 0 else "ACTION2"
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stagnation-seed",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=13,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        decision = self.executive.decide(context)

        self.assertNotEqual(decision.source, "reactive")
        self.assertTrue(all(proposal.source != "reactive" for proposal in decision.proposals))

    def test_executive_replan_mode_forces_llm_even_when_symbolic_gate_is_met(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(12):
            action_name = "ACTION1" if index % 2 == 0 else "ACTION2"
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stagnation-seed",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.95,
                "reason": "llm override",
                "action_candidates": [{"action": "ACTION2", "confidence": 0.95}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=13,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
        )
        context = build_context(observation, memory, llm_backend)

        decision = self.executive.decide(context)

        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.source, "llm")

    def test_basin_exhaustion_drops_symbolic_below_gate_and_triggers_llm(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION4")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-llm-basin"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-llm-basin"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stalled-basin",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.92,
                "reason": "llm escape",
                "action_candidates": [{"action": "ACTION4", "confidence": 0.92}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-llm-basin"},
        )
        context = build_context(observation, memory, llm_backend)

        decision = self.executive.decide(context)

        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.source, "llm")

    def test_basin_exhaustion_replan_mode_engages_before_global_threshold(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION4")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-local-basin"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-local-basin"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="basin-seed",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.88,
                "reason": "basin llm",
                "action_candidates": [{"action": "ACTION4", "confidence": 0.88}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-local-basin"},
        )
        context = build_context(observation, memory, llm_backend)

        decision = self.executive.decide(context)

        self.assertLess(memory.stagnation_steps, context.config.executive_replan_stagnation_threshold)
        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.source, "llm")

    def test_basin_exhaustion_invalid_llm_payload_still_yields_llm_replan(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION4")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-invalid-llm"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-invalid-llm"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="basin-seed",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(payload="bad-payload")
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-invalid-llm"},
        )
        context = build_context(observation, memory, llm_backend)

        decision = self.executive.decide(context)

        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.source, "llm")
        self.assertIn("memory-guided llm fallback", decision.reason)

    def test_episode_only_basin_exhaustion_enters_replan_without_persisted_bundle(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION4")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-episode-only"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-episode-only"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="episode-only-basin",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.90,
                "reason": "episode-only llm",
                "action_candidates": [{"action": "ACTION4", "confidence": 0.90}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-episode-only"},
        )
        context = build_context(observation, memory, llm_backend)
        context.memory_bundle = MemoryBundle()

        decision = self.executive.decide(context)

        self.assertLess(memory.stagnation_steps, context.config.executive_replan_stagnation_threshold)
        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.source, "llm")
        self.assertTrue(all(proposal.source != "reactive" for proposal in decision.proposals))

    def test_llm_prompt_includes_deterministic_memory_bundle(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-prompt"},
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-prompt"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="stall-seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(after)
        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.95,
                "reason": "llm replan",
                "action_candidates": [{"action": "ACTION2", "confidence": 0.95}],
            }
        )
        context = build_context(after, memory, llm_backend)

        context.hyperon.sync(context.observation, context.memory, context.goals)
        context.memory_bundle = context.hyperon.current_memory_bundle()
        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertIn("memory_basin_status=", llm_backend.last_prompt)
        self.assertIn("memory_recent_records=", llm_backend.last_prompt)
        self.assertIn("memory_basin_action_stats=", llm_backend.last_prompt)
        self.assertIn("If memory_basin_exhausted=true", llm_backend.last_prompt)

    def test_llm_prompt_marks_episode_only_basin_exhaustion(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-prompt-episode"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-prompt-episode"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="prompt-seed",
                )
            )
            memory.note_observation(after)
        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.95,
                "reason": "llm replan",
                "action_candidates": [{"action": "ACTION2", "confidence": 0.95}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-prompt-episode"},
        )
        context = build_context(observation, memory, llm_backend)
        context.memory_bundle = MemoryBundle()

        LLMSpecialist().propose(context)

        self.assertIn("memory_basin_exhausted=True", llm_backend.last_prompt)

    def test_critic_prefers_memory_recommended_action_over_close_confidence_alternative(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-critic-recommended"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION2"],
            basin_action_stats={
                "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
            },
            state_action_stats={
                "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
            },
            subgoal_action_stats={
                "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
            },
        )

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.76,
                    reason="default",
                ),
                ActionProposal(
                    action_name="ACTION2",
                    source="llm",
                    confidence=0.72,
                    reason="memory-backed",
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION2")

    def test_critic_demotes_memory_suppressed_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-critic-suppressed"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1"],
            basin_action_stats={
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 2, "total": 3},
            },
        )

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.80,
                    reason="suppressed",
                ),
                ActionProposal(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.70,
                    reason="fresh",
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION2")

    def test_executive_refreshes_subgoal_specific_memory_bundle_before_llm(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(4):
            action_name = "ACTION1" if index % 2 == 0 else "ACTION2"
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-subgoal-refresh"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-subgoal-refresh"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stagnation-seed",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.90,
                "reason": "llm action",
                "action_candidates": [{"action": "ACTION2", "confidence": 0.90}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-subgoal-refresh"},
        )
        context = build_context(observation, memory, llm_backend)
        fake_hyperon = FakeContextualHyperon()
        context.hyperon = fake_hyperon

        decision = self.executive.decide(context)

        self.assertEqual(
            fake_hyperon.contextual_calls,
            [
                ("break-stall", ["ACTION1", "ACTION2"]),
                ("replan", ["ACTION1", "ACTION2"]),
            ],
        )
        self.assertEqual(context.memory_bundle.recommended_actions, ["ACTION1"])
        self.assertEqual(context.llm_memory_bundle.recommended_actions, ["ACTION2"])
        self.assertIn("memory_recommended_actions=['ACTION2']", llm_backend.last_prompt)
        self.assertEqual(decision.source, "llm")

    def test_llm_specialist_falls_back_to_memory_recommendation_when_payload_invalid(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-llm-fallback"},
        )
        memory.note_observation(observation)
        context = build_context(observation, memory, FakeLLMBackend(payload="bad-payload"))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            recommended_actions=["ACTION3", "ACTION1"],
            suppressed_actions=["ACTION2"],
        )

        proposals = LLMSpecialist().propose(context)

        self.assertEqual(context.llm_backend.calls, 1)
        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION3")
        self.assertEqual(proposals[0].source, "llm")
        self.assertIn("memory-guided llm fallback", proposals[0].reason)

    def test_llm_specialist_fallback_avoids_suppressed_actions(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-llm-suppressed"},
        )
        after = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-llm-suppressed"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="stall-seed",
            )
        )
        memory.note_observation(after)
        context = build_context(after, memory, FakeLLMBackend(payload={"action_candidates": []}))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            recommended_actions=["ACTION1"],
            suppressed_actions=["ACTION1", "ACTION2"],
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION3")
        self.assertNotIn("ACTION1", [proposal.action_name for proposal in proposals[:1]])
        self.assertNotIn("ACTION2", [proposal.action_name for proposal in proposals[:1]])

    def test_llm_specialist_fallback_assigns_adapter_metadata_for_complex_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            derived_features={"frame_signature": "sig-llm-action6"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": [{"action": "ILLEGAL"}]}),
        )
        context.llm_memory_bundle = MemoryBundle(
            recommended_actions=["ACTION6"],
            suppressed_actions=["ACTION1"],
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 32, "y": 32})

    def test_llm_specialist_exhausted_fallback_ranks_suppressed_escape_actions_by_broader_memory(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-broader-escape"},
        )
        memory.note_observation(observation)
        context = build_context(observation, memory, FakeLLMBackend(payload={"action_candidates": []}))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION2", "ACTION3"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
                "ACTION2": {"progress": 2, "failed": 0, "stalled": 1, "total": 3},
                "ACTION3": {"progress": 0, "failed": 1, "stalled": 1, "total": 2},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 3, "total": 4},
                "ACTION2": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
                "ACTION3": {"progress": 0, "failed": 1, "stalled": 1, "total": 2},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 3, "total": 3},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION3": {"progress": 0, "failed": 1, "stalled": 1, "total": 2},
            },
            subgoal_action_stats={
                "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
            },
            trained_prior_bucket={"ACTION1": 1, "ACTION2": 9, "ACTION3": 2},
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION2")
        self.assertEqual(proposals[0].source, "llm")

    def test_llm_specialist_broader_escape_ranking_prefers_basin_progress_over_coarse_frame_stats(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-basin-llm"},
        )
        memory.note_observation(observation)
        context = build_context(observation, memory, FakeLLMBackend(payload={"action_candidates": []}))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION2"],
            basin_action_stats={
                "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 1, "total": 2},
            },
            frame_action_stats={
                "ACTION1": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 1, "total": 2},
                "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
            },
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION2")
