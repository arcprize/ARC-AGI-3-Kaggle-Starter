from __future__ import annotations

import unittest

from agi_core.config import RuntimeConfig
from agi_core.executive import Executive
from agi_core.image_input import BoardImageInput, compute_board_signature
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
    def __init__(self, payload, available: bool = True, last_error: str | None = None) -> None:
        self.payload = payload
        self.available = available
        self.calls = 0
        self.last_prompt = None
        self.last_error = last_error

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

    def test_recent_board_probe_grace_turn_skips_immediate_llm_escalation(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[1, 0], [0, 0]]
        image_input = BoardImageInput(
            board=board,
            height=2,
            width=2,
            frame_signature=compute_board_signature(board),
            source_attr="frame",
        )
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION6"],
            image_input=image_input,
            derived_features={"frame_signature": "sig-probe-grace"},
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=image_input,
            derived_features={"frame_signature": "sig-probe-grace"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.78,
                reason="symbolic board inspection target [board]",
                subgoal="inspect-active-board",
            )
        )
        memory.note_observation(after)
        context = build_context(
            after,
            memory,
            FakeLLMBackend(
                {
                    "subgoal": "replan",
                    "confidence": 0.95,
                    "reason": "llm override",
                    "action_candidates": [{"action": "ACTION1", "confidence": 0.95}],
                }
            ),
        )

        decision = self.executive.decide(context)

        self.assertEqual(context.llm_backend.calls, 0)
        self.assertNotEqual(decision.source, "llm")

    def test_second_board_probe_uses_alternate_probe_target(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        image_input = BoardImageInput(
            board=board,
            height=3,
            width=3,
            frame_signature=compute_board_signature(board),
            source_attr="frame",
        )
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION6"],
            image_input=image_input,
            derived_features={"frame_signature": "sig-alt-probe"},
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=image_input,
            derived_features={"frame_signature": "sig-alt-probe"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.78,
                reason="symbolic board inspection target [board]",
                subgoal="inspect-active-board",
            )
        )
        memory.note_observation(after)
        context = build_context(
            after,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )

        proposals = SymbolicSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 1, "y": 1})

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

    def test_executive_canonicalizes_winning_action6_decision_metadata(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 9, 9], [0, 9, 9]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-exec-canonicalize"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        class EmptyHyperon(FakeContextualHyperon):
            def run_rules(self, query: str):
                if query in {"goal-hints", "reset-needed", "action-priors", "action-priors:break-stall"}:
                    return []
                return []

        class FixedReactive:
            def propose(self, _context):
                return [
                    ActionProposal(
                        action_name="ACTION6",
                        source="reactive",
                        confidence=0.60,
                        reason="noncanonical board probe",
                        subgoal="maintain-control-loop",
                        metadata={"x": 1, "y": 1},
                    )
                ]

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=FixedReactive(),
            symbolic=SymbolicSpecialist(),
            llm=LLMSpecialist(),
            critic=ProposalCritic(),
        )
        context.hyperon = EmptyHyperon()

        decision = executive.decide(context)

        self.assertEqual(decision.action_name, "ACTION6")
        self.assertEqual(decision.metadata, {"x": 2, "y": 2})
        self.assertEqual(decision.subgoal, "inspect-active-board")

    def test_executive_does_not_infer_board_subgoal_for_non_action6_winner(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 9, 9], [0, 9, 9]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-exec-non-action6"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        class EmptyHyperon(FakeContextualHyperon):
            def run_rules(self, query: str):
                if query in {"goal-hints", "reset-needed", "action-priors", "action-priors:break-stall"}:
                    return []
                return []

        class FixedReactive:
            def propose(self, _context):
                return [
                    ActionProposal(
                        action_name="ACTION1",
                        source="reactive",
                        confidence=0.60,
                        reason="plain action",
                        subgoal=None,
                    )
                ]

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=FixedReactive(),
            symbolic=SymbolicSpecialist(),
            llm=LLMSpecialist(),
            critic=ProposalCritic(),
        )
        context.hyperon = EmptyHyperon()

        decision = executive.decide(context)

        self.assertEqual(decision.action_name, "ACTION1")
        self.assertIsNone(decision.subgoal)

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

    def test_reactive_no_explicit_legal_actions_keeps_board_probe_metadata_and_subgoal(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 6, 6], [0, 6, 6]]
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=[],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"legal_actions_known": True, "frame_signature": "sig-no-explicit-legal"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        class BoardProbeAdapter(FakeAdapter):
            def fallback_action_name(self, observation: Observation) -> str:
                return "ACTION6"

        context.adapter = BoardProbeAdapter()

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")

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
        self.assertIsNone(decision.subgoal)

    def test_fallback_prefers_board_probe_action_when_live_board_is_available(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 4, 4], [0, 4, 4]]
        observation = Observation(
            game_id="ls20",
            step_index=9,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle()

        decision = self.executive._memory_aware_fallback(context)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.action_name, "ACTION6")
        self.assertEqual(decision.reason, "memory-aware legal fallback")
        self.assertEqual(decision.metadata, {"x": 2, "y": 2})
        self.assertEqual(decision.subgoal, "inspect-active-board")

    def test_fallback_avoids_repeating_stalled_board_probe_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 4, 4], [0, 4, 4]]
        frame_signature = compute_board_signature(board)
        frames = [
            Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION6" if index else None,
                legal_actions=["ACTION1", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=frame_signature,
                    source_attr="frame",
                ),
                derived_features={"frame_signature": frame_signature},
            )
            for index in range(4)
        ]
        for before, after in zip(frames, frames[1:]):
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION6",
                    source="fallback",
                    confidence=0.05,
                    reason="stalled board probe",
                    subgoal="inspect-active-board",
                    metadata={"x": 2, "y": 2},
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=frame_signature,
                source_attr="frame",
            ),
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(recommended_actions=["ACTION6"])

        decision = self.executive._memory_aware_fallback(context)

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.action_name, "ACTION1")
        self.assertEqual(decision.reason, "memory-aware legal fallback")

    def test_exploratory_fallback_prefers_untried_current_frame_actions_before_revisits(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_signature = "sig-exploratory-untried"
        for index, action_name in enumerate(("ACTION5", "ACTION4")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION4", "ACTION5"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION4", "ACTION5"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor executive fallback",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION4", "ACTION5"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        decision = self.executive._memory_aware_fallback(
            context,
            candidate_actions=["ACTION1", "ACTION2", "ACTION4", "ACTION5"],
            exploratory=True,
            respect_suppressed_actions=False,
            honor_recommended_actions=True,
        )

        self.assertIn(decision.action_name, {"ACTION1", "ACTION2"})

    def test_exploratory_fallback_can_follow_memory_recommended_order(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION5",
            legal_actions=["ACTION1", "ACTION2", "ACTION4", "ACTION5"],
            derived_features={"frame_signature": "sig-exploratory-recommended"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(recommended_actions=["ACTION2", "ACTION1"])

        decision = self.executive._memory_aware_fallback(
            context,
            candidate_actions=["ACTION1", "ACTION2", "ACTION4", "ACTION5"],
            exploratory=True,
            respect_suppressed_actions=False,
            honor_recommended_actions=True,
        )

        self.assertEqual(decision.action_name, "ACTION2")

    def test_score_floor_fallback_can_reintroduce_fresh_board_probe_after_nonprobe_recovery(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 6, 6], [0, 6, 6]]
        frame_signature = compute_board_signature(board)
        actions = ("ACTION1", "ACTION2", "ACTION3")
        for index, action_name in enumerate(actions):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None if index == 0 else actions[index - 1],
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=frame_signature,
                    source_attr="frame",
                ),
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=frame_signature,
                    source_attr="frame",
                ),
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor executive fallback",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=frame_signature,
                source_attr="frame",
            ),
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(recommended_actions=["ACTION6"])

        decision = self.executive._memory_aware_fallback(
            context,
            candidate_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
            exploratory=True,
            respect_suppressed_actions=False,
            honor_recommended_actions=True,
            allow_exploratory_board_probe=True,
        )

        self.assertEqual(decision.action_name, "ACTION6")
        self.assertEqual(decision.metadata, {"x": 2, "y": 2})
        self.assertEqual(decision.subgoal, "inspect-active-board")

    def test_score_floor_fallback_appends_action6_when_late_reprobe_is_allowed(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 6, 6], [0, 6, 6]]
        frame_signature = compute_board_signature(board)
        recovery_signature = "sig-score-floor-late-reprobe"
        actions = ("ACTION6", "ACTION6", "ACTION1", "ACTION2", "ACTION3")
        for index, action_name in enumerate(actions):
            signature = frame_signature if index < 2 else recovery_signature
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None if index == 0 else actions[index - 1],
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=signature,
                    source_attr="frame",
                ),
                derived_features={"frame_signature": signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=signature,
                    source_attr="frame",
                ),
                derived_features={"frame_signature": signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason=(
                        "stalled board probe"
                        if action_name == "ACTION6"
                        else "score-floor executive fallback"
                    ),
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=recovery_signature,
                source_attr="frame",
            ),
            derived_features={"frame_signature": recovery_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(recommended_actions=["ACTION6"])
        rescored = [
            ActionProposal("ACTION1", "symbolic", 0.10, "fallback"),
            ActionProposal("ACTION2", "symbolic", 0.10, "fallback"),
            ActionProposal("ACTION3", "symbolic", 0.10, "fallback"),
            ActionProposal("ACTION6", "symbolic", 0.10, "fallback"),
        ]
        for proposal, score in zip(rescored, (-0.20, -0.22, -0.24, -0.30)):
            proposal.legal = True
            proposal.score = score

        decision = self.executive._score_floor_fallback(context, rescored)

        self.assertEqual(decision.action_name, "ACTION6")
        self.assertEqual(decision.reason, "score-floor executive fallback")
        self.assertEqual(decision.metadata, {"x": 2, "y": 2})
        self.assertEqual(decision.subgoal, "inspect-active-board")

    def test_repeated_board_probe_context_metadata_uses_alternate_probe_target(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 6, 6], [0, 6, 6]]
        frame_signature = compute_board_signature(board)
        frames = [
            Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION6" if index else None,
                legal_actions=["ACTION1", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=frame_signature,
                    source_attr="frame",
                ),
                derived_features={"frame_signature": frame_signature},
            )
            for index in range(3)
        ]
        for before, after in zip(frames, frames[1:]):
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION6",
                    source="fallback",
                    confidence=0.05,
                    reason="stalled board probe",
                    subgoal="inspect-active-board",
                    metadata={"x": 2, "y": 2},
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=frame_signature,
                source_attr="frame",
            ),
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        metadata = self.executive._board_probe_metadata_for_context(context)

        self.assertEqual(metadata, {"x": 2, "y": 1})

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

    def test_reactive_stall_breaker_prefers_board_probe_action_and_assigns_metadata(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION1")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION6"],
                image_input=BoardImageInput(
                    board=[[0, 0, 0], [0, 5, 5], [0, 5, 5]],
                    height=3,
                    width=3,
                    frame_signature="sig-stall-probe",
                    source_attr="frame",
                ),
                derived_features={"frame_signature": "sig-stall-probe"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION6"],
                image_input=BoardImageInput(
                    board=[[0, 0, 0], [0, 5, 5], [0, 5, 5]],
                    height=3,
                    width=3,
                    frame_signature="sig-stall-probe",
                    source_attr="frame",
                ),
                derived_features={"frame_signature": "sig-stall-probe"},
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
            legal_actions=["ACTION1", "ACTION2", "ACTION6"],
            image_input=BoardImageInput(
                board=[[0, 0, 0], [0, 5, 5], [0, 5, 5]],
                height=3,
                width=3,
                frame_signature="sig-stall-probe",
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-stall-probe"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")
        self.assertEqual(proposals[0].reason, "reactive stall breaker")
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")

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
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
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
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

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

    def test_reactive_deterministic_baseline_prefers_board_probe_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 6, 6], [0, 6, 6]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-reactive-board-probe"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        proposals = self.executive.reactive.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].reason, "deterministic reactive baseline")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")

    def test_symbolic_assigns_adapter_metadata_for_complex_action(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [4, 4, 0], [4, 4, 0]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="board",
            ),
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
        self.assertEqual(proposals[0].metadata, {"x": 1, "y": 2})

    def test_symbolic_adds_board_target_fallback_when_priors_are_absent(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 8, 8], [0, 8, 8]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="board",
            ),
            derived_features={"frame_signature": "sig-symbolic-fallback"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )

        class NoPriorHyperon(FakeContextualHyperon):
            def run_rules(self, query: str):
                if query in {"goal-hints", "reset-needed", "action-priors"}:
                    return []
                return []

        context.hyperon = NoPriorHyperon()

        proposals = self.executive.symbolic.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})
        self.assertIn("symbolic board-target fallback", proposals[0].reason)
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")

    def test_symbolic_board_target_fallback_respects_memory_suppression(self) -> None:
        memory = EpisodicMemory()
        board = [[5, 0], [0, 0]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=2,
                width=2,
                frame_signature=compute_board_signature(board),
                source_attr="board",
            ),
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )
        context.memory_bundle = MemoryBundle(suppressed_actions=["ACTION6"])

        class NoPriorHyperon(FakeContextualHyperon):
            def run_rules(self, query: str):
                if query in {"goal-hints", "reset-needed", "action-priors"}:
                    return []
                return []

        context.hyperon = NoPriorHyperon()

        proposals = self.executive.symbolic.propose(context)

        self.assertEqual(proposals, [])

    def test_symbolic_prefers_action6_when_goal_hint_requests_board_inspection(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 8, 8], [0, 8, 8]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="board",
            ),
            derived_features={"frame_signature": "sig-symbolic-board-inspect"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )

        class InspectBoardHyperon(FakeContextualHyperon):
            def run_rules(self, query: str):
                if query == "goal-hints":
                    return [{"subgoal": "inspect-active-board", "confidence": 0.74}]
                if query == "reset-needed":
                    return []
                if query == "action-priors:inspect-active-board":
                    return [{"action_name": "ACTION1", "confidence": 0.72, "reason": "generic prior"}]
                return []

        context.hyperon = InspectBoardHyperon()

        proposals = self.executive.symbolic.propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})
        self.assertIn("symbolic board inspection target", proposals[0].reason)

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

    def test_executive_prefers_board_probe_subgoal_for_contextual_llm_bundle(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 4, 4], [0, 4, 4]]
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-contextual-board"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        selected = self.executive._select_contextual_subgoal(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.55,
                    reason="higher raw confidence",
                    subgoal="advance-level",
                )
            ],
            [
                ActionProposal(
                    action_name="ACTION6",
                    source="reactive",
                    confidence=0.40,
                    reason="board probe",
                    subgoal="maintain-control-loop",
                    metadata={"x": 2, "y": 2},
                )
            ],
        )

        self.assertEqual(selected, "maintain-control-loop")

    def test_executive_prefers_canonical_board_probe_subgoal_over_noncanonical_probe(self) -> None:
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 4, 4], [0, 4, 4]]
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-contextual-canonical-board"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        selected = self.executive._select_contextual_subgoal(
            context,
            [
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.65,
                    reason="off-centroid board probe",
                    subgoal="inspect-offset",
                    metadata={"x": 1, "y": 1},
                )
            ],
            [
                ActionProposal(
                    action_name="ACTION6",
                    source="reactive",
                    confidence=0.55,
                    reason="canonical board probe",
                    subgoal="inspect-centroid",
                    metadata={"x": 2, "y": 2},
                )
            ],
        )

        self.assertEqual(selected, "inspect-centroid")

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

    def test_emergency_llm_replan_bypasses_normal_llm_budget_for_dead_basin_cycle(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        frame_signature = "sig-emergency-llm-replan"
        for index, action_name in enumerate(
            ("ACTION1", "ACTION2", "ACTION3", "ACTION1", "ACTION2", "ACTION3")
        ):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor executive fallback",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.91,
                "reason": "emergency llm override",
                "action_candidates": [{"action": "ACTION2", "confidence": 0.91}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=7,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, llm_backend)
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION3"],
        )
        context.llm_memory_bundle = context.memory_bundle

        decision = self.executive.decide(context)

        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(memory.llm_calls, 3)
        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.action_name, "ACTION2")

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

    def test_repeated_board_probe_dead_end_forces_llm_replan_before_global_threshold(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        first = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end"},
        )
        second = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end"},
        )
        third = Observation(
            game_id="ls20",
            step_index=2,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end"},
        )
        memory.note_observation(first)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.70,
                reason="board probe",
                subgoal="inspect-active-board",
            )
        )
        memory.note_observation(second)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.50,
                reason="board probe followup",
                subgoal="break-stall",
            )
        )
        memory.note_observation(third)
        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.93,
                "reason": "probe dead end llm",
                "action_candidates": [{"action": "ACTION2", "confidence": 0.93}],
            }
        )
        context = build_context(third, memory, llm_backend)

        decision = self.executive.decide(context)

        self.assertLess(memory.stagnation_steps, context.config.executive_replan_stagnation_threshold)
        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.action_name, "ACTION2")
        self.assertTrue(context.probe_dead_end_replan)

    def test_llm_specialist_filters_action6_during_probe_dead_end_replan(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        first = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-llm-filter"},
        )
        second = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-llm-filter"},
        )
        third = Observation(
            game_id="ls20",
            step_index=2,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-llm-filter"},
        )
        memory.note_observation(first)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.70,
                reason="board probe",
                subgoal="inspect-active-board",
            )
        )
        memory.note_observation(second)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.50,
                reason="board probe followup",
                subgoal="break-stall",
            )
        )
        memory.note_observation(third)
        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.93,
                "reason": "probe dead end llm",
                "action_candidates": [
                    {"action": "ACTION6", "confidence": 0.95},
                    {"action": "ACTION2", "confidence": 0.80},
                ],
            }
        )
        context = build_context(third, memory, llm_backend)
        context.probe_dead_end_replan = True
        context.force_llm_replan = True

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION2")
        self.assertNotIn("ACTION6", [proposal.action_name for proposal in proposals])

    def test_probe_dead_end_escape_overrides_top_symbolic_action_when_llm_unavailable(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = list(proposals)

            def propose(self, context):
                return list(self._proposals)

        class StubLLM:
            available = False

            def propose(self, context):
                return []

        class OrderedCritic:
            def rescore(self, context, proposals):
                scores = {
                    ("symbolic", "ACTION3"): 0.45,
                    ("symbolic", "ACTION2"): 0.25,
                }
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = scores.get((proposal.source, proposal.action_name), 0.1)
                    rescored.append(proposal)
                rescored.sort(key=lambda proposal: proposal.score, reverse=True)
                return rescored

        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        first = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-fallback"},
        )
        second = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-fallback"},
        )
        third = Observation(
            game_id="ls20",
            step_index=2,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-fallback"},
        )
        memory.note_observation(first)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.70,
                reason="board probe",
                subgoal="inspect-active-board",
            )
        )
        memory.note_observation(second)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.50,
                reason="board probe followup",
                subgoal="break-stall",
            )
        )
        memory.note_observation(third)
        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION3", "symbolic", 0.70, "weak symbolic break-stall"),
                    ActionProposal("ACTION2", "symbolic", 0.55, "secondary symbolic"),
                ]
            ),
            llm=StubLLM(),
            critic=OrderedCritic(),
        )
        context = build_context(third, memory, StubLLM())

        decision = executive.decide(context)

        self.assertTrue(context.probe_dead_end_replan)
        self.assertEqual(decision.action_name, "ACTION2")
        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "probe-dead-end executive escape")
        self.assertTrue(decision.used_fallback)

    def test_probe_dead_end_followup_escape_blocks_immediate_symbolic_rebound(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = list(proposals)

            def propose(self, context):
                return list(self._proposals)

        class OrderedCritic:
            def rescore(self, context, proposals):
                scores = {
                    ("symbolic", "ACTION3"): 0.42,
                    ("symbolic", "ACTION2"): 0.18,
                    ("symbolic", "ACTION4"): 0.16,
                }
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = scores.get((proposal.source, proposal.action_name), 0.1)
                    rescored.append(proposal)
                rescored.sort(key=lambda proposal: proposal.score, reverse=True)
                return rescored

        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        before_escape = Observation(
            game_id="ls20",
            step_index=2,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-followup"},
        )
        after_escape = Observation(
            game_id="ls20",
            step_index=3,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-dead-end-followup"},
        )
        memory.note_observation(before_escape)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="fallback",
                confidence=0.05,
                reason="probe-dead-end memory escape",
            )
        )
        memory.note_observation(after_escape)
        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION3", "symbolic", 0.70, "symbolic rebound"),
                    ActionProposal("ACTION2", "symbolic", 0.45, "alternate symbolic"),
                    ActionProposal("ACTION4", "symbolic", 0.40, "alternate symbolic"),
                ]
            ),
            llm=FakeLLMBackend(payload=None, available=False),
            critic=OrderedCritic(),
        )
        context = build_context(after_escape, memory, FakeLLMBackend(payload=None, available=False))

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "probe-dead-end followup escape")
        self.assertEqual(decision.action_name, "ACTION2")
        self.assertTrue(decision.used_fallback)

    def test_probe_dead_end_followup_escape_does_not_refresh_recovery_window(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before_escape = Observation(
            game_id="ls20",
            step_index=2,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
            derived_features={"frame_signature": "sig-probe-dead-end-followup-window"},
        )
        after_escape = Observation(
            game_id="ls20",
            step_index=3,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-probe-dead-end-followup-window"},
        )
        post_followup = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-probe-dead-end-followup-window"},
        )
        expired = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-probe-dead-end-followup-window"},
        )
        memory.note_observation(before_escape)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="fallback",
                confidence=0.05,
                reason="probe-dead-end memory escape",
            )
        )
        memory.note_observation(after_escape)
        self.assertTrue(memory.recent_probe_dead_end_escape(after_escape))

        memory.note_decision(
            ActionDecision(
                action_name="ACTION2",
                source="fallback",
                confidence=0.08,
                reason="probe-dead-end followup escape",
            )
        )
        memory.note_observation(post_followup)

        self.assertTrue(memory.recent_probe_dead_end_escape(post_followup))
        memory.note_decision(
            ActionDecision(
                action_name="ACTION3",
                source="fallback",
                confidence=0.08,
                reason="probe-dead-end followup escape",
            )
        )
        memory.note_observation(expired)

        self.assertFalse(memory.recent_probe_dead_end_escape(expired))

    def test_probe_recovery_sequence_can_force_score_floor_before_global_stagnation_rule(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = list(proposals)

            def propose(self, context):
                return list(self._proposals)

        class StubLLM:
            def propose(self, context):
                return []

        class NegativeCritic:
            def rescore(self, context, proposals):
                scores = {
                    ("symbolic", "ACTION5"): -0.016,
                    ("symbolic", "ACTION4"): -0.034,
                    ("symbolic", "ACTION1"): -0.086,
                }
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = scores.get((proposal.source, proposal.action_name), -0.2)
                    rescored.append(proposal)
                rescored.sort(key=lambda proposal: proposal.score, reverse=True)
                return rescored

        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        observations = [
            Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=(None if index == 0 else action_name),
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=compute_board_signature(board),
                    source_attr="frame",
                ),
                derived_features={"frame_signature": "sig-probe-recovery-score-floor"},
            )
            for index, action_name in enumerate(["", "ACTION6", "ACTION6", "ACTION1", "ACTION2", "ACTION3"])
        ]
        decisions = [
            ActionDecision("ACTION6", "symbolic", 0.05, "probe"),
            ActionDecision("ACTION6", "symbolic", 0.05, "probe"),
            ActionDecision("ACTION1", "fallback", 0.05, "probe-dead-end memory escape"),
            ActionDecision("ACTION2", "fallback", 0.08, "probe-dead-end followup escape"),
            ActionDecision("ACTION3", "fallback", 0.08, "probe-dead-end followup escape"),
        ]
        memory.note_observation(observations[0])
        for index, decision in enumerate(decisions):
            memory.note_decision(decision)
            memory.note_observation(observations[index + 1])

        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION3", "ACTION4", "ACTION5"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-probe-recovery-score-floor"},
        )
        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION5", "symbolic", 0.164, "symbolic rebound"),
                    ActionProposal("ACTION4", "symbolic", 0.140, "alternate"),
                    ActionProposal("ACTION1", "symbolic", 0.120, "alternate"),
                ]
            ),
            llm=StubLLM(),
            critic=NegativeCritic(),
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor executive fallback")

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

    def test_invalid_llm_payload_reason_surfaces_backend_error(self) -> None:
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
                payload="bad-payload",
                last_error="response did not contain text output",
            ),
        )
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION2"],
            suppressed_actions=["ACTION1"],
            basin_status={"is_exhausted": True},
        )
        context.llm_memory_bundle = context.memory_bundle

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertIn("backend=response did not contain text output", proposals[0].reason)

    def test_invalid_local_llm_payload_reason_surfaces_raw_backend_text_snippet(self) -> None:
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
                payload=None,
                last_error="response did not contain a valid JSON object: ACTION6 because that seems safest",
            ),
        )
        context.memory_bundle = MemoryBundle(
            recommended_actions=["ACTION2"],
            suppressed_actions=["ACTION1"],
            basin_status={"is_exhausted": True},
        )
        context.llm_memory_bundle = context.memory_bundle

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertIn("ACTION6 because that seems safest", proposals[0].reason)

    def test_score_floor_legal_proposals_force_executive_fallback(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION2",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=StubSpecialist([]),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2", "ACTION6"],
            image_input=BoardImageInput(
                board=[[1, 0], [0, 0]],
                height=2,
                width=2,
                frame_signature=compute_board_signature([[1, 0], [0, 0]]),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-floor"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle()
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor executive fallback")
        self.assertEqual(decision.action_name, "ACTION1")

    def test_score_floor_fallback_uses_legal_proposal_actions_when_observation_actions_are_narrow(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION2",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=StubSpecialist([]),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION6"],
            image_input=BoardImageInput(
                board=[[1, 0], [0, 0]],
                height=2,
                width=2,
                frame_signature=compute_board_signature([[1, 0], [0, 0]]),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-floor-narrow"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle()
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor executive fallback")
        self.assertEqual(decision.action_name, "ACTION1")

    def test_score_floor_fallback_uses_memory_penalties_to_avoid_repeating_stalled_action(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION2",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=StubSpecialist([]),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(3):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-score-floor-loop"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-score-floor-loop"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.5,
                    reason="stalled-score-floor-loop",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-score-floor-loop"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle()
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor executive fallback")
        self.assertEqual(decision.action_name, "ACTION2")

    def test_soft_score_floor_enters_fallback_when_basin_is_exhausted_and_all_scores_are_non_positive(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                scores = {
                    "ACTION1": -0.01,
                    "ACTION2": -0.08,
                    "ACTION3": -0.12,
                }
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = scores[proposal.action_name]
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION1", "symbolic", 0.30, "weak symbolic prior"),
                    ActionProposal("ACTION2", "symbolic", 0.20, "weak symbolic prior"),
                    ActionProposal("ACTION3", "symbolic", 0.10, "weak symbolic prior"),
                ]
            ),
            llm=StubSpecialist([]),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(
            ("ACTION1", "ACTION2", "ACTION3", "ACTION1", "ACTION2", "ACTION3")
        ):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": "sig-soft-score-floor"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": "sig-soft-score-floor"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.2,
                    reason="stalled weak symbolic prior",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=7,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-soft-score-floor"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )
        context.memory_bundle = MemoryBundle(basin_status={"is_exhausted": True})
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor executive fallback")
        self.assertEqual(decision.action_name, "ACTION1")

    def test_soft_score_floor_enters_fallback_for_diverse_stall_cycle_without_explicit_basin_exhaustion(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                scores = {
                    "ACTION1": -0.02,
                    "ACTION2": -0.05,
                    "ACTION3": -0.08,
                    "ACTION4": -0.10,
                }
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = scores[proposal.action_name]
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION1", "symbolic", 0.30, "weak symbolic prior"),
                    ActionProposal("ACTION2", "symbolic", 0.28, "weak symbolic prior"),
                    ActionProposal("ACTION3", "symbolic", 0.24, "weak symbolic prior"),
                    ActionProposal("ACTION4", "symbolic", 0.20, "weak symbolic prior"),
                ]
            ),
            llm=StubSpecialist([]),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(
            ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION1", "ACTION2")
        ):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.2,
                    reason="diverse stalled symbolic prior",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=7,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )
        context.memory_bundle = MemoryBundle()
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor executive fallback")

    def test_score_floor_fallback_does_not_collapse_to_adapter_probe_when_bundle_suppresses_candidates(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION2",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION3",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=StubSpecialist([]),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=[[1, 0], [0, 0]],
                height=2,
                width=2,
                frame_signature=compute_board_signature([[1, 0], [0, 0]]),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-score-floor-suppressed"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(suppressed_actions=["ACTION1", "ACTION2", "ACTION3"])
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor executive fallback")
        self.assertNotEqual(decision.action_name, "ACTION6")

    def test_score_floor_fallback_resets_when_basin_is_exhausted_and_cycle_repeats(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION2",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION3",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=StubSpecialist([]),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_signature = "sig-score-floor-reset"
        actions = ("ACTION1", "ACTION2", "ACTION3", "ACTION1", "ACTION2", "ACTION3")
        for index, action_name in enumerate(actions):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["RESET", "ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["RESET", "ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor executive fallback",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=7,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["RESET", "ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle()
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor basin reset")
        self.assertEqual(decision.action_name, "RESET")

    def test_score_floor_fallback_uses_llm_memory_escape_after_emergency_replan_is_spent(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION4",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION6",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        board = [[0, 0, 0], [0, 8, 8], [0, 8, 8]]
        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION4", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="grid",
            ),
            derived_features={"frame_signature": "sig-score-floor-memory-escape"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION4", "ACTION6"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
                "ACTION4": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 3, "total": 4},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 3, "total": 3},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            trained_prior_bucket={"ACTION1": 1, "ACTION4": 9, "ACTION6": 0},
        )
        context.llm_memory_bundle = context.memory_bundle
        context.hyperon = FakeContextualHyperon()
        context.hyperon._current_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor memory escape")
        self.assertEqual(decision.action_name, "ACTION1")
        self.assertEqual(decision.subgoal, "replan")
        self.assertIn("score_floor_debug", decision.metadata)

    def test_score_floor_memory_escape_avoids_repeating_stalled_board_probe(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION4",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION6",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        board = [[0, 0, 0], [0, 8, 8], [0, 8, 8]]
        frame_signature = compute_board_signature(board)
        frames = [
            Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION6" if index else None,
                legal_actions=["ACTION1", "ACTION4", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=frame_signature,
                    source_attr="grid",
                ),
                derived_features={"frame_signature": frame_signature},
            )
            for index in range(4)
        ]
        for before, after in zip(frames, frames[1:]):
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION6",
                    source="fallback",
                    confidence=0.05,
                    reason="stalled board probe",
                    subgoal="inspect-active-board",
                    metadata={"x": 2, "y": 2},
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION4", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=frame_signature,
                source_attr="grid",
            ),
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION4", "ACTION6"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
                "ACTION4": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 3, "total": 4},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 3, "total": 3},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
            },
            trained_prior_bucket={"ACTION1": 1, "ACTION4": 9, "ACTION6": 0},
        )
        context.llm_memory_bundle = context.memory_bundle
        context.hyperon = FakeContextualHyperon()
        context.hyperon._current_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor memory escape")
        self.assertEqual(decision.action_name, "ACTION1")

    def test_score_floor_memory_escape_stays_within_score_floor_candidate_set(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION4",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        board = [[0, 0, 0], [0, 8, 8], [0, 8, 8]]
        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION4", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="grid",
            ),
            derived_features={"frame_signature": "sig-score-floor-constrained-memory-escape"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            recommended_actions=["ACTION6"],
            suppressed_actions=["ACTION1", "ACTION4", "ACTION6"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
            },
            trained_prior_bucket={"ACTION1": 1, "ACTION4": 1, "ACTION6": 9},
        )
        context.llm_memory_bundle = context.memory_bundle
        context.hyperon = FakeContextualHyperon()
        context.hyperon._current_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor memory escape")
        self.assertIn(decision.action_name, {"ACTION1", "ACTION4"})

    def test_score_floor_memory_escape_temporarily_excludes_recent_replan_fallback_actions(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal(
                        action_name="ACTION5",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION6",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                    ActionProposal(
                        action_name="ACTION7",
                        source="symbolic",
                        confidence=0.05,
                        reason="symbolic floor",
                    ),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frame_signature = "sig-temporary-basin-exclusion"
        for index, action_name in enumerate(("ACTION6", "ACTION7")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION7",
            legal_actions=["ACTION5", "ACTION6", "ACTION7"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION5", "ACTION6", "ACTION7"],
            action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            basin_action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            state_action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            frame_action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            trained_prior_bucket={"ACTION5": 5, "ACTION6": 1, "ACTION7": 1},
        )
        context.llm_memory_bundle = context.memory_bundle
        context.hyperon = FakeContextualHyperon()
        context.hyperon._current_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor memory escape")
        self.assertEqual(decision.action_name, "ACTION5")

    def test_score_floor_memory_escape_prefers_unseen_action_over_exhausted_basin_actions(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION5", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION6", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION7", "symbolic", 0.05, "symbolic floor"),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frame_signature = "sig-basin-break-exclusion"
        for index, action_name in enumerate(("ACTION6", "ACTION7")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION7",
            legal_actions=["ACTION5", "ACTION6", "ACTION7"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=[],
            action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            basin_action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            state_action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            frame_action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
        )
        context.llm_memory_bundle = context.memory_bundle
        context.hyperon = FakeContextualHyperon()
        context.hyperon._current_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor memory escape")
        self.assertEqual(decision.action_name, "ACTION5")

    def test_score_floor_memory_escape_uses_untried_replan_frontier_before_reopening_seen_actions(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION4", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION5", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION6", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION7", "symbolic", 0.05, "symbolic floor"),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frame_signature = "sig-replan-frontier"
        for index, action_name in enumerate(("ACTION6", "ACTION7")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION4", "ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION4", "ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION7",
            legal_actions=["ACTION4", "ACTION5", "ACTION6", "ACTION7"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=[],
            action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            basin_action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            state_action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            frame_action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            trained_prior_bucket={"ACTION4": 4, "ACTION5": 3, "ACTION6": 1, "ACTION7": 1},
        )
        context.llm_memory_bundle = context.memory_bundle
        context.hyperon = FakeContextualHyperon()
        context.hyperon._current_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.reason, "score-floor memory escape")
        self.assertIn(decision.action_name, {"ACTION4", "ACTION5"})

    def test_score_floor_frontier_exhausted_resets_when_reset_is_legal(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION4", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION5", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("RESET", "symbolic", 0.05, "symbolic floor"),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frame_signature = "sig-score-floor-frontier-reset"
        for index, action_name in enumerate(("ACTION4", "ACTION5")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION4", "ACTION5", "RESET"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION4", "ACTION5", "RESET"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION5",
            legal_actions=["ACTION4", "ACTION5", "RESET"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(basin_status={"is_exhausted": True})
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.action_name, "RESET")
        self.assertEqual(decision.reason, "score-floor frontier exhausted reset")

    def test_score_floor_frontier_reopen_resets_when_reset_is_legal(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION1", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION2", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("RESET", "symbolic", 0.05, "symbolic floor"),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frames = ("sig-a", "sig-b", "sig-a", "sig-b")
        actions = ("ACTION1", "ACTION2", "ACTION1", "ACTION2")
        for index, action_name in enumerate(actions):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "RESET"],
                derived_features={"frame_signature": frames[index]},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "RESET"],
                derived_features={"frame_signature": frames[(index + 1) % len(frames)]},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "RESET"],
            derived_features={"frame_signature": "sig-a"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(basin_status={"is_exhausted": True})
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.action_name, "RESET")
        self.assertEqual(decision.reason, "score-floor frontier reopen reset")

    def test_score_floor_frontier_reopen_resets_from_active_subset_when_dormant_candidates_exist(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION1", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION2", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION3", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION4", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("RESET", "symbolic", 0.05, "symbolic floor"),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frames = ("sig-a", "sig-b", "sig-c", "sig-d", "sig-e", "sig-f")
        actions = ("ACTION1", "ACTION2", "ACTION3", "ACTION1", "ACTION2", "ACTION3")
        for index, action_name in enumerate(actions):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"],
                derived_features={"frame_signature": frames[index]},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"],
                derived_features={"frame_signature": frames[(index + 1) % len(frames)]},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=7,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"],
            derived_features={"frame_signature": "sig-a"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(basin_status={"is_exhausted": True})
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.action_name, "RESET")
        self.assertEqual(decision.reason, "score-floor frontier reopen reset")
        self.assertIn("active_reopen", decision.metadata["score_floor_debug"])

    def test_pending_score_floor_frontier_reopen_resets_once_reset_becomes_legal(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubLLM:
            def __init__(self, memory_escape_proposals) -> None:
                self._memory_escape_proposals = list(memory_escape_proposals)

            def propose(self, context):
                return []

            def memory_escape_proposals(self, context, reason, preferred_actions=None):
                return list(self._memory_escape_proposals)

        class ScoreFloorCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        class PositiveCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = 0.5
                    rescored.append(proposal)
                return rescored

        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frames = ("sig-a", "sig-b", "sig-a", "sig-b")
        actions = ("ACTION1", "ACTION2", "ACTION1", "ACTION2")
        for index, action_name in enumerate(actions):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": frames[index]},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": frames[(index + 1) % len(frames)]},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)

        score_floor_executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION1", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION2", "symbolic", 0.05, "symbolic floor"),
                ]
            ),
            llm=StubLLM(
                [
                    ActionProposal(
                        "ACTION1",
                        "llm",
                        0.05,
                        "memory escape",
                        subgoal="replan",
                    )
                ]
            ),
            critic=ScoreFloorCritic(),
        )
        first_observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-a"},
        )
        first_context = build_context(
            first_observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )
        first_context.memory_bundle = MemoryBundle(basin_status={"is_exhausted": True})
        first_context.llm_memory_bundle = first_context.memory_bundle

        first_decision = score_floor_executive.decide(first_context)

        self.assertEqual(first_decision.reason, "score-floor memory escape")
        self.assertTrue(first_decision.metadata["score_floor_debug"]["pending_reset"])
        self.assertIsNotNone(memory.pending_score_floor_reopen(first_observation))

        second_observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "RESET"],
            derived_features={"frame_signature": "sig-b"},
        )
        memory.note_observation(second_observation)
        pending_reset_executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist(
                [ActionProposal("ACTION2", "reactive", 0.9, "normal reactive")]
            ),
            symbolic=StubSpecialist(
                [ActionProposal("ACTION1", "symbolic", 0.8, "normal symbolic")]
            ),
            llm=StubLLM([]),
            critic=PositiveCritic(),
        )
        second_context = build_context(
            second_observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )
        second_context.memory_bundle = MemoryBundle(basin_status={"is_exhausted": True})
        second_context.llm_memory_bundle = second_context.memory_bundle

        second_decision = pending_reset_executive.decide(second_context)

        self.assertEqual(second_decision.action_name, "RESET")
        self.assertEqual(second_decision.reason, "score-floor pending frontier reopen reset")
        self.assertIsNone(memory.pending_score_floor_reopen(second_observation))

    def test_score_floor_memory_escape_stops_when_replan_frontier_is_exhausted(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubCritic:
            def rescore(self, context, proposals):
                rescored = []
                for proposal in proposals:
                    proposal.legal = True
                    proposal.score = -1.0
                    rescored.append(proposal)
                return rescored

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION4", "symbolic", 0.05, "symbolic floor"),
                    ActionProposal("ACTION5", "symbolic", 0.05, "symbolic floor"),
                ]
            ),
            llm=LLMSpecialist(),
            critic=StubCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        memory.note_llm_call()
        memory.note_llm_call()
        memory.note_llm_call()
        frame_signature = "sig-score-floor-frontier-stop"
        for index, action_name in enumerate(("ACTION4", "ACTION5")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION4", "ACTION5"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION4", "ACTION5"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION5",
            legal_actions=["ACTION4", "ACTION5"],
            derived_features={"frame_signature": frame_signature},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))
        context.memory_bundle = MemoryBundle(basin_status={"is_exhausted": True})
        context.llm_memory_bundle = context.memory_bundle

        decision = executive.decide(context)

        self.assertEqual(decision.source, "fallback")
        self.assertNotEqual(decision.reason, "score-floor memory escape")

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

    def test_diverse_stalled_actions_enter_replan_before_default_threshold(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(
            ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION1", "ACTION2")
        ):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="diverse-stall-seed",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        llm_backend = FakeLLMBackend(
            {
                "subgoal": "replan",
                "confidence": 0.91,
                "reason": "diverse stalled llm",
                "action_candidates": [{"action": "ACTION4", "confidence": 0.91}],
            }
        )
        observation = Observation(
            game_id="ls20",
            step_index=7,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        )
        context = build_context(observation, memory, llm_backend)
        context.memory_bundle = MemoryBundle()

        decision = self.executive.decide(context)

        self.assertLess(memory.stagnation_steps, context.config.executive_replan_stagnation_threshold)
        self.assertEqual(llm_backend.calls, 1)
        self.assertEqual(decision.source, "llm")

    def test_replan_mode_rescoring_biases_llm_above_stale_symbolic_candidates(self) -> None:
        class StubSpecialist:
            def __init__(self, proposals) -> None:
                self._proposals = proposals

            def propose(self, context):
                return list(self._proposals)

        class StubLLM:
            def propose(self, context):
                return [
                    ActionProposal(
                        "ACTION4",
                        "llm",
                        0.30,
                        "llm replan escape",
                        subgoal="replan",
                    )
                ]

        executive = Executive(
            config=RuntimeConfig.from_env(),
            reactive=StubSpecialist([]),
            symbolic=StubSpecialist(
                [
                    ActionProposal("ACTION1", "symbolic", 0.34, "stale symbolic"),
                    ActionProposal("ACTION2", "symbolic", 0.33, "stale symbolic"),
                ]
            ),
            llm=StubLLM(),
            critic=ProposalCritic(),
        )
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(
            ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION1", "ACTION2")
        ):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.4,
                    reason="diverse-stall-seed",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=7,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )
        context.memory_bundle = MemoryBundle()

        decision = executive.decide(context)

        self.assertTrue(context.replan_mode)
        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.action_name, "ACTION4")

    def test_recent_score_floor_fallback_skips_immediate_followup_llm_replan(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
        )
        after = Observation(
            game_id="ls20",
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION3",
                source="fallback",
                confidence=0.05,
                reason="score-floor executive fallback",
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
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
        )
        context = build_context(observation, memory, llm_backend)
        context.memory_bundle = MemoryBundle()

        decision = self.executive.decide(context)

        self.assertEqual(llm_backend.calls, 0)
        self.assertNotEqual(decision.source, "llm")

    def test_llm_prompt_includes_deterministic_memory_bundle(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[1, 2], [3, 4]]
        image_input = BoardImageInput(
            board=board,
            height=2,
            width=2,
            frame_signature=compute_board_signature(board),
            source_attr="frame",
        )
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            image_input=image_input,
            derived_features={"frame_signature": "sig-prompt"},
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            image_input=image_input,
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
        self.assertIn("state_json=", llm_backend.last_prompt)
        self.assertIn('"basin_status"', llm_backend.last_prompt)
        self.assertIn('"recent_records"', llm_backend.last_prompt)
        self.assertIn('"basin_action_stats"', llm_backend.last_prompt)
        self.assertIn("If memory_basin_exhausted=true", llm_backend.last_prompt)
        self.assertIn('"height":2', llm_backend.last_prompt)
        self.assertIn('"width":2', llm_backend.last_prompt)
        self.assertIn('"active_cells":4', llm_backend.last_prompt)
        self.assertIn('"default_probe":{"x":1,"y":1}', llm_backend.last_prompt)
        self.assertIn('"source_attr":"frame"', llm_backend.last_prompt)
        self.assertIn('"matrix":[[1,2],[3,4]]', llm_backend.last_prompt)
        self.assertIn(
            "When board_active_cells>0 and ACTION6 is legal, use board_default_probe",
            llm_backend.last_prompt,
        )

    def test_symbolic_reason_mentions_board_summary_when_image_input_present(self) -> None:
        memory = EpisodicMemory()
        board = [[1, 1, 1], [2, 2, 2]]
        observation = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            image_input=BoardImageInput(
                board=board,
                height=2,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="board",
            ),
            derived_features={"frame_signature": "sig-symbolic-board"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload=None, available=False),
        )
        context.hyperon.sync(context.observation, context.memory, context.goals)

        proposals = self.executive.symbolic.propose(context)

        self.assertTrue(proposals)
        self.assertIn("board 3x2", proposals[0].reason)

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

        self.assertIn('"basin_exhausted":true', llm_backend.last_prompt)

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

    def test_critic_prefers_action6_targeting_active_board_cells(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[7, 0, 0], [0, 0, 0], [0, 0, 0]]
        observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-critic-action6-board"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION6",
                    source="llm",
                    confidence=0.60,
                    reason="active-cell target",
                    metadata={"x": 0, "y": 0},
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="llm",
                    confidence=0.60,
                    reason="empty target",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].metadata, {"x": 0, "y": 0})
        self.assertGreater(rescored[0].score, rescored[1].score)

    def test_critic_prefers_valid_board_probe_over_slightly_higher_confidence_generic_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 6, 6], [0, 6, 6]]
        observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-critic-board-probe-vs-generic"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.75,
                    reason="generic action",
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.58,
                    reason="board probe",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION6")
        self.assertGreater(rescored[0].score, rescored[1].score)

    def test_critic_prefers_canonical_board_probe_over_other_active_cell_targets(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-critic-canonical-board-probe"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.60,
                    reason="off-centroid active target",
                    metadata={"x": 1, "y": 1},
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.60,
                    reason="canonical centroid target",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].metadata, {"x": 2, "y": 2})
        self.assertGreater(rescored[0].score, rescored[1].score)

    def test_critic_prefers_board_probe_with_explicit_inspection_subgoal(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        observation = Observation(
            game_id="ls20",
            step_index=6,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-critic-board-probe-subgoal"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION6",
                    source="reactive",
                    confidence=0.60,
                    reason="generic board probe",
                    subgoal="maintain-control-loop",
                    metadata={"x": 2, "y": 2},
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="reactive",
                    confidence=0.60,
                    reason="inspection board probe",
                    subgoal="inspect-active-board",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].subgoal, "inspect-active-board")
        self.assertGreater(rescored[0].score, rescored[1].score)

    def test_critic_tiebreak_prefers_inspection_board_probe_over_equal_generic_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        observation = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-critic-tiebreak-board-probe"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION3",
                    source="symbolic",
                    confidence=0.60,
                    reason="generic action",
                    subgoal="inspect-active-board",
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.60,
                    reason="inspection board probe",
                    subgoal="inspect-active-board",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION6")

    def test_critic_opening_board_probe_bonus_prefers_action6_for_first_inspection_turn(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        observation = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-opening-board-probe"},
        )
        context = build_context(observation, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION3",
                    source="symbolic",
                    confidence=0.25,
                    reason="generic inspect opener",
                    subgoal="inspect-active-board",
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.20,
                    reason="board inspect opener",
                    subgoal="inspect-active-board",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION6")

    def test_critic_recent_board_probe_continuity_bonus_keeps_action6_above_generic_followup(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-post-probe-continuity"},
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-post-probe-continuity"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.70,
                reason="board probe",
                subgoal="inspect-active-board",
            )
        )
        memory.note_observation(after)
        context = build_context(after, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION3",
                    source="symbolic",
                    confidence=0.20,
                    reason="generic followup",
                    subgoal="break-stall",
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.14,
                    reason="board followup",
                    subgoal="break-stall",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION6")

    def test_critic_repeated_board_probe_exit_penalty_pushes_action6_below_generic_followup(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        first = Observation(
            game_id="ls20",
            step_index=0,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-post-probe-exit"},
        )
        second = Observation(
            game_id="ls20",
            step_index=1,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-post-probe-exit"},
        )
        third = Observation(
            game_id="ls20",
            step_index=2,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION3", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-post-probe-exit"},
        )
        memory.note_observation(first)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.70,
                reason="board probe",
                subgoal="inspect-active-board",
            )
        )
        memory.note_observation(second)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION6",
                source="symbolic",
                confidence=0.50,
                reason="board probe followup",
                subgoal="break-stall",
            )
        )
        memory.note_observation(third)
        context = build_context(third, memory, FakeLLMBackend(payload=None, available=False))

        rescored = self.executive.critic.rescore(
            context,
            [
                ActionProposal(
                    action_name="ACTION3",
                    source="symbolic",
                    confidence=0.20,
                    reason="generic followup",
                    subgoal="break-stall",
                ),
                ActionProposal(
                    action_name="ACTION6",
                    source="symbolic",
                    confidence=0.14,
                    reason="third board followup",
                    subgoal="break-stall",
                    metadata={"x": 2, "y": 2},
                ),
            ],
        )

        self.assertEqual(rescored[0].action_name, "ACTION3")

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
        self.assertIn('"recommended_actions":["ACTION2"]', llm_backend.last_prompt)
        self.assertIn('"current_subgoal_hint":"break-stall"', llm_backend.last_prompt)
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

    def test_llm_specialist_replaces_stale_emergency_candidates_with_memory_escape(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_signature = "sig-llm-emergency-stale"
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION1", "ACTION2")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor executive fallback",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": frame_signature},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(
                payload={
                    "subgoal": "replan",
                    "confidence": 0.9,
                    "reason": "emergency stale output",
                    "action_candidates": [
                        {"action": "ACTION1", "confidence": 0.9},
                        {"action": "ACTION2", "confidence": 0.85},
                    ],
                }
            ),
        )
        context.force_llm_replan = True
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            recommended_actions=["ACTION3"],
            suppressed_actions=["ACTION1", "ACTION2"],
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION3")
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

    def test_llm_specialist_fallback_prefers_board_probe_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 7, 7], [0, 7, 7]]
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-llm-board-probe"},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1"],
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

    def test_llm_specialist_fallback_preserves_board_inspection_subgoal_hint(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 7, 7], [0, 7, 7]]
        observation = Observation(
            game_id="ls20",
            step_index=5,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-llm-board-subgoal"},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.llm_subgoal_hint = "inspect-active-board"
        context.memory_bundle = MemoryBundle(
            suppressed_actions=["ACTION1"],
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

    def test_llm_specialist_fallback_assigns_adapter_metadata_for_complex_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 0, 6], [0, 0, 6]]
        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="grid",
            ),
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
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

    def test_llm_specialist_repairs_action6_metadata_to_board_target(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 9, 9], [0, 9, 9]]
        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="grid",
            ),
            derived_features={"frame_signature": "sig-llm-action6-repair"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(
                payload={
                    "action_candidates": [
                        {"action": "ACTION6", "confidence": 0.88, "x": 99, "y": 99}
                    ]
                }
            ),
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

    def test_llm_specialist_canonicalizes_noncanonical_active_action6_target(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 9, 9], [0, 9, 9]]
        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="grid",
            ),
            derived_features={"frame_signature": "sig-llm-action6-canonicalize"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(
                payload={
                    "action_candidates": [
                        {"action": "ACTION6", "confidence": 0.88, "x": 1, "y": 1}
                    ]
                }
            ),
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

    def test_llm_specialist_preserves_board_inspection_hint_for_structured_action6(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 9, 9], [0, 9, 9]]
        observation = Observation(
            game_id="ls20",
            step_index=4,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="grid",
            ),
            derived_features={"frame_signature": "sig-llm-action6-subgoal-hint"},
        )
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(
                payload={
                    "reason": "llm board probe",
                    "action_candidates": [
                        {"action": "ACTION6", "confidence": 0.88, "x": 2, "y": 2}
                    ],
                }
            ),
        )
        context.llm_subgoal_hint = "inspect-active-board"

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].subgoal, "inspect-active-board")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

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

    def test_llm_specialist_exhausted_fallback_prefers_board_probe_escape_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 8, 8], [0, 8, 8]]
        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION4", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=compute_board_signature(board),
                source_attr="grid",
            ),
            derived_features={"frame_signature": "sig-llm-broader-board-probe"},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION4", "ACTION6"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
                "ACTION4": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 3, "total": 4},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 3, "total": 3},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            trained_prior_bucket={"ACTION1": 1, "ACTION4": 9, "ACTION6": 0},
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION6")
        self.assertEqual(proposals[0].subgoal, "replan")
        self.assertEqual(proposals[0].metadata, {"x": 2, "y": 2})

    def test_llm_specialist_exhausted_fallback_avoids_stale_board_probe_escape_action(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        board = [[0, 0, 0], [0, 8, 8], [0, 8, 8]]
        frame_signature = compute_board_signature(board)
        frames = [
            Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION6" if index else None,
                legal_actions=["ACTION1", "ACTION4", "ACTION6"],
                image_input=BoardImageInput(
                    board=board,
                    height=3,
                    width=3,
                    frame_signature=frame_signature,
                    source_attr="grid",
                ),
                derived_features={"frame_signature": frame_signature},
            )
            for index in range(4)
        ]
        for before, after in zip(frames, frames[1:]):
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION6",
                    source="fallback",
                    confidence=0.05,
                    reason="stalled board probe",
                    subgoal="inspect-active-board",
                    metadata={"x": 2, "y": 2},
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=8,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION6",
            legal_actions=["ACTION1", "ACTION4", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=frame_signature,
                source_attr="grid",
            ),
            derived_features={"frame_signature": frame_signature},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION4", "ACTION6"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
                "ACTION4": {"progress": 2, "failed": 0, "stalled": 0, "total": 2},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 1, "stalled": 3, "total": 4},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 3, "total": 3},
                "ACTION4": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 4, "total": 4},
            },
            trained_prior_bucket={"ACTION1": 1, "ACTION4": 9, "ACTION6": 0},
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION4")

    def test_llm_specialist_exhausted_fallback_temporarily_excludes_recent_replan_escape_actions(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_signature = "sig-llm-temporary-exclusions"
        for index, action_name in enumerate(("ACTION6", "ACTION7")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)

        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION7",
            legal_actions=["ACTION5", "ACTION6", "ACTION7"],
            derived_features={"frame_signature": frame_signature},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION5", "ACTION6", "ACTION7"],
            action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            basin_action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            state_action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            frame_action_stats={
                "ACTION5": {"progress": 1, "failed": 0, "stalled": 0, "total": 1},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            trained_prior_bucket={"ACTION5": 5, "ACTION6": 1, "ACTION7": 1},
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION5")

    def test_llm_specialist_exhausted_fallback_breaks_recent_abab_replan_loop(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2", "ACTION1", "ACTION2")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": f"sig-loop-{index % 2}"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": f"sig-loop-{(index + 1) % 2}"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
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
            derived_features={"frame_signature": "sig-loop-1"},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION2", "ACTION3"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            basin_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 1, "total": 1},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 1, "total": 1},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION3")

    def test_llm_specialist_exhausted_fallback_breaks_recent_two_frame_replan_cycle(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_sequence = ("sig-a", "sig-b", "sig-a", "sig-b")
        action_sequence = ("ACTION1", "ACTION2", "ACTION1", "ACTION2")
        for index, action_name in enumerate(action_sequence):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_sequence[index]},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": frame_sequence[(index + 1) % len(frame_sequence)]},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
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
            derived_features={"frame_signature": "sig-a"},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=["ACTION1", "ACTION2", "ACTION3"],
            action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            basin_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            state_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
            frame_action_stats={
                "ACTION1": {"progress": 0, "failed": 0, "stalled": 1, "total": 1},
                "ACTION2": {"progress": 0, "failed": 0, "stalled": 1, "total": 1},
                "ACTION3": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
            },
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION3")

    def test_llm_specialist_exhausted_fallback_prefers_unseen_action_in_same_frame_basin(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_signature = "sig-llm-basin-break"
        for index, action_name in enumerate(("ACTION6", "ACTION7")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION7",
            legal_actions=["ACTION5", "ACTION6", "ACTION7"],
            derived_features={"frame_signature": frame_signature},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=[],
            action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            basin_action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            state_action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            frame_action_stats={
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action_name, "ACTION5")

    def test_llm_specialist_exhausted_fallback_prefers_untried_replan_frontier_actions(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_signature = "sig-llm-replan-frontier"
        for index, action_name in enumerate(("ACTION6", "ACTION7")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION4", "ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION4", "ACTION5", "ACTION6", "ACTION7"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION7",
            legal_actions=["ACTION4", "ACTION5", "ACTION6", "ACTION7"],
            derived_features={"frame_signature": frame_signature},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=[],
            action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            basin_action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            state_action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            frame_action_stats={
                "ACTION4": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION5": {"progress": 0, "failed": 0, "stalled": 0, "total": 0},
                "ACTION6": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
                "ACTION7": {"progress": 0, "failed": 0, "stalled": 2, "total": 2},
            },
            trained_prior_bucket={"ACTION4": 4, "ACTION5": 3, "ACTION6": 1, "ACTION7": 1},
        )

        proposals = LLMSpecialist().propose(context)

        self.assertTrue(proposals)
        self.assertIn(proposals[0].action_name, {"ACTION4", "ACTION5"})

    def test_llm_specialist_exhausted_fallback_returns_none_when_replan_frontier_is_exhausted(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        frame_signature = "sig-llm-replan-frontier-exhausted"
        for index, action_name in enumerate(("ACTION4", "ACTION5")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION4", "ACTION5"],
                derived_features={"frame_signature": frame_signature},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="ACTIVE",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION4", "ACTION5"],
                derived_features={"frame_signature": frame_signature},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.05,
                    reason="score-floor memory escape",
                    subgoal="replan",
                )
            )
            memory.note_observation(after)
        observation = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION5",
            legal_actions=["ACTION4", "ACTION5"],
            derived_features={"frame_signature": frame_signature},
        )
        memory.note_observation(observation)
        context = build_context(
            observation,
            memory,
            FakeLLMBackend(payload={"action_candidates": []}),
        )
        context.memory_bundle = MemoryBundle(
            basin_status={"is_exhausted": True},
            suppressed_actions=[],
        )

        proposals = LLMSpecialist().propose(context)

        self.assertEqual(proposals, [])
