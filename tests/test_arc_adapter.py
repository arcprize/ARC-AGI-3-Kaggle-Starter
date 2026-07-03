from __future__ import annotations

import unittest
from dataclasses import dataclass
from enum import Enum

from agi_core.adapters.arc import ArcAdapter
from agi_core.types import ActionDecision


class StubGameState(Enum):
    NOT_PLAYED = "not_played"
    GAME_OVER = "game_over"
    ACTIVE = "active"
    WIN = "win"


class StubGameAction(Enum):
    RESET = "reset"
    ACTION1 = "action1"
    ACTION2 = "action2"
    ACTION3 = "action3"
    ACTION4 = "action4"
    ACTION5 = "action5"
    ACTION6 = "action6"

    def is_complex(self) -> bool:
        return self is type(self).ACTION6

    def set_data(self, payload) -> None:
        self.payload = payload


@dataclass
class StubFrame:
    state: StubGameState
    levels_completed: int = 0
    last_action: StubGameAction | None = None
    available_actions: list[StubGameAction] | None = None
    step_index: int | None = None


class ArcAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ArcAdapter(
            game_action_cls=StubGameAction,
            default_game_id="ls20",
            deterministic_action="ACTION2",
        )

    def test_observe_normalizes_frame_state_and_actions(self) -> None:
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            levels_completed=3,
            last_action=StubGameAction.ACTION1,
            available_actions=[StubGameAction.ACTION2, StubGameAction.ACTION6],
            step_index=7,
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.game_id, "ls20")
        self.assertEqual(observation.game_state, "ACTIVE")
        self.assertEqual(observation.step_index, 7)
        self.assertEqual(observation.levels_completed, 3)
        self.assertEqual(observation.last_action, "ACTION1")
        self.assertEqual(observation.legal_actions, ["ACTION2", "ACTION6"])
        self.assertTrue(observation.derived_features["step_index_reliable"])

    def test_observe_normalizes_integer_action_ids(self) -> None:
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[1, 4, 6],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["ACTION1", "ACTION4", "ACTION6"])

    def test_reset_state_reports_reset_as_only_action(self) -> None:
        frame = StubFrame(state=StubGameState.GAME_OVER)
        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["RESET"])
        fallback = self.adapter.fallback_decision(observation)
        self.assertEqual(fallback.action_name, "RESET")

    def test_active_frame_without_explicit_actions_fails_closed(self) -> None:
        frame = StubFrame(state=StubGameState.ACTIVE)

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, [])
        self.assertFalse(observation.derived_features["legal_actions_known"])
        self.assertEqual(self.adapter.fallback_action_name(observation), "ACTION2")
        self.assertEqual(observation.step_index, 0)
        self.assertFalse(observation.derived_features["step_index_reliable"])

    def test_custom_reset_state_names_drive_legality_and_fallback(self) -> None:
        adapter = ArcAdapter(
            game_action_cls=StubGameAction,
            default_game_id="ls20",
            deterministic_action="ACTION2",
            reset_state_names=("READY", "LOSE"),
        )
        frame = StubFrame(state="lose")

        observation = adapter.observe([frame], frame)

        self.assertEqual(observation.game_state, "LOSE")
        self.assertEqual(observation.legal_actions, ["RESET"])
        self.assertEqual(adapter.fallback_action_name(observation), "RESET")

    def test_complex_action_maps_coordinates(self) -> None:
        decision = ActionDecision(
            action_name="ACTION6",
            source="llm",
            confidence=0.5,
            reason="click center",
            metadata={"x": 9, "y": 13},
        )

        action = self.adapter.to_game_action(decision)

        self.assertIs(action, StubGameAction.ACTION6)
        self.assertEqual(getattr(action, "payload", None), {"x": 9, "y": 13})
