from __future__ import annotations

import importlib
import io
import os
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum
from contextlib import redirect_stdout
from unittest.mock import patch

from agi_core.types import MemoryBundle


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
class StubFrameData:
    state: StubGameState
    levels_completed: int = 0
    last_action: StubGameAction | None = None
    available_actions: list[StubGameAction] | None = None


class StubAgent:
    def __init__(self, *args, **kwargs) -> None:
        self.game_id = kwargs.get("game_id", "ls20")
        self._name = kwargs.get("agent_name", "StubAgent")

    @property
    def name(self) -> str:
        return self._name


class FakeLlamaBackend:
    def __init__(self, *args, **kwargs) -> None:
        self.available = False
        self.last_error = None


class FakeOpenAIBackend(FakeLlamaBackend):
    pass


class FakeHyperonBridge:
    def __init__(self, *args, **kwargs) -> None:
        self.available = False
        self.last_error = None
        self._bundle = MemoryBundle()
        self.recorded_transitions = []

    def sync(self, observation, memory, goals) -> None:
        return None

    def current_memory_bundle(self):
        return self._bundle

    def record_transition(self, record) -> None:
        self.recorded_transitions.append(record)

    def run_rules(self, query: str):
        return []

    def update_proposals(self, proposals) -> None:
        return None


class MyAgentIntegrationTests(unittest.TestCase):
    def test_my_agent_uses_executive_runtime(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            module = importlib.import_module("agent.my_agent")
            module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")

                    reset_frame = StubFrameData(
                        state=StubGameState.NOT_PLAYED,
                        available_actions=[StubGameAction.RESET],
                    )
                    reset_action = agent.choose_action([reset_frame], reset_frame)
                    self.assertIs(reset_action, StubGameAction.RESET)

                    active_frame = StubFrameData(
                        state=StubGameState.ACTIVE,
                        levels_completed=0,
                        last_action=StubGameAction.ACTION1,
                        available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                    )
                    active_action = agent.choose_action([active_frame], active_frame)
                    self.assertEqual(active_action.name, "ACTION1")

    def test_my_agent_treats_explicit_empty_actions_as_known_empty_legality(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            module = importlib.import_module("agent.my_agent")
            module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    active_frame = StubFrameData(
                        state=StubGameState.ACTIVE,
                        levels_completed=0,
                        last_action=StubGameAction.ACTION1,
                        available_actions=[],
                    )

                    decision = agent.runtime.decide([active_frame], active_frame)
                    action = agent.runtime.adapter.to_game_action(decision)

                    self.assertEqual(decision.reason, "fail-closed empty legality reset")
                    self.assertIs(action, StubGameAction.RESET)

    def test_my_agent_reset_state_with_explicit_empty_actions_still_reports_reset_legality(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            module = importlib.import_module("agent.my_agent")
            module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    reset_frame = StubFrameData(
                        state=StubGameState.GAME_OVER,
                        available_actions=[],
                    )

                    observation = agent.runtime.adapter.observe([reset_frame], reset_frame)
                    action = agent.choose_action([reset_frame], reset_frame)

                    self.assertEqual(observation.legal_actions, ["RESET"])
                    self.assertTrue(observation.derived_features["legal_actions_known"])
                    self.assertIs(action, StubGameAction.RESET)

    def test_my_agent_reset_state_ignores_supported_non_reset_explicit_actions(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            module = importlib.import_module("agent.my_agent")
            module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    reset_frame = StubFrameData(
                        state=StubGameState.NOT_PLAYED,
                        available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                    )

                    observation = agent.runtime.adapter.observe([reset_frame], reset_frame)
                    action = agent.choose_action([reset_frame], reset_frame)

                    self.assertEqual(observation.legal_actions, ["RESET"])
                    self.assertTrue(observation.derived_features["legal_actions_known"])
                    self.assertIs(action, StubGameAction.RESET)

    def test_my_agent_can_emit_debug_decision_trace(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            with patch.dict(os.environ, {"AGI_DEBUG_DECISIONS": "1"}, clear=False):
                module = importlib.import_module("agent.my_agent")
                module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    active_frame = StubFrameData(
                        state=StubGameState.ACTIVE,
                        levels_completed=0,
                        last_action=StubGameAction.ACTION1,
                        available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                    )
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        agent.choose_action([active_frame], active_frame)
                    self.assertIn("decision action=", buffer.getvalue())

    def test_my_agent_debug_trace_can_include_score_floor_metadata(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            with patch.dict(os.environ, {"AGI_DEBUG_DECISIONS": "1"}, clear=False):
                module = importlib.import_module("agent.my_agent")
                module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        agent._log_decision(
                            types.SimpleNamespace(
                                action_name="ACTION1",
                                source="fallback",
                                confidence=0.05,
                                subgoal="replan",
                                reason="score-floor memory escape",
                                proposals=[],
                                metadata={
                                    "score_floor_debug": {
                                        "is_reopened": False,
                                        "recent_match_count": 6,
                                        "candidate_actions": ["ACTION1", "ACTION2", "ACTION3"],
                                        "frame_signatures": ["sig-a", "sig-b"],
                                        "action_counts": {"ACTION1": 3, "ACTION2": 2, "ACTION3": 1},
                                    }
                                },
                            )
                        )
                    trace = buffer.getvalue()
                    self.assertIn("meta[score_floor", trace)
                    self.assertIn("matches=6", trace)
                    self.assertIn("frames=sig-a,sig-b", trace)

    def test_my_agent_debug_trace_can_include_probe_target_metadata(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            with patch.dict(os.environ, {"AGI_DEBUG_DECISIONS": "1"}, clear=False):
                module = importlib.import_module("agent.my_agent")
                module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        agent._log_decision(
                            types.SimpleNamespace(
                                action_name="ACTION6",
                                source="fallback",
                                confidence=0.05,
                                subgoal="inspect-active-board",
                                reason="score-floor executive fallback",
                                proposals=[],
                                metadata={"x": 2, "y": 1},
                            )
                        )
                    trace = buffer.getvalue()
                    self.assertIn("meta[target=2,1]", trace)

    def test_my_agent_debug_trace_can_include_board_context_metadata(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            with patch.dict(os.environ, {"AGI_DEBUG_DECISIONS": "1"}, clear=False):
                module = importlib.import_module("agent.my_agent")
                module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    with patch.object(
                        agent.runtime.memory,
                        "summarize_for_hyperon",
                        return_value={
                            "current_board_context": {
                                "board_height": 64,
                                "board_width": 64,
                                "board_active_cells": 0,
                                "board_source_attr": "frame",
                                "board_default_probe": {"x": 32, "y": 32},
                            }
                        },
                    ):
                        buffer = io.StringIO()
                        with redirect_stdout(buffer):
                            agent._log_decision(
                                types.SimpleNamespace(
                                    action_name="ACTION6",
                                    source="symbolic",
                                    confidence=0.05,
                                    subgoal="inspect-active-board",
                                    reason="symbolic board inspection target",
                                    proposals=[],
                                    metadata={"x": 32, "y": 32},
                                )
                            )
                        trace = buffer.getvalue()
                        self.assertIn("meta[board size=64x64 active=0 source=frame default_probe=32,32]", trace)

    def test_my_agent_debug_trace_flushes_output(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            with patch.dict(os.environ, {"AGI_DEBUG_DECISIONS": "1"}, clear=False):
                module = importlib.import_module("agent.my_agent")
                module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")
                    with patch("builtins.print") as mock_print:
                        agent._log_decision(
                            types.SimpleNamespace(
                                action_name="ACTION1",
                                source="fallback",
                                confidence=0.05,
                                subgoal="replan",
                                reason="score-floor memory escape",
                                proposals=[],
                                metadata={},
                            )
                        )
                    self.assertTrue(mock_print.called)
                    self.assertTrue(mock_print.call_args.kwargs.get("flush"))

    def test_my_agent_flushes_terminal_win_transition_into_runtime_memory(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            module = importlib.import_module("agent.my_agent")
            module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")

                    active_frame = StubFrameData(
                        state=StubGameState.ACTIVE,
                        levels_completed=0,
                        last_action=StubGameAction.ACTION1,
                        available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                    )
                    agent.choose_action([active_frame], active_frame)

                    win_frame = StubFrameData(
                        state=StubGameState.WIN,
                        levels_completed=1,
                        last_action=StubGameAction.ACTION1,
                        available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                    )
                    self.assertTrue(agent.is_done([active_frame, win_frame], win_frame))
                    self.assertEqual(len(agent.runtime.hyperon.recorded_transitions), 1)
                    record = agent.runtime.hyperon.recorded_transitions[0]
                    self.assertEqual(record.outcome, "progress")
                    self.assertEqual(record.next_game_state, "WIN")

    def test_my_agent_honors_custom_configured_win_state(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            with patch.dict(os.environ, {"AGI_WIN_STATE_NAME": "DONE"}, clear=False):
                module = importlib.import_module("agent.my_agent")
                module = importlib.reload(module)
                with patch.object(module, "build_llm_backend", return_value=FakeLlamaBackend()):
                    with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                        agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")

                        active_frame = StubFrameData(
                            state=StubGameState.ACTIVE,
                            levels_completed=0,
                            last_action=StubGameAction.ACTION1,
                            available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                        )
                        agent.choose_action([active_frame], active_frame)

                        done_frame = StubFrameData(
                            state="done",
                            levels_completed=1,
                            last_action=StubGameAction.ACTION1,
                            available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                        )
                        self.assertTrue(agent.is_done([active_frame, done_frame], done_frame))
                        self.assertEqual(len(agent.runtime.hyperon.recorded_transitions), 1)
                        record = agent.runtime.hyperon.recorded_transitions[0]
                        self.assertEqual(record.outcome, "progress")
                        self.assertEqual(record.next_game_state, "DONE")

    def test_my_agent_can_use_openai_backend_without_runtime_changes(self) -> None:
        arcengine_module = types.ModuleType("arcengine")
        arcengine_module.FrameData = StubFrameData
        arcengine_module.GameAction = StubGameAction
        arcengine_module.GameState = StubGameState

        agents_module = types.ModuleType("agents")
        agents_agent_module = types.ModuleType("agents.agent")
        agents_agent_module.Agent = StubAgent

        module_overrides = {
            "arcengine": arcengine_module,
            "agents": agents_module,
            "agents.agent": agents_agent_module,
        }

        with patch.dict(sys.modules, module_overrides, clear=False):
            sys.modules.pop("agent.my_agent", None)
            module = importlib.import_module("agent.my_agent")
            module = importlib.reload(module)
            with patch.object(module, "build_llm_backend", return_value=FakeOpenAIBackend()):
                with patch.object(module, "HyperonBridge", FakeHyperonBridge):
                    agent = module.MyAgent(game_id="ls20", agent_name="StubAgent")

                    active_frame = StubFrameData(
                        state=StubGameState.ACTIVE,
                        levels_completed=0,
                        last_action=StubGameAction.ACTION1,
                        available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
                    )
                    action = agent.choose_action([active_frame], active_frame)
                    self.assertEqual(action.name, "ACTION1")
