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


class FakeHyperonBridge:
    def __init__(self, *args, **kwargs) -> None:
        self.available = False
        self.last_error = None

    def sync(self, observation, memory, goals) -> None:
        return None

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
            with patch.object(module, "LlamaCppBackend", FakeLlamaBackend):
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
            with patch.object(module, "LlamaCppBackend", FakeLlamaBackend):
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
