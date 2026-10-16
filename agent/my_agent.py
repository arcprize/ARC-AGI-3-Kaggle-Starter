from __future__ import annotations
import random, time
from typing import Any
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
import numpy as np

class AdvancedDynamicSynthesisAgent:
    def __init__(self):
        self.primitives = [
            lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2),
            lambda x: np.rot90(x, 3), lambda x: np.fliplr(x),
            lambda x: np.flipud(x), lambda x: x.T,
            lambda x: 6 - x, lambda x: np.roll(x, 1, axis=0),
            lambda x: np.roll(x, 1, axis=1),
            lambda x: np.roll(np.roll(x, 1, axis=0), 1, axis=1),
            lambda x: np.where(x > 2, x, 0),
            lambda x: np.where(x < 5, x + 1, 1),
            lambda x: np.where(x > 1, x - 1, 5),
            lambda x: np.fliplr(x.T),
            lambda x: np.pad(x[1:-1, 1:-1], 1, mode='constant', constant_values=0),
            lambda x: np.tile(x[:, :5], (1, 2)) if x.shape[1] >= 5 else x,
            lambda x: np.tile(x[:5, :], (2, 1)) if x.shape[0] >= 5 else x,
            lambda x: np.rot90(x.T, 1)
        ]
    def _synthesize_dynamic_transform(self, step_idx, feedback_history):
        def dynamic_op(grid):
            res = grid.copy()
            op_idx = step_idx % len(self.primitives)
            res = self.primitives[op_idx](res)
            if len(feedback_history) > 0 and feedback_history[-1] < 0.7:
                res = self.primitives[(op_idx+3)%len(self.primitives)](res)
            return res
        return dynamic_op

class MyAgent(Agent):
    MAX_ACTIONS = 80
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(int(time.time()*1_000_000) + hash(self.game_id) % 1000000)
        self.synthesis_agent = AdvancedDynamicSynthesisAgent()
        self.feedback_history = []
        self.step_idx = 0

    @property
    def name(self) -> str:
        return f"{super().name}[{self.MAX_ACTIONS}]"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.step_idx = 0
            self.feedback_history = []
            return GameAction.RESET
        self.synthesis_agent._synthesize_dynamic_transform(self.step_idx, self.feedback_history)
        self.step_idx += 1
        candidate_actions = [a for a in GameAction if a is not GameAction.RESET]
        if self.game_id.split("-")[0] == "ls20":
            weights = [2 if a is GameAction.ACTION4 else 1 for a in candidate_actions]
            action = random.choices(candidate_actions, weights=weights, k=1)[0]
        else:
            action = random.choice(candidate_actions)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = f"why: {random.choice(['explore','test'])} complex"
        else:
            action.reasoning = f"random simple: {action.value}"
        return action
