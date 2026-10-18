import random
from typing import List
from arc import GameAction, GameState, FrameData

class MyAgent:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.step_idx = 0

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.step_idx = 0
            return GameAction.RESET
        
        self.step_idx += 1
        
        if self.game_id.split("-")[0] == "vc33":
            act = GameAction.ACTION1
            try:
                act.set_data({"x": 32, "y": 32})
            except:
                pass
            return act

        candidate_actions = [a for a in GameAction if a is not GameAction.RESET]
        action = random.choice(candidate_actions)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        return action
