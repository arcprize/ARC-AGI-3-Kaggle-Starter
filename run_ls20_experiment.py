"""
ARC-AGI-3 ls20 Hypothesis-Generate-Simulate-Prune Loop Experiment
==================================================================
This script implements and compares:
1. Hypothesis-Generate-Simulate-Prune Loop Policy
2. Random-Valid-Action Policy

Tested on ls20 (Levels 0 and 1+).
"""

from __future__ import annotations
import os
import sys
import json
import time
import random
import dataclasses
import numpy as np
from typing import Optional, Any, Callable, Dict, List, Tuple

import arc_agi

# Set path to local environment files
ENV_DIR = r"d:\AI_ARMY\arc_agi3_solver\datasets\arc-prize-2026-arc-agi-3\environment_files"
LOG_FILE = r"d:\AI_ARMY\arc_agi3_solver\ls20_experiment_log.jsonl"


# ============================================================================
# STEP 1: Hypothesis Data Structure & Template Library
# ============================================================================

@dataclasses.dataclass
class Hypothesis:
    description: str
    predictor_fn: Callable[[np.ndarray, int], Dict[Tuple[int, int], Tuple[int, int]]]
    confidence: float = 0.5
    source: str = "template"  # "template" or "generated"
    template_id: str = ""

    def predict(self, frame_grid: np.ndarray, action: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Predict expected diff: {(r, c): (old_val, new_val)}"""
        return self.predictor_fn(frame_grid, action)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "template_id": self.template_id,
        }


def extract_grid(frame_obj) -> np.ndarray:
    """Extract last 2D grid (H, W) from frame_obj.frame (handles multi-frame animations)."""
    arr = np.array(frame_obj.frame)
    while arr.ndim > 2:
        arr = arr[-1]
    return arr


def compute_actual_diff(grid_before: np.ndarray, grid_after: np.ndarray) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Compute dictionary of pixel differences between grid_before and grid_after."""
    g0 = extract_grid(grid_before) if hasattr(grid_before, 'frame') else (grid_before[-1] if grid_before.ndim > 2 else grid_before)
    g1 = extract_grid(grid_after) if hasattr(grid_after, 'frame') else (grid_after[-1] if grid_after.ndim > 2 else grid_after)
    diff_mask = g1 != g0
    diff_dict = {}
    rows, cols = np.where(diff_mask)
    for r, c in zip(rows, cols):
        diff_dict[(int(r), int(c))] = (int(g0[r, c]), int(g1[r, c]))
    return diff_dict


class TemplateLibrary:
    """Fixed template library for generating hypotheses deterministically."""

    def __init__(self):
        self.templates: List[Tuple[str, Callable[[int], Hypothesis]]] = []
        self._build_templates()
        self.next_index = 0

    def _build_templates(self):
        actions = [1, 2, 3, 4]
        
        # 1. No-op template: Action produces no change
        for a in actions:
            tid = f"noop_a{a}"
            desc = f"Action {a} is a no-op (produces zero grid diff)"
            def make_noop_predictor(act_target=a):
                def predictor(grid: np.ndarray, action: int):
                    if action == act_target:
                        return {}  # Expect no pixel change
                    return {}
                return predictor
            
            self.templates.append((tid, lambda a=a, tid=tid, desc=desc, fn=make_noop_predictor(a): Hypothesis(
                description=desc, predictor_fn=fn, confidence=0.5, source="template", template_id=tid
            )))

        # 2. Local Translation / Move Templates (up, down, left, right offset for player pixel)
        offsets = [( -1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
        for a in actions:
            for dr, dc, dir_name in offsets:
                tid = f"translate_a{a}_{dir_name}"
                desc = f"Action {a} moves entity/player {dir_name} by (dr={dr}, dc={dc})"
                def make_trans_predictor(act_target=a, dr=dr, dc=dc):
                    def predictor(grid: np.ndarray, action: int):
                        if action != act_target:
                            return {}
                        vals, counts = np.unique(grid, return_counts=True)
                        bg_color = vals[np.argmax(counts)]
                        entity_coords = np.argwhere(grid != bg_color)
                        diffs = {}
                        H, W = grid.shape
                        for r, c in entity_coords:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                diffs[(int(r), int(c))] = (int(grid[r, c]), int(bg_color))
                                diffs[(int(nr), int(nc))] = (int(grid[nr, nc]), int(grid[r, c]))
                        return diffs
                    return predictor
                
                self.templates.append((tid, lambda tid=tid, desc=desc, fn=make_trans_predictor(a, dr, dc): Hypothesis(
                    description=desc, predictor_fn=fn, confidence=0.5, source="template", template_id=tid
                )))

        # 3. Recolor / Color Swap Templates
        for a in actions:
            for target_color in range(10):
                tid = f"recolor_a{a}_c{target_color}"
                desc = f"Action {a} recolors focal entity to color {target_color}"
                def make_recolor_predictor(act_target=a, tc=target_color):
                    def predictor(grid: np.ndarray, action: int):
                        if action != act_target:
                            return {}
                        vals, counts = np.unique(grid, return_counts=True)
                        bg_color = vals[np.argmax(counts)]
                        entity_coords = np.argwhere(grid != bg_color)
                        diffs = {}
                        for r, c in entity_coords:
                            if grid[r, c] != tc:
                                diffs[(int(r), int(c))] = (int(grid[r, c]), int(tc))
                        return diffs
                    return predictor
                
                self.templates.append((tid, lambda tid=tid, desc=desc, fn=make_recolor_predictor(a, target_color): Hypothesis(
                    description=desc, predictor_fn=fn, confidence=0.5, source="template", template_id=tid
                )))

        # 4. Toggle / Invert Templates
        for a in actions:
            tid = f"toggle_a{a}"
            desc = f"Action {a} toggles active pixel state"
            def make_toggle_predictor(act_target=a):
                def predictor(grid: np.ndarray, action: int):
                    if action != act_target:
                        return {}
                    vals, counts = np.unique(grid, return_counts=True)
                    bg_color = vals[np.argmax(counts)]
                    entity_coords = np.argwhere(grid != bg_color)
                    diffs = {}
                    for r, c in entity_coords[:5]:
                        diffs[(int(r), int(c))] = (int(grid[r, c]), int(bg_color))
                    return diffs
                return predictor
            
            self.templates.append((tid, lambda tid=tid, desc=desc, fn=make_toggle_predictor(a): Hypothesis(
                description=desc, predictor_fn=fn, confidence=0.5, source="template", template_id=tid
            )))

    def get_next_hypothesis(self, used_template_ids: set) -> Optional[Hypothesis]:
        """Return next template-generated hypothesis not yet used in deterministic order."""
        start_idx = self.next_index
        while True:
            if self.next_index >= len(self.templates):
                self.next_index = 0
            
            tid, factory = self.templates[self.next_index]
            self.next_index += 1
            if tid not in used_template_ids:
                used_template_ids.add(tid)
                return factory()
            
            if self.next_index == start_idx:
                break
        return None


# ============================================================================
# STEP 1 & 2: Hypothesis Loop Policy & Logging
# ============================================================================

class HypothesisLoopPolicy:
    def __init__(self, kill_threshold: float = 0.1):
        self.kill_threshold = kill_threshold
        self.template_library = TemplateLibrary()
        self.live_hypotheses: List[Hypothesis] = []
        self.used_template_ids: set = set()
        self.unexplored_actions: Dict[str, set] = {}

        # Seed initial hypotheses from template library
        for _ in range(4):
            h = self.template_library.get_next_hypothesis(self.used_template_ids)
            if h:
                self.live_hypotheses.append(h)

    def _grid_hash(self, grid: np.ndarray) -> str:
        return str(hash(grid.tobytes()))

    def choose_action(self, current_grid: np.ndarray, available_actions: List[int]) -> Tuple[int, Optional[Hypothesis]]:
        alive = [h for h in self.live_hypotheses if h.confidence >= self.kill_threshold]
        alive.sort(key=lambda h: h.confidence, reverse=True)

        selected_hypothesis = None
        chosen_action = None

        for h in alive:
            for a in available_actions:
                pred = h.predict(current_grid, a)
                if pred:
                    chosen_action = a
                    selected_hypothesis = h
                    break
            if chosen_action is not None:
                break

        if chosen_action is None and alive:
            selected_hypothesis = alive[0]

        if chosen_action is None:
            gh = self._grid_hash(current_grid)
            if gh not in self.unexplored_actions:
                self.unexplored_actions[gh] = set(available_actions)
            
            unexplored = self.unexplored_actions[gh]
            if unexplored:
                chosen_action = sorted(list(unexplored))[0]
                unexplored.remove(chosen_action)
            else:
                chosen_action = available_actions[0]

        return chosen_action, selected_hypothesis

    def update_and_prune(
        self,
        grid_before: np.ndarray,
        action: int,
        grid_after: np.ndarray,
        chosen_hypothesis: Optional[Hypothesis],
        step_idx: int,
        level_idx: int,
        log_file_handle
    ):
        actual_diff = compute_actual_diff(grid_before, grid_after)

        for h in list(self.live_hypotheses):
            pred_diff = h.predict(grid_before, action)
            matched = (pred_diff == actual_diff)

            if matched:
                h.confidence = min(1.0, h.confidence + 0.15)
            else:
                h.confidence -= 0.25

        self.live_hypotheses = [h for h in self.live_hypotheses if h.confidence >= self.kill_threshold]

        if len(self.live_hypotheses) == 0:
            new_h = self.template_library.get_next_hypothesis(self.used_template_ids)
            if new_h:
                new_h.source = "generated"
                self.live_hypotheses.append(new_h)

        used_hypo_dict = chosen_hypothesis.to_dict() if chosen_hypothesis else {
            "description": "Unexplored action exploration",
            "source": "exploration",
            "confidence": 0.0,
            "template_id": "none"
        }
        
        pred_diff_str = str(chosen_hypothesis.predict(grid_before, action)) if chosen_hypothesis else "{}"
        
        log_entry = {
            "step": step_idx,
            "level": level_idx,
            "action_taken": action,
            "hypothesis_used": used_hypo_dict,
            "predicted_diff": pred_diff_str[:150],
            "actual_diff": str(actual_diff)[:150],
            "matched": bool(chosen_hypothesis and chosen_hypothesis.predict(grid_before, action) == actual_diff),
            "confidence_after": round(chosen_hypothesis.confidence, 4) if chosen_hypothesis else 0.0,
            "hypotheses_alive_count": len(self.live_hypotheses),
        }

        log_file_handle.write(json.dumps(log_entry) + "\n")
        log_file_handle.flush()


# ============================================================================
# STEP 3: Experiment Runner & Comparison
# ============================================================================

def run_hypothesis_agent(arcade: arc_agi.Arcade, max_steps_per_level: int = 150) -> Dict[int, int]:
    env = arcade.make("ls20")
    frame = env.reset()
    policy = HypothesisLoopPolicy(kill_threshold=0.1)

    level_actions: Dict[int, int] = {}
    current_level = 0
    step_count = 0
    level_start_step = 0

    with open(LOG_FILE, "w") as log_file:
        while current_level < 2:
            state_str = str(frame.state)
            if "GAME_OVER" in state_str:
                print(f"[HypothesisPolicy] Game Over on Level {current_level}. Resetting level...")
                frame = env.reset()
                continue

            grid_before = extract_grid(frame)
            actions = frame.available_actions
            if not actions:
                actions = [1, 2, 3, 4]
            
            action, chosen_hypo = policy.choose_action(grid_before, actions)
            
            frame_after = env.step(action)
            grid_after = extract_grid(frame_after)
            
            step_count += 1

            policy.update_and_prune(
                grid_before, action, grid_after, chosen_hypo,
                step_count, current_level, log_file
            )

            state_after_str = str(frame_after.state)
            if frame_after.levels_completed > current_level or "WIN" in state_after_str:
                acts_used = step_count - level_start_step
                level_actions[current_level] = acts_used
                print(f"[HypothesisPolicy] Completed Level {current_level} in {acts_used} actions!")
                current_level = frame_after.levels_completed
                level_start_step = step_count
                if current_level >= 2:
                    break

            if (step_count - level_start_step) >= max_steps_per_level:
                print(f"[HypothesisPolicy] Reached max step budget ({max_steps_per_level}) for level {current_level}")
                level_actions[current_level] = max_steps_per_level
                break

            frame = frame_after

    return level_actions


def run_random_agent(arcade: arc_agi.Arcade, max_steps_per_level: int = 150) -> Dict[int, int]:
    env = arcade.make("ls20")
    frame = env.reset()
    random.seed(42)

    level_actions: Dict[int, int] = {}
    current_level = 0
    step_count = 0
    level_start_step = 0

    while current_level < 2:
        state_str = str(frame.state)
        if "GAME_OVER" in state_str:
            print(f"[RandomPolicy] Game Over on Level {current_level}. Resetting level...")
            frame = env.reset()
            continue

        actions = frame.available_actions
        if not actions:
            actions = [1, 2, 3, 4]
        action = random.choice(actions)
        
        frame_after = env.step(action)
        step_count += 1

        state_after_str = str(frame_after.state)
        if frame_after.levels_completed > current_level or "WIN" in state_after_str:
            acts_used = step_count - level_start_step
            level_actions[current_level] = acts_used
            print(f"[RandomPolicy] Completed Level {current_level} in {acts_used} actions!")
            current_level = frame_after.levels_completed
            level_start_step = step_count
            if current_level >= 2:
                break

        if (step_count - level_start_step) >= max_steps_per_level:
            print(f"[RandomPolicy] Reached max step budget ({max_steps_per_level}) for level {current_level}")
            level_actions[current_level] = max_steps_per_level
            break

        frame = frame_after

    return level_actions


def main():
    print("==========================================================================")
    print("  ARC-AGI-3 ls20 HYPOTHESIS LOOP vs RANDOM BASELINE EXPERIMENT")
    print("==========================================================================")

    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR)
    
    print("\n--- Running Hypothesis-Generate-Simulate-Prune Loop Policy ---")
    hypo_results = run_hypothesis_agent(arcade, max_steps_per_level=150)

    print("\n--- Running Random-Valid-Action Policy Baseline ---")
    random_results = run_random_agent(arcade, max_steps_per_level=150)

    print("\n==========================================================================")
    print("                      RESULTS COMPARISON TABLE                            ")
    print("==========================================================================")
    print(f"{'Level':<10} | {'Hypothesis-Loop Policy':<25} | {'Random Policy':<20} | {'Efficiency Gain':<15}")
    print("-" * 78)

    all_levels = sorted(list(set(list(hypo_results.keys()) + list(random_results.keys()))))
    for lvl in all_levels:
        h_acts = hypo_results.get(lvl, "N/A")
        r_acts = random_results.get(lvl, "N/A")
        if isinstance(h_acts, int) and isinstance(r_acts, int):
            gain = f"{((r_acts - h_acts) / r_acts) * 100:.1f}% faster" if r_acts != 0 else "N/A"
        else:
            gain = "N/A"
        print(f"{lvl:<10} | {str(h_acts) + ' actions':<25} | {str(r_acts) + ' actions':<20} | {gain:<15}")

    print("==========================================================================")
    print(f"Log file saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
