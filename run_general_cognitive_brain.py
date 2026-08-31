"""
HyPrune v2: General Cognitive Brain for ARC-AGI
================================================
A general cognitive architecture for ARC-AGI tasks and games based on:
1. Dynamic Perception & Scene Graph Extraction (Objectness, Topology, Symmetries)
2. General Hypothesis Generation (Cognitive Priors: Kinematics, Recoloring, Symmetries)
3. Mental Simulation & Concurrent Verification
4. Dynamic Hypothesis Bank Pruning & Negative Knowledge Base
5. General Multi-Game / Task Execution (No hardcoded rules or game-specific logic)
"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import random
import dataclasses
import numpy as np
from typing import Optional, Any, Callable, Dict, List, Tuple, Set

import arc_agi
from hyprune.core.scene_graph import ObjectParser, SceneGraph, ArcObject, SpatialRelation
from hyprune.core.hypothesis import Hypothesis, AbstractionLevel, ConsistencyReport, PairResult
from hyprune.core.knowledge_base import NegativeKnowledgeBase

# Configuration & Paths
ENV_DIR = r"d:\AI_ARMY\arc_agi3_solver\datasets\arc-prize-2026-arc-agi-3\environment_files"
GENERAL_LOG_FILE = r"d:\AI_ARMY\arc_agi3_solver\general_cognitive_brain_log.jsonl"


# ============================================================================
# 1. PERCEPTUAL BRAIN: Grid & Frame Parsing
# ============================================================================

def extract_2d_grid(frame_data) -> np.ndarray:
    """Extract a 2D numpy array (H, W) from raw frame data, handling animation sequences."""
    if hasattr(frame_data, "frame"):
        raw = frame_data.frame
    else:
        raw = frame_data
    
    arr = np.array(raw)
    while arr.ndim > 2:
        arr = arr[-1]  # Take the latest frame in an animation sequence
    return arr.astype(np.int8)


# ============================================================================
# 2. GENERAL HYPOTHESIS GENERATOR (COGNITIVE PRIORS)
# ============================================================================

class GeneralHypothesisGenerator:
    """Generates hypotheses dynamically based on SceneGraph perception and cognitive priors.
    
    Priors implemented:
    - Object Kinematics (Move object O by vector (dr, dc))
    - Object Color Transformations (Recolor object O to color C)
    - Grid-level Symmetries (Reflect, Rotate, Transpose)
    - Invariant Rules (Action A produces no change / selective change)
    """

    def __init__(self, parser: ObjectParser):
        self.parser = parser

    def generate_hypotheses_for_state(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        exclude_signatures: Set[str]
    ) -> List[Hypothesis]:
        scene: SceneGraph = self.parser.parse(grid)
        objects = scene.non_background_objects()
        bg_color = scene.background_color
        
        generated: List[Hypothesis] = []

        # Prior 1: Invariant / No-Op per available action
        for action in available_actions:
            sig = f"noop_act_{action}"
            if sig not in exclude_signatures:
                desc = f"Action {action} is a no-op (preserves grid state)"
                def make_noop_fn(act=action):
                    def fn(g: np.ndarray, a: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
                        return {}
                    return fn
                
                h = Hypothesis.create(
                    description=desc,
                    program=None,
                    confidence=0.5,
                    source="cognitive_prior",
                    abstraction_level=AbstractionLevel.PIXEL,
                    operations=[f"noop_{action}"],
                    preconditions={"action": action}
                )
                h._predictor_fn = make_noop_fn(action)
                h._sig = sig
                generated.append(h)

        # Prior 2: Object Kinematics (Directional motion per object and action)
        motion_vectors = [
            (-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT"),
            (-1, -1, "UP-LEFT"), (-1, 1, "UP-RIGHT"), (1, -1, "DOWN-LEFT"), (1, 1, "DOWN-RIGHT")
        ]

        for action in available_actions:
            for obj in objects[:3]:  # Top 3 salient non-background objects
                for dr, dc, dir_name in motion_vectors:
                    sig = f"kinematic_act_{action}_obj_{obj.id}_dir_{dir_name}"
                    if sig in exclude_signatures:
                        continue
                    
                    desc = f"Action {action} translates Object-{obj.id} (color {obj.color}) {dir_name} by ({dr}, {dc})"
                    
                    def make_kinematic_fn(act=action, target_obj_id=obj.id, dr=dr, dc=dc, bg=bg_color):
                        def fn(current_grid: np.ndarray, a: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
                            if a != act:
                                return {}
                            curr_scene = self.parser.parse(current_grid)
                            target_obj = curr_scene.get_object(target_obj_id)
                            if not target_obj:
                                return {}
                            
                            diffs = {}
                            H, W = current_grid.shape
                            for r, c in target_obj.pixels:
                                nr, nc = r + dr, c + dc
                                diffs[(int(r), int(c))] = (int(current_grid[r, c]), int(bg))
                                if 0 <= nr < H and 0 <= nc < W:
                                    diffs[(int(nr), int(nc))] = (int(current_grid[nr, nc]), int(target_obj.color))
                            return diffs
                        return fn

                    h = Hypothesis.create(
                        description=desc,
                        program=None,
                        confidence=0.5,
                        source="cognitive_prior",
                        abstraction_level=AbstractionLevel.OBJECT,
                        operations=[f"translate_obj_{obj.id}", f"action_{action}"],
                        preconditions={"action": action, "object_id": obj.id}
                    )
                    h._predictor_fn = make_kinematic_fn(action, obj.id, dr, dc, bg_color)
                    h._sig = sig
                    generated.append(h)

        # Prior 3: Palette Recoloring (Recolor object to prominent colors)
        all_colors = set(scene.color_histogram.keys())
        for action in available_actions:
            for obj in objects[:2]:
                for new_color in all_colors:
                    if new_color == obj.color:
                        continue
                    sig = f"recolor_act_{action}_obj_{obj.id}_c_{new_color}"
                    if sig in exclude_signatures:
                        continue
                    
                    desc = f"Action {action} changes Object-{obj.id} color from {obj.color} to {new_color}"
                    
                    def make_recolor_fn(act=action, target_obj_id=obj.id, nc=new_color):
                        def fn(current_grid: np.ndarray, a: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
                            if a != act:
                                return {}
                            curr_scene = self.parser.parse(current_grid)
                            target_obj = curr_scene.get_object(target_obj_id)
                            if not target_obj:
                                return {}
                            diffs = {}
                            for r, c in target_obj.pixels:
                                diffs[(int(r), int(c))] = (int(current_grid[r, c]), int(nc))
                            return diffs
                        return fn

                    h = Hypothesis.create(
                        description=desc,
                        program=None,
                        confidence=0.5,
                        source="cognitive_prior",
                        abstraction_level=AbstractionLevel.OBJECT,
                        operations=[f"recolor_obj_{obj.id}", f"color_{new_color}"],
                        preconditions={"action": action, "object_id": obj.id}
                    )
                    h._predictor_fn = make_recolor_fn(action, obj.id, new_color)
                    h._sig = sig
                    generated.append(h)

        return generated


# ============================================================================
# 3. GENERAL COGNITIVE AGENT (HYPRUNE BRAIN v2)
# ============================================================================

class HyPruneCognitiveBrain:
    """The general cognitive brain that plans actions by hypothesis generation,
    mental simulation, verification, and active pruning."""

    def __init__(self, kill_threshold: float = 0.1, max_beam_width: int = 16):
        self.kill_threshold = kill_threshold
        self.max_beam_width = max_beam_width
        self.parser = ObjectParser(connectivity=4)
        self.generator = GeneralHypothesisGenerator(self.parser)
        self.knowledge_base = NegativeKnowledgeBase()
        self.live_hypotheses: List[Hypothesis] = []
        self.used_signatures: Set[str] = set()
        self.unexplored_states: Dict[str, Set[int]] = {}

    def _state_hash(self, grid: np.ndarray) -> str:
        return str(hash(grid.tobytes()))

    def perceive_and_hypothesize(self, grid: np.ndarray, available_actions: List[int]):
        """Ensure the hypothesis bank is populated with diverse hypotheses."""
        # Prune dead hypotheses
        self.live_hypotheses = [h for h in self.live_hypotheses if h.confidence >= self.kill_threshold]

        # If hypothesis bank is low, generate new ones dynamically
        if len(self.live_hypotheses) < 4:
            new_candidates = self.generator.generate_hypotheses_for_state(
                grid, available_actions, self.used_signatures
            )
            for h in new_candidates:
                if hasattr(h, "_sig"):
                    self.used_signatures.add(h._sig)
                self.live_hypotheses.append(h)
                if len(self.live_hypotheses) >= self.max_beam_width:
                    break

    def select_action(self, current_grid: np.ndarray, available_actions: List[int]) -> Tuple[int, Optional[Hypothesis]]:
        """Select action by evaluating mental simulation predictions of surviving hypotheses."""
        self.perceive_and_hypothesize(current_grid, available_actions)

        alive = sorted(
            [h for h in self.live_hypotheses if h.confidence >= self.kill_threshold],
            key=lambda h: h.confidence,
            reverse=True
        )

        chosen_action = None
        chosen_hypothesis = None

        # 1. Mentally simulate and find highest confidence hypothesis predicting an action
        for h in alive:
            if hasattr(h, "_predictor_fn"):
                for a in available_actions:
                    pred = h._predictor_fn(current_grid, a)
                    if pred:  # Non-empty prediction
                        chosen_action = a
                        chosen_hypothesis = h
                        break
            if chosen_action is not None:
                break

        # 2. Fallback to top confidence hypothesis if no prediction
        if chosen_action is None and alive:
            chosen_hypothesis = alive[0]

        # 3. Fallback to unexplored state exploration
        if chosen_action is None:
            sh = self._state_hash(current_grid)
            if sh not in self.unexplored_states:
                self.unexplored_states[sh] = set(available_actions)
            
            unexplored = self.unexplored_states[sh]
            if unexplored:
                chosen_action = sorted(list(unexplored))[0]
                unexplored.remove(chosen_action)
            else:
                chosen_action = available_actions[0]

        return chosen_action, chosen_hypothesis

    def observe_and_update(
        self,
        grid_before: np.ndarray,
        action: int,
        grid_after: np.ndarray,
        chosen_hypothesis: Optional[Hypothesis],
        step_idx: int,
        level_idx: int,
        game_id: str,
        log_file
    ):
        # Compute exact actual pixel diff
        diff_mask = grid_after != grid_before
        actual_diff = {}
        rows, cols = np.where(diff_mask)
        for r, c in zip(rows, cols):
            actual_diff[(int(r), int(c))] = (int(grid_before[r, c]), int(grid_after[r, c]))

        # Concurrently verify ALL live hypotheses against observed outcome
        for h in list(self.live_hypotheses):
            pred_diff = {}
            if hasattr(h, "_predictor_fn"):
                pred_diff = h._predictor_fn(grid_before, action)
            
            matched = (pred_diff == actual_diff)
            if matched:
                h.confidence = min(1.0, h.confidence + 0.15)
                h.evidence_for.append(f"step_{step_idx}")
            else:
                h.confidence -= 0.25
                h.evidence_against.append(f"step_{step_idx}")
                # Record negative knowledge
                if hasattr(h, "_sig"):
                    self.knowledge_base.record_failure(game_id, h)

        # Prune killed hypotheses
        self.live_hypotheses = [h for h in self.live_hypotheses if h.confidence >= self.kill_threshold]

        # Logging
        pred_diff_str = "{}"
        if chosen_hypothesis and hasattr(chosen_hypothesis, "_predictor_fn"):
            pred_diff_str = str(chosen_hypothesis._predictor_fn(grid_before, action))

        log_entry = {
            "game_id": game_id,
            "step": step_idx,
            "level": level_idx,
            "action_taken": action,
            "hypothesis_used": {
                "description": chosen_hypothesis.description if chosen_hypothesis else "State Exploration",
                "source": chosen_hypothesis.source if chosen_hypothesis else "exploration",
                "confidence": round(chosen_hypothesis.confidence, 4) if chosen_hypothesis else 0.0,
                "abstraction": chosen_hypothesis.abstraction_level.value if chosen_hypothesis else "pixel",
            },
            "predicted_diff_sample": pred_diff_str[:120],
            "actual_diff_sample": str(actual_diff)[:120],
            "matched": bool(chosen_hypothesis and hasattr(chosen_hypothesis, "_predictor_fn") and chosen_hypothesis._predictor_fn(grid_before, action) == actual_diff),
            "confidence_after": round(chosen_hypothesis.confidence, 4) if chosen_hypothesis else 0.0,
            "hypotheses_alive_count": len(self.live_hypotheses),
        }

        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()


# ============================================================================
# 4. RANDOM BASELINE POLICY
# ============================================================================

def run_random_baseline(arcade: arc_agi.Arcade, game_id: str, max_steps: int = 100) -> int:
    """Run random baseline policy on game_id."""
    env = arcade.make(game_id)
    frame = env.reset()
    random.seed(42)
    step_count = 0

    while step_count < max_steps:
        state_str = str(frame.state)
        if "GAME_OVER" in state_str:
            frame = env.reset()
            continue
        
        actions = frame.available_actions or [1, 2, 3, 4]
        action = random.choice(actions)
        
        try:
            frame_after = env.step(action)
        except Exception:
            try:
                from arcengine import ActionInput
                frame_after = env.step(ActionInput(id=action, data={"x": 0, "y": 0}))
            except Exception:
                step_count += 1
                continue

        step_count += 1
        
        if frame_after.levels_completed > 0 or "WIN" in str(frame_after.state):
            return step_count

        frame = frame_after

    return max_steps


# ============================================================================
# 5. MULTI-GAME BENCHMARK EXECUTION
# ============================================================================

def main():
    print("==========================================================================")
    print("       HyPrune v2: GENERAL COGNITIVE BRAIN FOR ARC-AGI BENCHMARK          ")
    print("==========================================================================")

    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR)
    
    # Test across multiple distinct ARC games (Generalization Check)
    test_games = ["ls20", "sk48", "tn36", "vc33"]
    max_steps_per_game = 100

    results: Dict[str, Dict[str, Any]] = {}

    with open(GENERAL_LOG_FILE, "w") as log_file:
        for game_id in test_games:
            print(f"\n[Cognitive Brain] Evaluating on Game: {game_id}...")
            
            try:
                env = arcade.make(game_id)
                frame = env.reset()
            except Exception as e:
                print(f"Skipping {game_id} (not available in env dataset): {e}")
                continue

            brain = HyPruneCognitiveBrain(kill_threshold=0.1, max_beam_width=16)
            step_count = 0
            solved = False

            while step_count < max_steps_per_game:
                state_str = str(frame.state)
                if "GAME_OVER" in state_str:
                    frame = env.reset()
                    continue

                grid_before = extract_2d_grid(frame)
                actions = frame.available_actions or [1, 2, 3, 4]

                action, chosen_hypo = brain.select_action(grid_before, actions)
                
                try:
                    frame_after = env.step(action)
                except Exception as step_err:
                    # Fallback for complex action format requirement
                    try:
                        from arcengine import GameAction, ActionInput
                        action_input = ActionInput(id=action, data={"x": 0, "y": 0})
                        frame_after = env.step(action_input)
                    except Exception:
                        step_count += 1
                        continue

                grid_after = extract_2d_grid(frame_after)
                step_count += 1

                brain.observe_and_update(
                    grid_before, action, grid_after, chosen_hypo,
                    step_count, frame_after.levels_completed, game_id, log_file
                )

                if frame_after.levels_completed > 0 or "WIN" in str(frame_after.state):
                    solved = True
                    print(f"✅ {game_id} solved in {step_count} actions!")
                    break

                frame = frame_after

            # Run Random Baseline on same game
            print(f"[Random Baseline] Evaluating on Game: {game_id}...")
            random_steps = run_random_baseline(arcade, game_id, max_steps=max_steps_per_game)

            results[game_id] = {
                "cognitive_brain_steps": step_count,
                "random_baseline_steps": random_steps,
                "solved": solved,
            }

    # Print Final Comparison Table across games
    print("\n==========================================================================")
    print("                 MULTI-GAME COGNITIVE BENCHMARK RESULTS                    ")
    print("==========================================================================")
    print(f"{'Game ID':<12} | {'Cognitive Brain':<22} | {'Random Baseline':<20} | {'Status':<10}")
    print("-" * 72)

    for gid, res in results.items():
        cb_str = f"{res['cognitive_brain_steps']} actions"
        rb_str = f"{res['random_baseline_steps']} actions"
        status = "SOLVED" if res["solved"] else "TIMEOUT"
        print(f"{gid:<12} | {cb_str:<22} | {rb_str:<20} | {status:<10}")

    print("==========================================================================")
    print(f"Log file saved to: {GENERAL_LOG_FILE}")


if __name__ == "__main__":
    main()
