# Swarm Architectures inside Kaggle (Ideas)

This document captures ideas for implementing Multi-Agent / Ensemble / Swarm behavior inside the strict limits of the Kaggle environment (12-hour timeout, single GPU).

## Idea 1: The "Pseudo-Swarm" (Generate and Verify)
Instead of spawning multiple distinct LLM context instances (which multiplies prefill latency and guarantees a timeout), we force a single agent turn to act like a swarm internally.

**How it works:**
1. **Prompt Engineering:** We instruct the LLM to act as a committee. It must generate exactly 3 different Python candidate scripts to solve the puzzle.
2. **Sandbox Execution:** The framework executes all 3 candidates in the sandbox using our enhanced Level 2 tools (`find_objects`, `find_path`).
3. **Internal Arbiter:** A lightweight verification function scores the output of the 3 scripts against the training examples. The script with the highest pass rate is selected as the final submission.

**Pros:** Minimal token overhead. Avoids vLLM context swapping penalties.
**Cons:** The LLM might blend the logic of the 3 candidates if they are in the same output block.

## Idea 2: True Multi-Agent Swarm (The AlphaCode Approach)
Implementing distinct agents with specialized roles.

**Roles:**
- **Agent A (The Coder):** Writes the initial transformation logic.
- **Agent B (The Critic):** Reviews the code, identifies edge cases, and provides feedback without executing.
- **Agent C (The Executor):** Synthesizes the feedback, runs the code, and debugs.

**How to make it work in Kaggle:**
To avoid the 12-hour timeout, a True Swarm CANNOT be used on every single puzzle. It must be paired with an **Arbiter/Triage System**:
1. Run a fast, single-pass baseline on all 400 puzzles first.
2. Identify puzzles where the model is highly confident (e.g., training examples passed on the first try). Submit those immediately.
3. For the remaining "hard" puzzles, route them to the Multi-Agent Swarm.
4. Set a hard wall-clock budget for the Swarm (e.g., 5 minutes per puzzle). If the Swarm doesn't converge, force a fallback submission.

**Pros:** Extremely high reasoning quality for complex puzzles.
**Cons:** High risk of wall-clock timeouts. Requires complex state management and routing logic inside the `ToolAgent` execution loop.

## Idea 3: Dynamic Agent Scaling (The Triage Swarm)
A brilliant evolution to beat the Kaggle 12-hour limit by using a sliding scale of compute power based on puzzle complexity.

**How it works:**
1. **The "Scout" Phase:** For every puzzle, we initially send in just *one* fast agent. If it's a simple puzzle (like recoloring a few pixels) and it passes the training examples immediately, we submit the answer and move on.
2. **The "Dynamic Swarm" Phase:** If the Scout agent fails or determines the puzzle has extreme complexity (e.g., massive grid with physics logic), the system flags it as a "Boss Level."
3. **Spawning the Army:** Because time was saved on easy puzzles, the system dynamically spawns 5 to 20 specialized agents specifically for this hard puzzle:
   - *Rotation Agent*: Focuses solely on geometric rotation.
   - *Color Agent*: Focuses purely on color mapping.
   - *Physics Agent*: Focuses on gravity/connectivity logic.
   - *Manager Agent*: Collects hypotheses and tests them in the sandbox.

**Pros:** Maximum compute is only spent where it is mathematically necessary. Prevents Kaggle timeouts while allowing deep exploration on hard puzzles.
**Cons:** The orchestration logic required to pause, spawn, and sync 20 async agents inside the TAAF loop is highly complex to implement.
