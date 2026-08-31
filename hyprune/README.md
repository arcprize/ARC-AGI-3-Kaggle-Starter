# HyPrune

**HyPrune** is a Hypothesis Pruning cognitive architecture designed for solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles.

## Overview

HyPrune uses structured hypothesis generation, teacher-guided evaluations, and adaptive pruning algorithms to search, refine, and verify programs or visual rules that solve ARC-AGI tasks efficiently.

## Directory Structure

- `hyprune/core/`: Core abstractions, grid representations, hypothesis spaces, and search algorithms.
- `hyprune/tasks/`: ARC puzzle loaders, parsers, and task manipulation utilities.
- `hyprune/teachers/`: Guiding agents, heuristics, and evaluation teachers for hypothesis generation and pruning.
- `hyprune/training/`: Training loops, optimization procedures, and reinforcement/feedback routines.
- `hyprune/evaluation/`: Benchmarking, metrics evaluation, and validation tools.
- `tests/`: Unit and integration test suite.

## Installation

Requires Python 3.11+.

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e .[dev]
```
