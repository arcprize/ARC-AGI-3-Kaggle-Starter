"""
ARC task loader module.
"""
from __future__ import annotations
import dataclasses
import json
import numpy as np
from pathlib import Path
from typing import Optional, Any

@dataclasses.dataclass
class ArcPair:
    input_grid: np.ndarray  # 2D array of ints 0-9
    output_grid: np.ndarray
    
    @property
    def input_shape(self) -> tuple[int, int]:
        return self.input_grid.shape
        
    @property
    def output_shape(self) -> tuple[int, int]:
        return self.output_grid.shape
        
    @property  
    def shape_changes(self) -> bool:
        """Does the output have different dimensions than input?"""
        return self.input_shape != self.output_shape

@dataclasses.dataclass
class ArcTask:
    id: str
    train_pairs: list[ArcPair]
    test_pairs: list[ArcPair]  # May be empty if solutions not available
    
    @property
    def num_train(self) -> int:
        return len(self.train_pairs)
        
    @property
    def num_test(self) -> int:
        return len(self.test_pairs)
        
    @property
    def has_test_solutions(self) -> bool:
        return any(p.output_grid.size > 0 for p in self.test_pairs)
        
    @property
    def max_grid_size(self) -> tuple[int, int]:
        max_r, max_c = 0, 0
        for p in self.train_pairs + self.test_pairs:
            for grid in (p.input_grid, p.output_grid):
                if grid.size > 0:
                    r, c = grid.shape
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)
        return max_r, max_c
        
    @property
    def all_colors_used(self) -> set[int]:
        colors = set()
        for p in self.train_pairs + self.test_pairs:
            for grid in (p.input_grid, p.output_grid):
                if grid.size > 0:
                    colors.update(np.unique(grid))
        return colors
        
    @property
    def shape_preserving(self) -> bool:
        """Do all pairs preserve grid dimensions?"""
        return not any(p.shape_changes for p in self.train_pairs + self.test_pairs if p.output_grid.size > 0)

class ArcTaskLoader:
    """Loads ARC tasks from the standard JSON format."""
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
    
    def load_task(self, task_id: str) -> ArcTask:
        file_path = self.data_dir / f"{task_id}.json"
        if not file_path.exists():
            # Try searching in subdirs
            paths = list(self.data_dir.rglob(f"{task_id}.json"))
            if not paths:
                raise FileNotFoundError(f"Task {task_id} not found in {self.data_dir}")
            file_path = paths[0]
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return self._parse_task_json(task_id, data)
        
    def load_all(self) -> list[ArcTask]:
        tasks = []
        for file_path in self.data_dir.rglob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tasks.append(self._parse_task_json(file_path.stem, data))
        return tasks
        
    def load_split(self, split: str = 'training') -> list[ArcTask]:
        """Load a specific split: 'training' or 'evaluation'."""
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
        tasks = []
        for file_path in split_dir.rglob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tasks.append(self._parse_task_json(file_path.stem, data))
        return tasks
    
    @staticmethod
    def _parse_task_json(task_id: str, data: dict[str, Any]) -> ArcTask:
        train_pairs = []
        for item in data.get('train', []):
            inp = np.array(item['input'], dtype=np.int32)
            outp = np.array(item['output'], dtype=np.int32)
            train_pairs.append(ArcPair(inp, outp))
            
        test_pairs = []
        for item in data.get('test', []):
            inp = np.array(item['input'], dtype=np.int32)
            # Some test pairs might not have an output if they are true test sets
            outp = np.array(item['output'], dtype=np.int32) if 'output' in item else np.array([[]], dtype=np.int32)
            test_pairs.append(ArcPair(inp, outp))
            
        return ArcTask(id=task_id, train_pairs=train_pairs, test_pairs=test_pairs)
    
    @staticmethod
    def grid_to_ascii(grid: np.ndarray) -> str:
        """Pretty-print a grid using ARC color symbols."""
        if grid.size == 0:
            return "(Empty Grid)"
        rows, cols = grid.shape
        lines = []
        lines.append("   " + " ".join(f"{c:x}" for c in range(cols)))
        lines.append("  +" + "--" * cols)
        for r in range(rows):
            lines.append(f"{r:x} |" + " ".join(str(int(cell)) for cell in grid[r]))
        return "\n".join(lines)
