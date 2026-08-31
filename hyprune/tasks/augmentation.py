"""
ARC task augmentation module.
"""
from __future__ import annotations
import numpy as np
import random
from copy import deepcopy
from typing import Any
from .loader import ArcTask, ArcPair

class TaskAugmenter:
    """Augments ARC tasks with symmetry-preserving transformations."""
    
    def augment(self, task: ArcTask, augmentations: list[str] | None = None) -> list[ArcTask]:
        """Generate augmented versions of a task.
        
        Available augmentations:
        - 'rotate_90': Rotate all grids 90°
        - 'rotate_180': Rotate all grids 180°
        - 'rotate_270': Rotate all grids 270°
        - 'flip_h': Horizontal flip
        - 'flip_v': Vertical flip
        - 'color_permute': Randomly permute non-background colors
        - 'transpose': Transpose all grids
        """
        if augmentations is None:
            augmentations = ['rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v', 'transpose']
            
        results = []
        for aug in augmentations:
            new_task = deepcopy(task)
            new_task.id = f"{task.id}_{aug}"
            
            for split in (new_task.train_pairs, new_task.test_pairs):
                for pair in split:
                    if aug == 'rotate_90':
                        pair.input_grid = self.rotate_grid(pair.input_grid, 1)
                        if pair.output_grid.size > 0:
                            pair.output_grid = self.rotate_grid(pair.output_grid, 1)
                    elif aug == 'rotate_180':
                        pair.input_grid = self.rotate_grid(pair.input_grid, 2)
                        if pair.output_grid.size > 0:
                            pair.output_grid = self.rotate_grid(pair.output_grid, 2)
                    elif aug == 'rotate_270':
                        pair.input_grid = self.rotate_grid(pair.input_grid, 3)
                        if pair.output_grid.size > 0:
                            pair.output_grid = self.rotate_grid(pair.output_grid, 3)
                    elif aug == 'flip_h':
                        pair.input_grid = self.flip_grid(pair.input_grid, 1)
                        if pair.output_grid.size > 0:
                            pair.output_grid = self.flip_grid(pair.output_grid, 1)
                    elif aug == 'flip_v':
                        pair.input_grid = self.flip_grid(pair.input_grid, 0)
                        if pair.output_grid.size > 0:
                            pair.output_grid = self.flip_grid(pair.output_grid, 0)
                    elif aug == 'transpose':
                        pair.input_grid = self.transpose_grid(pair.input_grid)
                        if pair.output_grid.size > 0:
                            pair.output_grid = self.transpose_grid(pair.output_grid)
                    elif aug == 'color_permute':
                        # Permute colors 1-9
                        colors = list(range(1, 10))
                        shuffled = colors.copy()
                        random.shuffle(shuffled)
                        perm = {c: s for c, s in zip(colors, shuffled)}
                        perm[0] = 0 # background is always 0
                        
                        pair.input_grid = self.permute_colors(pair.input_grid, perm)
                        if pair.output_grid.size > 0:
                            pair.output_grid = self.permute_colors(pair.output_grid, perm)
            results.append(new_task)
            
        return results
    
    @staticmethod
    def rotate_grid(grid: np.ndarray, k: int = 1) -> np.ndarray:
        """Rotate grid by 90 degrees * k."""
        return np.rot90(grid, k=k)
        
    @staticmethod
    def flip_grid(grid: np.ndarray, axis: int) -> np.ndarray:
        """Flip grid horizontally (axis=1) or vertically (axis=0)."""
        return np.flip(grid, axis=axis)
        
    @staticmethod
    def permute_colors(grid: np.ndarray, permutation: dict[int, int]) -> np.ndarray:
        """Map colors according to a permutation dictionary."""
        out = np.zeros_like(grid)
        for old_val, new_val in permutation.items():
            out[grid == old_val] = new_val
        return out
        
    @staticmethod
    def transpose_grid(grid: np.ndarray) -> np.ndarray:
        """Transpose the grid."""
        return np.transpose(grid)
