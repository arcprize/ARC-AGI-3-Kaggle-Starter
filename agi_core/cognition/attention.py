from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SalienceMap:
    """A 64×64 float map where each cell holds a salience value [0, 1].

    Regions near the player, goals, recent changes, and moving entities
    get higher values.  Low-salience regions can be processed at reduced
    resolution or skipped entirely by ``build_game_model``.
    """

    def __init__(self, width: int = 64, height: int = 64) -> None:
        self.width = width
        self.height = height
        self.data: list[list[float]] = [[0.0] * width for _ in range(height)]
        self._threshold: float = 0.05

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    def get(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y][x]
        return 0.0

    def above_threshold(self, x: int, y: int) -> bool:
        return self.get(x, y) >= self._threshold

    def is_relevant(self, x: int, y: int) -> bool:
        """Returns True if the cell is salient enough to analyse at full resolution."""
        return self.above_threshold(x, y)


class AttentionFocus:
    """Computes a salience map over the board each frame.

    High-salience regions (player, goals, moving entities, recent changes)
    get full-resolution analysis.  Low-salience regions are skipped or
    downsampled, mimicking human foveal + peripheral vision.
    """

    def __init__(self, width: int = 64, height: int = 64) -> None:
        self.salience = SalienceMap(width, height)
        self._prev_board: list[list[int]] | None = None
        self._visitation_counts: list[list[int]] = [
            [0] * width for _ in range(height)
        ]

    def compute(
        self,
        board: list[list[int]],
        player_pos: tuple[int, int] | None,
        goal_positions: list[tuple[int, int]],
        prev_board: list[list[int]] | None = None,
    ) -> SalienceMap:
        """Compute the salience map for the current frame.

        Factors (each contributes a weight to the final salience):
          1. **Motion**: cells that changed since last frame  (weight 0.25)
          2. **Player vicinity**: gaussian falloff from player  (weight 0.30)
          3. **Goal vicinity**: distance from nearest goal      (weight 0.20)
          4. **Novelty**: inverse visitation count             (weight 0.15)
          5. **Edge**: board edges (mirror reflection zone)    (weight 0.10)
        """
        h, w = self.salience.height, self.salience.width
        pb = prev_board or self._prev_board
        for y in range(h):
            for x in range(w):
                s = 0.0
                # 1. Motion
                if pb is not None and y < len(pb) and x < len(pb[y]):
                    if pb[y][x] != board[y][x]:
                        s += 0.25
                # 2. Player vicinity
                if player_pos is not None:
                    dx = abs(x - player_pos[0])
                    dy = abs(y - player_pos[1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < 15:
                        s += 0.30 * (1.0 - dist / 15.0)
                # 3. Goal vicinity
                for gx, gy in goal_positions:
                    dx = abs(x - gx)
                    dy = abs(y - gy)
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < 12:
                        s += 0.20 * (1.0 - dist / 12.0)
                        break
                # 4. Novelty (inverse visitation)
                vis = self._visitation_counts[y][x]
                s += 0.15 * max(0.0, 1.0 - vis / 50.0)
                # 5. Edge regions (mirror reflection zone)
                if x < 3 or x >= w - 3 or y < 3 or y >= h - 3:
                    s += 0.10
                self.salience.data[y][x] = min(1.0, s)
        self._prev_board = board
        return self.salience

    def mark_visited(self, player_pos: tuple[int, int] | None) -> None:
        """Increment visitation count near the player (exploration bonus decay)."""
        if player_pos is None:
            return
        px, py = player_pos
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                x, y = px + dx, py + dy
                if 0 <= x < self.salience.width and 0 <= y < self.salience.height:
                    self._visitation_counts[y][x] = min(
                        50, self._visitation_counts[y][x] + 1,
                    )

    def novelty_bonus(self, x: int, y: int) -> float:
        """Return a novelty bonus [0, 0.2] for under-visited cells.
        
        This drives exploration — cells that have been visited fewer times
        get a bonus that makes the agent more likely to explore them.
        """
        if 0 <= x < self.salience.width and 0 <= y < self.salience.height:
            vis = self._visitation_counts[y][x]
            return 0.2 * max(0.0, 1.0 - vis / 30.0)
        return 0.0

    def reset_visitation(self) -> None:
        """Reset visitation counts (new level)."""
        h, w = self.salience.height, self.salience.width
        self._visitation_counts = [[0] * w for _ in range(h)]
        self._prev_board = None
