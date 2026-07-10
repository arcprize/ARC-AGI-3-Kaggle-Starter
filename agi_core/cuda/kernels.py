"""CUDA kernels for salience computation and batch A* pathfinding.

All kernels use ``numba.cuda.jit`` decorators.  Every kernel is designed
for a 64×64 grid (the standard ARC-AGI-3 board size) but accepts
dynamic width/height parameters for smaller boards.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Salience kernel
# 2-D grid: blockDim = (16, 16), gridDim = (4, 4) → 64×64 threads
# Each thread computes one cell's salience value.
# ---------------------------------------------------------------------------


def _launch_salience_kernel(board, px, py, goal_xs, goal_ys, out, prev_board=None):
    """Launch the CUDA salience kernel if numba is available; else raise."""
    from numba import cuda, float32, int32

    @cuda.jit
    def _salience_kernel(
        board, prev_board, width, height,
        px, py,
        goal_xs, goal_ys, n_goals,
        out,
    ):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return

        s = 0.0

        # 1. Motion (if prev_board provided)
        if prev_board is not None and prev_board[y, x] != board[y, x]:
            s += 0.25

        # 2. Player vicinity (Manhattan)
        dist = abs(x - px) + abs(y - py)
        if dist < 20:
            s += 0.30 * (1.0 - dist / 20.0)

        # 3. Goal vicinity (nearest goal)
        best = 16.0
        for i in range(n_goals):
            gx = goal_xs[i]
            gy = goal_ys[i]
            d = float(abs(x - gx) + abs(y - gy))
            if d < best:
                best = d
        if best < 16.0:
            s += 0.20 * (1.0 - best / 16.0)

        # 4. Edge bonus
        if x < 3 or x >= width - 3 or y < 3 or y >= height - 3:
            s += 0.10

        out[y, x] = min(1.0, s)

    h, w = board.shape[:2]
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (w + threads_per_block[0] - 1) // threads_per_block[0],
        (h + threads_per_block[1] - 1) // threads_per_block[1],
    )
    _salience_kernel[blocks_per_grid, threads_per_block](
        board, prev_board, w, h,
        px, py,
        goal_xs, goal_ys, len(goal_xs),
        out,
    )
    cuda.synchronize()


# ---------------------------------------------------------------------------
# Batch A* kernel
# 1-D grid: one block per search (up to 32 concurrent searches)
# Each block uses shared memory for open/closed sets.
# ---------------------------------------------------------------------------


def _launch_astar_kernel(walkable, starts_x, starts_y, goals_x, goals_y, paths_out, path_lens):
    """Launch the CUDA A* kernel if numba is available; else raise."""
    from numba import cuda, int32, uint8

    @cuda.jit
    def _astar_kernel(
        walkable, width, height,
        starts_x, starts_y,
        goals_x, goals_y,
        paths_out, path_lens_out,
    ):
        """One block per search.  Shared memory layout (42 KB total):
        
        - open_set:   256 × 2 int32  = 2 KB  (x, y pairs)
        - open_count: 1 int32         = 4 B
        - g_score:    4096 float32   = 16 KB (64×64)
        - closed:     4096 uint8     = 4 KB
        - parent:     4096 uint16    = 8 KB
        - path_buf:   256 uint16     = 0.5 KB
        Total: ~30.5 KB (fits in 48 KB shared memory)
        """
        bid = cuda.blockIdx.x
        if bid >= path_lens_out.shape[0]:
            return

        sx = starts_x[bid]
        sy = starts_y[bid]
        gx = goals_x[bid]
        gy = goals_y[bid]

        # Shared memory allocation
        open_x = cuda.shared.array(256, dtype=int32)
        open_y = cuda.shared.array(256, dtype=int32)
        open_count = cuda.shared.array(1, dtype=int32)
        g_score = cuda.shared.array(4096, dtype=float32)
        closed = cuda.shared.array(4096, dtype=uint8)
        parent = cuda.shared.array(4096, dtype=int32)

        tid = cuda.threadIdx.x
        stride = cuda.blockDim.x

        # Initialise shared memory
        for i in range(tid, 4096, stride):
            g_score[i] = 1e18
            closed[i] = 0
            parent[i] = -1

        cuda.syncthreads()

        # Start node
        if tid == 0:
            start_idx = sy * width + sx
            g_score[start_idx] = 0.0
            open_x[0] = sx
            open_y[0] = sy
            open_count[0] = 1

        cuda.syncthreads()

        goal_idx = gy * width + gx
        found = 0

        for _ in range(width * height):  # at most one iteration per cell
            cuda.syncthreads()
            if found:
                break

            # Find minimum f_score in open set (parallel reduction)
            best_idx = -1
            best_f = 1e18
            best_open_pos = -1

            for i in range(tid, open_count[0], stride):
                ox = open_x[i]
                oy = open_y[i]
                oi = oy * width + ox
                if closed[oi]:
                    continue
                h = abs(ox - gx) + abs(oy - gy)
                f = g_score[oi] + float(h)
                if f < best_f:
                    best_f = f
                    best_idx = oi
                    best_open_pos = i

            # Parallel reduction to find global minimum
            # (simplified: thread 0 picks the first found by the lowest thread ID)
            cuda.syncthreads()

            if tid == 0 and best_idx >= 0:
                closed[best_idx] = 1
                # Remove from open set
                if best_open_pos >= 0 and best_open_pos < open_count[0] - 1:
                    open_x[best_open_pos] = open_x[open_count[0] - 1]
                    open_y[best_open_pos] = open_y[open_count[0] - 1]
                open_count[0] -= 1

                bx = best_idx % width
                by = best_idx // width

                if best_idx == goal_idx:
                    found = 1
                else:
                    # Expand neighbours (4-directional)
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nx = bx + dx
                        ny = by + dy
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue
                        ni = ny * width + nx
                        if closed[ni] or walkable[ni] == 0:
                            continue
                        if walkable[ny, nx] == 0:
                            continue
                        new_g = g_score[best_idx] + 1.0
                        if new_g < g_score[ni]:
                            g_score[ni] = new_g
                            parent[ni] = best_idx
                            # Add to open set
                            if open_count[0] < 256:
                                open_x[open_count[0]] = nx
                                open_y[open_count[0]] = ny
                                open_count[0] += 1

        # Reconstruct path
        if tid == 0:
            if found:
                total_len = 0
                cur = goal_idx
                while cur >= 0 and cur != start_idx and total_len < 256:
                    paths_out[bid, total_len] = int32(cur)
                    total_len += 1
                    cur = parent[cur]
                if cur == start_idx:
                    paths_out[bid, total_len] = int32(start_idx)
                    total_len += 1
                path_lens_out[bid] = total_len
                # Reverse path in-place
                for i in range(total_len // 2):
                    j = total_len - 1 - i
                    tmp = paths_out[bid, i]
                    paths_out[bid, i] = paths_out[bid, j]
                    paths_out[bid, j] = tmp
            else:
                path_lens_out[bid] = 0

    n_searches = len(starts_x)
    _astar_kernel[n_searches, 128](
        walkable, walkable.shape[1], walkable.shape[0],
        starts_x, starts_y,
        goals_x, goals_y,
        paths_out, path_lens,
    )
    cuda.synchronize()
