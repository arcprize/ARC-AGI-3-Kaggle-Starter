"""CUDA kernels for salience computation, batch A* pathfinding,
and connected-component labeling.

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
                        if closed[ni] or walkable[bid, ny, nx] == 0:
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
        walkable, walkable.shape[2], walkable.shape[1],
        starts_x, starts_y,
        goals_x, goals_y,
        paths_out, path_lens,
    )
    cuda.synchronize()


# ---------------------------------------------------------------------------
# Connected-component labeling kernels (GPU Union-Find)
# 2-D grid: blockDim = (16, 16), gridDim = (4, 4) → 64×64 threads
# Each thread manages one pixel's label.
#
# Algorithm: iterative label propagation (4-connectivity)
#   Phase 1 — Initialise: each pixel gets label = y * width + x
#   Phase 2 — Propagate: adopt min label of same-color neighbours, repeat
#   Phase 3 — Compact: renumber labels to dense 0..N-1 range
# ---------------------------------------------------------------------------


def _launch_init_labels(board, labels_out, width, height):
    """Initialise each pixel's label to its linear index."""
    from numba import cuda, int32

    @cuda.jit
    def _init_labels_kernel(board, labels, width, height):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return
        val = int(board[y, x])
        if val < 0:
            labels[y, x] = -1
        else:
            labels[y, x] = y * width + x

    _init_labels_kernel[(height + 15) // 16, (width + 15) // 16, (16, 16)](
        board, labels_out, width, height,
    )
    cuda.synchronize()


def _launch_propagate_labels(board, labels, width, height):
    """Run one propagation iteration.  Returns True if converged."""
    from numba import cuda, int32

    @cuda.jit
    def _propagate_kernel(board, labels, width, height, converged):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return
        my_label = labels[y, x]
        if my_label < 0:
            return
        my_color = int(board[y, x])
        new_label = my_label
        # Check 4-neighbours
        if x > 0:
            nv = int(board[y, x - 1])
            if nv == my_color:
                nl = labels[y, x - 1]
                if nl >= 0 and nl < new_label:
                    new_label = nl
        if x < width - 1:
            nv = int(board[y, x + 1])
            if nv == my_color:
                nl = labels[y, x + 1]
                if nl >= 0 and nl < new_label:
                    new_label = nl
        if y > 0:
            nv = int(board[y - 1, x])
            if nv == my_color:
                nl = labels[y - 1, x]
                if nl >= 0 and nl < new_label:
                    new_label = nl
        if y < height - 1:
            nv = int(board[y + 1, x])
            if nv == my_color:
                nl = labels[y + 1, x]
                if nl >= 0 and nl < new_label:
                    new_label = nl
        if new_label != my_label:
            labels[y, x] = new_label
            converged[0] = 0

    d_converged = cuda.device_array(1, dtype=np.int32)
    d_converged[0] = 1
    _propagate_kernel[(height + 15) // 16, (width + 15) // 16, (16, 16)](
        board, labels, width, height, d_converged,
    )
    cuda.synchronize()
    return bool(d_converged.copy_to_host()[0])


def _launch_compact_labels(board, labels, width, height):
    """Collect (label, x, y, color) from the GPU and build component metadata.

    Returns the same dict format as ``_compact_labels_cpu``.
    """
    from numba import cuda, int32

    max_entries = width * height  # 4096 for 64×64

    @cuda.jit
    def _compact_kernel(board, labels, width, height, offsets, count, max_entries):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return
        my_label = labels[y, x]
        if my_label < 0:
            return
        pos = cuda.atomic.add(count, 0, 1)
        if pos < max_entries:
            offsets[pos, 0] = my_label
            offsets[pos, 1] = x
            offsets[pos, 2] = y
            offsets[pos, 3] = int(board[y, x])

    d_offsets = cuda.device_array((max_entries, 4), dtype=np.int32)
    d_count = cuda.device_array(1, dtype=np.int32)
    d_count[0] = 0

    threads = (16, 16)
    blocks = (
        (width + 15) // 16,
        (height + 15) // 16,
    )
    _compact_kernel[blocks, threads](
        board, labels, width, height, d_offsets, d_count, max_entries,
    )
    cuda.synchronize()

    n = int(d_count.copy_to_host()[0])
    if n == 0:
        return []

    offsets = d_offsets.copy_to_host()[:n]

    # Sort by label on CPU (fast for ≤ 4096 items)
    offsets = offsets[offsets[:, 0].argsort()]

    # Build component metadata from sorted array
    return _components_from_sorted(offsets, board, width, height)


def _components_from_sorted(
    offsets: np.ndarray,
    board: list[list[int]],
    width: int, height: int,
) -> list[dict]:
    """Build component dicts from a sorted (label, x, y, color) array.

    The array must be sorted by the label column (column 0).
    Returns the same format as ``_compact_labels_cpu``.
    """
    results: list[dict] = []
    i = 0
    n = len(offsets)
    while i < n:
        label = int(offsets[i, 0])
        color = int(offsets[i, 3])
        min_x = max_x = int(offsets[i, 1])
        min_y = max_y = int(offsets[i, 2])
        cells: list[tuple[int, int]] = [(int(offsets[i, 1]), int(offsets[i, 2]))]
        j = i + 1
        while j < n and int(offsets[j, 0]) == label:
            x, y = int(offsets[j, 1]), int(offsets[j, 2])
            cells.append((x, y))
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
            j += 1
        size = j - i
        sx = sum(c[0] for c in cells) // size
        sy = sum(c[1] for c in cells) // size
        tb = (min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1)
        results.append({
            "label": label,
            "color": color,
            "cells": cells,
            "bbox": (min_x, min_y, max_x, max_y),
            "size": size,
            "center": (sx, sy),
            "touches_border": tb,
        })
        i = j
    return results


# CPU-based label compaction (fallback)
def _compact_labels_cpu(labels, board, width, height):
    """Compact GPU labels on CPU and build component metadata.
    
    Returns list of dicts with keys:
        label, color, cells, bbox, size, touches_border, center
    """
    comp_map: dict[int, dict] = {}
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label < 0:
                continue
            color = int(board[y][x])
            if label not in comp_map:
                comp_map[label] = {
                    "label": label,
                    "color": color,
                    "cells": [],
                    "min_x": x, "max_x": x,
                    "min_y": y, "max_y": y,
                }
            entry = comp_map[label]
            entry["cells"].append((x, y))
            if x < entry["min_x"]: entry["min_x"] = x
            if x > entry["max_x"]: entry["max_x"] = x
            if y < entry["min_y"]: entry["min_y"] = y
            if y > entry["max_y"]: entry["max_y"] = y

    results = []
    for entry in comp_map.values():
        cells = entry["cells"]
        bbox = (entry["min_x"], entry["min_y"], entry["max_x"], entry["max_y"])
        size = len(cells)
        sx = sum(x for x, _ in cells) // size
        sy = sum(y for _, y in cells) // size
        tb = (entry["min_x"] == 0 or entry["min_y"] == 0
              or entry["max_x"] == width - 1 or entry["max_y"] == height - 1)
        results.append({
            "label": entry["label"],
            "color": entry["color"],
            "cells": cells,
            "bbox": bbox,
            "size": size,
            "center": (sx, sy),
            "touches_border": tb,
        })
    return results


# ---------------------------------------------------------------------------
# Mirror placement brute-force kernel
# 2-D grid: one thread per (tx, ty) placement position
# Each thread tests one position, atomic-max updates the global best.
# ---------------------------------------------------------------------------


def _launch_mirror_placement_kernel(
    goal_bitmask, shape_mask, shape_w, shape_h,
    horizontal, axis, width, height, best_out,
):
    """Launch the mirror placement kernel.
    
    ``goal_bitmask`` : uint8 array (height × width), 1 = goal cell.
    ``shape_mask``   : uint8 array (shape_h × shape_w), 1 = shape cell.
    ``best_out``     : int32 array [fresh_hits, center_x, center_y] — in-place update.
    
    The kernel writes the best found placement into ``best_out`` via atomic
    compare-and-swap on a packed score (hit_count << 20 | -distance).
    """
    from numba import cuda, int32

    @cuda.jit
    def _mirror_placement_kernel(
        goal, shape, sw, sh, horiz, axis,
        width, height, best,
    ):
        tx, ty = cuda.grid(2)
        max_tx = width - sw
        max_ty = height - sh
        if tx > max_tx or ty > max_ty:
            return

        # Count reflected goal hits for this placement
        hits = 0
        for dy in range(sh):
            for dx in range(sw):
                if shape[dy, dx]:
                    if horiz:
                        rx = tx + dx
                        ry = 2 * axis - (ty + dy)
                    else:
                        rx = 2 * axis - (tx + dx)
                        ry = ty + dy
                    if 0 <= rx < width and 0 <= ry < height and goal[ry, rx]:
                        hits += 1

        if hits == 0:
            return

        cx = tx + sw // 2
        cy = ty + sh // 2

        # Pack score: higher hits wins, then smaller distance wins
        # (use negative distance in packed value)
        # Score = (hits << 20) | (max_dist - distance)
        # where max_dist = width + height (safe upper bound)
        max_dist = width + height
        dist = abs(cx - tx) + abs(cy - ty)  # approx movement from shape center
        neg_dist = max_dist - dist
        packed = (hits << 20) | neg_dist

        # Atomic compare-and-swap: update best if this is better
        cuda.atomic.max(best, 0, packed)
        cuda.atomic.max(best, 1, int32(cx))
        cuda.atomic.max(best, 2, int32(cy))
        cuda.atomic.max(best, 3, int32(hits))

    max_tx = max(0, width - shape_w)
    max_ty = max(0, height - shape_h)
    if max_tx == 0 or max_ty == 0:
        tx_dim = 1
        ty_dim = 1
    else:
        tx_dim = min(32, max_tx)
        ty_dim = min(32, max_ty)
    threads = (tx_dim, ty_dim)
    blocks = (
        (max_tx + tx_dim) // tx_dim if tx_dim > 0 else 1,
        (max_ty + ty_dim) // ty_dim if ty_dim > 0 else 1,
    )
    _mirror_placement_kernel[blocks, threads](
        goal_bitmask, shape_mask, shape_w, shape_h,
        int(horizontal), int(axis), width, height,
        best_out,
    )
    cuda.synchronize()


def _launch_reflection_score_kernel(
    goal_bitmask,
    shape_mask,
    shape_w,
    shape_h,
    horizontal,
    axis,
    width,
    height,
    score_out,
):
    """Score reflected-goal hits for every top-left placement position."""
    from numba import cuda, int32

    @cuda.jit
    def _reflection_score_kernel(
        goal, shape, sw, sh, horiz, axis,
        width, height, score_out,
    ):
        tx, ty = cuda.grid(2)
        max_tx = width - sw
        max_ty = height - sh
        if tx > max_tx or ty > max_ty:
            return

        hits = 0
        for dy in range(sh):
            for dx in range(sw):
                if shape[dy, dx]:
                    if horiz:
                        rx = tx + dx
                        ry = 2 * axis - (ty + dy)
                    else:
                        rx = 2 * axis - (tx + dx)
                        ry = ty + dy
                    if 0 <= rx < width and 0 <= ry < height and goal[ry, rx]:
                        hits += 1
        score_out[ty, tx] = int32(hits)

    max_tx = max(0, width - shape_w)
    max_ty = max(0, height - shape_h)
    tx_dim = min(16, max_tx + 1) if max_tx >= 0 else 1
    ty_dim = min(16, max_ty + 1) if max_ty >= 0 else 1
    threads = (max(1, tx_dim), max(1, ty_dim))
    blocks = (
        (max_tx + threads[0]) // threads[0] if threads[0] > 0 else 1,
        (max_ty + threads[1]) // threads[1] if threads[1] > 0 else 1,
    )
    _reflection_score_kernel[blocks, threads](
        goal_bitmask, shape_mask, shape_w, shape_h,
        int(horizontal), int(axis), width, height, score_out,
    )
    cuda.synchronize()


# ---------------------------------------------------------------------------
# Grid scale detection kernels
# 1-D grid: 17 blocks × 256 threads for uniformity (one block per scale)
# 1-D grid: 64 blocks × 64 threads for edge periodicity (one block per row)
# ---------------------------------------------------------------------------


def _launch_uniformity_kernel(board, uniformity_out, width, height):
    """Compute block uniformity for each scale 2–16.
    
    ``uniformity_out`` is a 17-element float32 array (index = scale).
    Each block computes the fraction of scale×scale blocks that are uniform.
    """
    from numba import cuda, int32, float32

    @cuda.jit
    def _uniformity_kernel(board, width, height, uniformity_out):
        scale = cuda.blockIdx.x
        if scale < 2 or scale > 16:
            if cuda.threadIdx.x == 0:
                uniformity_out[scale] = 0.0
            return

        bw = width // scale
        bh = height // scale
        if bw < 4 or bh < 4:
            if cuda.threadIdx.x == 0:
                uniformity_out[scale] = 0.0
            return

        total = bw * bh
        tid = cuda.threadIdx.x
        bdim = cuda.blockDim.x

        # Shared memory for partial counts
        shared_counts = cuda.shared.array(256, dtype=int32)
        shared_counts[tid] = 0

        # Each thread checks a strided subset of blocks
        for idx in range(tid, total, bdim):
            bx = idx % bw
            by = idx // bw
            px = bx * scale
            py = by * scale
            block_color = board[py, px]
            uniform = 1
            for dy in range(scale):
                for dx in range(scale):
                    if board[py + dy, px + dx] != block_color:
                        uniform = 0
                        break
                if not uniform:
                    break
            shared_counts[tid] += uniform

        cuda.syncthreads()

        # Parallel reduction
        s = 128
        while s > 0:
            if tid < s:
                shared_counts[tid] += shared_counts[tid + s]
            cuda.syncthreads()
            s //= 2

        if tid == 0:
            uniformity_out[scale] = float32(shared_counts[0]) / float32(total)

    n_scales = 17
    _uniformity_kernel[n_scales, 256](
        board, width, height, uniformity_out,
    )
    cuda.synchronize()


def _launch_edge_kernel(board, edge_scores_out, width, height):
    """Count colour-transition gaps of each period (1–9) per row.
    
    ``edge_scores_out`` is a 17-element int32 array (index = gap size).
    """
    from numba import cuda, int32

    @cuda.jit
    def _edge_kernel(board, width, height, edge_scores_out):
        y = cuda.blockIdx.x
        if y >= height:
            return
        tid = cuda.threadIdx.x  # = x position in row

        # Shared memory for gap counts per row
        shared_gaps = cuda.shared.array(16, dtype=int32)
        if tid < 16:
            shared_gaps[tid] = 0
        cuda.syncthreads()

        # Scan the row looking for colour transitions
        prev = -2
        gap = 0
        for x in range(width):
            val = int(board[y, x])
            if val != prev and prev >= 0:
                if 0 < gap < 16:
                    cuda.atomic.add(shared_gaps, gap, 1)
                gap = 0
            gap += 1
            prev = val

        cuda.syncthreads()

        # Reduce shared gaps into global output
        if tid < 16 and shared_gaps[tid] > 0:
            cuda.atomic.add(edge_scores_out, tid, shared_gaps[tid])

    _edge_kernel[height, width](board, width, height, edge_scores_out)
    cuda.synchronize()
