"""CuPy-backed CUDA kernels for ARC-AGI-3 runtime hotspots."""

from __future__ import annotations

import logging

import numpy as np

from agi_core.cuda import get_cupy

logger = logging.getLogger(__name__)

_RAW_KERNELS: dict[str, object] = {}


def _cp():
    return get_cupy()


def _raw_kernel(name: str, code: str):
    kernel = _RAW_KERNELS.get(name)
    if kernel is not None:
        return kernel
    cp = _cp()
    kernel = cp.RawKernel(code, name)
    _RAW_KERNELS[name] = kernel
    return kernel


def _to_numpy(arr):
    cp = _cp()
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def _assign_array(dst, src) -> None:
    cp = _cp()
    if isinstance(dst, cp.ndarray):
        dst[...] = src
    else:
        dst[...] = cp.asnumpy(src) if isinstance(src, cp.ndarray) else src


def _launch_salience_kernel(board, px, py, goal_xs, goal_ys, out, prev_board=None):
    cp = _cp()
    board_cp = cp.asarray(board, dtype=cp.int8)
    prev_cp = cp.asarray(prev_board, dtype=cp.int8) if prev_board is not None else cp.zeros_like(board_cp)
    goal_xs_cp = cp.asarray(goal_xs, dtype=cp.int32)
    goal_ys_cp = cp.asarray(goal_ys, dtype=cp.int32)
    out_cp = cp.zeros(board_cp.shape, dtype=cp.float32)
    h, w = board_cp.shape
    kernel = _raw_kernel(
        "salience_kernel",
        r"""
        extern "C" __global__
        void salience_kernel(
            const signed char* board,
            const signed char* prev_board,
            const int width,
            const int height,
            const int px,
            const int py,
            const int* goal_xs,
            const int* goal_ys,
            const int n_goals,
            const int has_prev,
            float* out
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }
            int idx = y * width + x;
            float s = 0.0f;
            if (has_prev && prev_board[idx] != board[idx]) {
                s += 0.25f;
            }
            int dist = abs(x - px) + abs(y - py);
            if (dist < 20) {
                s += 0.30f * (1.0f - ((float)dist / 20.0f));
            }
            float best = 16.0f;
            for (int i = 0; i < n_goals; ++i) {
                float d = (float)(abs(x - goal_xs[i]) + abs(y - goal_ys[i]));
                if (d < best) {
                    best = d;
                }
            }
            if (best < 16.0f) {
                s += 0.20f * (1.0f - best / 16.0f);
            }
            if (x < 3 || x >= width - 3 || y < 3 || y >= height - 3) {
                s += 0.10f;
            }
            out[idx] = s > 1.0f ? 1.0f : s;
        }
        """,
    )
    threads = (16, 16, 1)
    blocks = ((w + 15) // 16, (h + 15) // 16, 1)
    kernel(
        blocks,
        threads,
        (
            board_cp,
            prev_cp,
            np.int32(w),
            np.int32(h),
            np.int32(px),
            np.int32(py),
            goal_xs_cp,
            goal_ys_cp,
            np.int32(int(goal_xs_cp.size)),
            np.int32(1 if prev_board is not None else 0),
            out_cp,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(out, out_cp)


def _launch_astar_kernel(walkable, starts_x, starts_y, goals_x, goals_y, paths_out, path_lens):
    cp = _cp()
    walkable_cp = cp.asarray(walkable, dtype=cp.uint8)
    starts_x_cp = cp.asarray(starts_x, dtype=cp.int32)
    starts_y_cp = cp.asarray(starts_y, dtype=cp.int32)
    goals_x_cp = cp.asarray(goals_x, dtype=cp.int32)
    goals_y_cp = cp.asarray(goals_y, dtype=cp.int32)
    paths_out_cp = cp.full(paths_out.shape, -1, dtype=cp.int32)
    path_lens_cp = cp.zeros(path_lens.shape, dtype=cp.int32)
    n_searches = int(starts_x_cp.size)
    kernel = _raw_kernel(
        "astar_kernel",
        r"""
        extern "C" __global__
        void astar_kernel(
            const unsigned char* walkable,
            const int width,
            const int height,
            const int* starts_x,
            const int* starts_y,
            const int* goals_x,
            const int* goals_y,
            const int n_searches,
            int* paths_out,
            int* path_lens_out
        ) {
            int bid = blockIdx.x;
            if (bid >= n_searches || threadIdx.x != 0) {
                return;
            }
            const int max_cells = 4096;
            const int max_open = 256;
            int open_nodes[max_open];
            int open_count = 0;
            float g_score[max_cells];
            unsigned char closed[max_cells];
            int parent[max_cells];
            for (int i = 0; i < max_cells; ++i) {
                g_score[i] = 1.0e20f;
                closed[i] = 0;
                parent[i] = -1;
            }
            int sx = starts_x[bid];
            int sy = starts_y[bid];
            int gx = goals_x[bid];
            int gy = goals_y[bid];
            int start_idx = sy * width + sx;
            int goal_idx = gy * width + gx;
            open_nodes[0] = start_idx;
            open_count = 1;
            g_score[start_idx] = 0.0f;
            bool found = false;
            while (open_count > 0) {
                int best_pos = 0;
                int best_idx = open_nodes[0];
                float best_f = g_score[best_idx] + (float)(abs((best_idx % width) - gx) + abs((best_idx / width) - gy));
                for (int i = 1; i < open_count; ++i) {
                    int idx = open_nodes[i];
                    int x = idx % width;
                    int y = idx / width;
                    float f = g_score[idx] + (float)(abs(x - gx) + abs(y - gy));
                    if (f < best_f) {
                        best_f = f;
                        best_idx = idx;
                        best_pos = i;
                    }
                }
                open_nodes[best_pos] = open_nodes[open_count - 1];
                --open_count;
                if (closed[best_idx]) {
                    continue;
                }
                closed[best_idx] = 1;
                if (best_idx == goal_idx) {
                    found = true;
                    break;
                }
                int bx = best_idx % width;
                int by = best_idx / width;
                const int dxs[4] = {-1, 1, 0, 0};
                const int dys[4] = {0, 0, -1, 1};
                for (int n = 0; n < 4; ++n) {
                    int nx = bx + dxs[n];
                    int ny = by + dys[n];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    int ni = ny * width + nx;
                    if (closed[ni]) {
                        continue;
                    }
                    int walkable_idx = bid * width * height + ny * width + nx;
                    if (walkable[walkable_idx] == 0) {
                        continue;
                    }
                    float new_g = g_score[best_idx] + 1.0f;
                    if (new_g < g_score[ni]) {
                        g_score[ni] = new_g;
                        parent[ni] = best_idx;
                        bool present = false;
                        for (int i = 0; i < open_count; ++i) {
                            if (open_nodes[i] == ni) {
                                present = true;
                                break;
                            }
                        }
                        if (!present && open_count < max_open) {
                            open_nodes[open_count++] = ni;
                        }
                    }
                }
            }
            if (!found) {
                path_lens_out[bid] = 0;
                return;
            }
            int rev_path[256];
            int total_len = 0;
            int cur = goal_idx;
            while (cur >= 0 && total_len < 256) {
                rev_path[total_len++] = cur;
                if (cur == start_idx) {
                    break;
                }
                cur = parent[cur];
            }
            if (total_len == 0 || rev_path[total_len - 1] != start_idx) {
                path_lens_out[bid] = 0;
                return;
            }
            path_lens_out[bid] = total_len;
            for (int i = 0; i < total_len; ++i) {
                paths_out[bid * 256 + i] = rev_path[total_len - 1 - i];
            }
        }
        """,
    )
    kernel(
        (max(1, n_searches), 1, 1),
        (1, 1, 1),
        (
            walkable_cp,
            np.int32(int(walkable_cp.shape[2])),
            np.int32(int(walkable_cp.shape[1])),
            starts_x_cp,
            starts_y_cp,
            goals_x_cp,
            goals_y_cp,
            np.int32(n_searches),
            paths_out_cp,
            path_lens_cp,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(paths_out, paths_out_cp)
    _assign_array(path_lens, path_lens_cp)


def _launch_init_labels(board, labels_out, width, height):
    cp = _cp()
    board_cp = cp.asarray(board, dtype=cp.int8)
    labels_cp = labels_out if isinstance(labels_out, cp.ndarray) else cp.asarray(labels_out, dtype=cp.int32)
    kernel = _raw_kernel(
        "init_labels_kernel",
        r"""
        extern "C" __global__
        void init_labels_kernel(const signed char* board, int* labels, const int width, const int height) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }
            int idx = y * width + x;
            labels[idx] = board[idx] < 0 ? -1 : idx;
        }
        """,
    )
    blocks = ((width + 15) // 16, (height + 15) // 16, 1)
    kernel(blocks, (16, 16, 1), (board_cp, labels_cp, np.int32(width), np.int32(height)))
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(labels_out, labels_cp)


def _launch_propagate_labels(board, labels, width, height):
    cp = _cp()
    board_cp = cp.asarray(board, dtype=cp.int8)
    labels_cp = labels if isinstance(labels, cp.ndarray) else cp.asarray(labels, dtype=cp.int32)
    converged_cp = cp.ones(1, dtype=cp.int32)
    kernel = _raw_kernel(
        "propagate_labels_kernel",
        r"""
        extern "C" __global__
        void propagate_labels_kernel(
            const signed char* board,
            int* labels,
            const int width,
            const int height,
            int* converged
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }
            int idx = y * width + x;
            int my_label = labels[idx];
            if (my_label < 0) {
                return;
            }
            signed char my_color = board[idx];
            int new_label = my_label;
            if (x > 0) {
                int nidx = idx - 1;
                if (board[nidx] == my_color) {
                    int nl = labels[nidx];
                    if (nl >= 0 && nl < new_label) {
                        new_label = nl;
                    }
                }
            }
            if (x < width - 1) {
                int nidx = idx + 1;
                if (board[nidx] == my_color) {
                    int nl = labels[nidx];
                    if (nl >= 0 && nl < new_label) {
                        new_label = nl;
                    }
                }
            }
            if (y > 0) {
                int nidx = idx - width;
                if (board[nidx] == my_color) {
                    int nl = labels[nidx];
                    if (nl >= 0 && nl < new_label) {
                        new_label = nl;
                    }
                }
            }
            if (y < height - 1) {
                int nidx = idx + width;
                if (board[nidx] == my_color) {
                    int nl = labels[nidx];
                    if (nl >= 0 && nl < new_label) {
                        new_label = nl;
                    }
                }
            }
            if (new_label != my_label) {
                labels[idx] = new_label;
                atomicExch(converged, 0);
            }
        }
        """,
    )
    blocks = ((width + 15) // 16, (height + 15) // 16, 1)
    kernel(
        blocks,
        (16, 16, 1),
        (board_cp, labels_cp, np.int32(width), np.int32(height), converged_cp),
    )
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(labels, labels_cp)
    return bool(int(cp.asnumpy(converged_cp)[0]))


def _launch_compact_labels(board, labels, width, height):
    cp = _cp()
    board_np = _to_numpy(board)
    labels_np = _to_numpy(labels)
    offsets = []
    for y in range(height):
        for x in range(width):
            my_label = int(labels_np[y, x])
            if my_label < 0:
                continue
            offsets.append((my_label, x, y, int(board_np[y, x])))
    if not offsets:
        return []
    offset_np = np.asarray(offsets, dtype=np.int32)
    offset_np = offset_np[offset_np[:, 0].argsort()]
    return _components_from_sorted(offset_np, board_np.tolist(), width, height)


def _components_from_sorted(
    offsets: np.ndarray,
    board: list[list[int]],
    width: int,
    height: int,
) -> list[dict]:
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
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            j += 1
        size = j - i
        sx = sum(c[0] for c in cells) // size
        sy = sum(c[1] for c in cells) // size
        tb = min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1
        results.append(
            {
                "label": label,
                "color": color,
                "cells": cells,
                "bbox": (min_x, min_y, max_x, max_y),
                "size": size,
                "center": (sx, sy),
                "touches_border": tb,
            }
        )
        i = j
    return results


def _compact_labels_cpu(labels, board, width, height):
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
                    "min_x": x,
                    "max_x": x,
                    "min_y": y,
                    "max_y": y,
                }
            entry = comp_map[label]
            entry["cells"].append((x, y))
            if x < entry["min_x"]:
                entry["min_x"] = x
            if x > entry["max_x"]:
                entry["max_x"] = x
            if y < entry["min_y"]:
                entry["min_y"] = y
            if y > entry["max_y"]:
                entry["max_y"] = y

    results = []
    for entry in comp_map.values():
        cells = entry["cells"]
        bbox = (entry["min_x"], entry["min_y"], entry["max_x"], entry["max_y"])
        size = len(cells)
        sx = sum(x for x, _ in cells) // size
        sy = sum(y for _, y in cells) // size
        tb = (
            entry["min_x"] == 0
            or entry["min_y"] == 0
            or entry["max_x"] == width - 1
            or entry["max_y"] == height - 1
        )
        results.append(
            {
                "label": entry["label"],
                "color": entry["color"],
                "cells": cells,
                "bbox": bbox,
                "size": size,
                "center": (sx, sy),
                "touches_border": tb,
            }
        )
    return results


def _launch_mirror_placement_kernel(
    goal_bitmask, shape_mask, shape_w, shape_h, horizontal, axis, width, height, best_out
):
    cp = _cp()
    goal_cp = cp.asarray(goal_bitmask, dtype=cp.uint8)
    shape_cp = cp.asarray(shape_mask, dtype=cp.uint8)
    best_cp = best_out if isinstance(best_out, cp.ndarray) else cp.asarray(best_out, dtype=cp.int32)
    kernel = _raw_kernel(
        "mirror_placement_kernel",
        r"""
        extern "C" __global__
        void mirror_placement_kernel(
            const unsigned char* goal,
            const unsigned char* shape,
            const int sw,
            const int sh,
            const int horiz,
            const int axis,
            const int width,
            const int height,
            int* best
        ) {
            int tx = blockDim.x * blockIdx.x + threadIdx.x;
            int ty = blockDim.y * blockIdx.y + threadIdx.y;
            int max_tx = width - sw;
            int max_ty = height - sh;
            if (tx > max_tx || ty > max_ty) {
                return;
            }
            int hits = 0;
            for (int dy = 0; dy < sh; ++dy) {
                for (int dx = 0; dx < sw; ++dx) {
                    if (shape[dy * sw + dx] == 0) {
                        continue;
                    }
                    int rx;
                    int ry;
                    if (horiz) {
                        rx = tx + dx;
                        ry = 2 * axis - (ty + dy);
                    } else {
                        rx = 2 * axis - (tx + dx);
                        ry = ty + dy;
                    }
                    if (rx >= 0 && rx < width && ry >= 0 && ry < height && goal[ry * width + rx]) {
                        ++hits;
                    }
                }
            }
            if (hits <= 0) {
                return;
            }
            int cx = tx + sw / 2;
            int cy = ty + sh / 2;
            int max_dist = width + height;
            int dist = abs(cx - tx) + abs(cy - ty);
            int neg_dist = max_dist - dist;
            int packed = (hits << 20) | neg_dist;
            atomicMax(&best[0], packed);
            atomicMax(&best[1], cx);
            atomicMax(&best[2], cy);
            atomicMax(&best[3], hits);
        }
        """,
    )
    max_tx = max(0, width - shape_w)
    max_ty = max(0, height - shape_h)
    tx_dim = max(1, min(16, max_tx + 1))
    ty_dim = max(1, min(16, max_ty + 1))
    blocks = ((max_tx + tx_dim) // tx_dim, (max_ty + ty_dim) // ty_dim, 1)
    kernel(
        blocks,
        (tx_dim, ty_dim, 1),
        (
            goal_cp,
            shape_cp,
            np.int32(shape_w),
            np.int32(shape_h),
            np.int32(int(horizontal)),
            np.int32(axis),
            np.int32(width),
            np.int32(height),
            best_cp,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(best_out, best_cp)


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
    cp = _cp()
    goal_cp = cp.asarray(goal_bitmask, dtype=cp.uint8)
    shape_cp = cp.asarray(shape_mask, dtype=cp.uint8)
    score_cp = score_out if isinstance(score_out, cp.ndarray) else cp.asarray(score_out, dtype=cp.int32)
    kernel = _raw_kernel(
        "reflection_score_kernel",
        r"""
        extern "C" __global__
        void reflection_score_kernel(
            const unsigned char* goal,
            const unsigned char* shape,
            const int sw,
            const int sh,
            const int horiz,
            const int axis,
            const int width,
            const int height,
            int* score_out
        ) {
            int tx = blockDim.x * blockIdx.x + threadIdx.x;
            int ty = blockDim.y * blockIdx.y + threadIdx.y;
            int max_tx = width - sw;
            int max_ty = height - sh;
            if (tx > max_tx || ty > max_ty) {
                return;
            }
            int hits = 0;
            for (int dy = 0; dy < sh; ++dy) {
                for (int dx = 0; dx < sw; ++dx) {
                    if (shape[dy * sw + dx] == 0) {
                        continue;
                    }
                    int rx;
                    int ry;
                    if (horiz) {
                        rx = tx + dx;
                        ry = 2 * axis - (ty + dy);
                    } else {
                        rx = 2 * axis - (tx + dx);
                        ry = ty + dy;
                    }
                    if (rx >= 0 && rx < width && ry >= 0 && ry < height && goal[ry * width + rx]) {
                        ++hits;
                    }
                }
            }
            score_out[ty * width + tx] = hits;
        }
        """,
    )
    max_tx = max(0, width - shape_w)
    max_ty = max(0, height - shape_h)
    tx_dim = max(1, min(16, max_tx + 1))
    ty_dim = max(1, min(16, max_ty + 1))
    blocks = ((max_tx + tx_dim) // tx_dim, (max_ty + ty_dim) // ty_dim, 1)
    kernel(
        blocks,
        (tx_dim, ty_dim, 1),
        (
            goal_cp,
            shape_cp,
            np.int32(shape_w),
            np.int32(shape_h),
            np.int32(int(horizontal)),
            np.int32(axis),
            np.int32(width),
            np.int32(height),
            score_cp,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(score_out, score_cp)


def _launch_uniformity_kernel(board, uniformity_out, width, height):
    cp = _cp()
    board_cp = cp.asarray(board, dtype=cp.int8)
    uniformity_cp = uniformity_out if isinstance(uniformity_out, cp.ndarray) else cp.asarray(uniformity_out, dtype=cp.float32)
    kernel = _raw_kernel(
        "uniformity_kernel",
        r"""
        extern "C" __global__
        void uniformity_kernel(
            const signed char* board,
            const int width,
            const int height,
            float* uniformity_out
        ) {
            int scale = blockIdx.x;
            if (threadIdx.x != 0) {
                return;
            }
            if (scale < 2 || scale > 16) {
                uniformity_out[scale] = 0.0f;
                return;
            }
            int bw = width / scale;
            int bh = height / scale;
            if (bw < 4 || bh < 4) {
                uniformity_out[scale] = 0.0f;
                return;
            }
            int total = bw * bh;
            int uniform = 0;
            for (int by = 0; by < bh; ++by) {
                for (int bx = 0; bx < bw; ++bx) {
                    int px = bx * scale;
                    int py = by * scale;
                    signed char color = board[py * width + px];
                    bool ok = true;
                    for (int dy = 0; dy < scale && ok; ++dy) {
                        for (int dx = 0; dx < scale; ++dx) {
                            if (board[(py + dy) * width + (px + dx)] != color) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if (ok) {
                        ++uniform;
                    }
                }
            }
            uniformity_out[scale] = total > 0 ? ((float)uniform / (float)total) : 0.0f;
        }
        """,
    )
    kernel((17, 1, 1), (1, 1, 1), (board_cp, np.int32(width), np.int32(height), uniformity_cp))
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(uniformity_out, uniformity_cp)


def _launch_edge_kernel(board, edge_scores_out, width, height):
    cp = _cp()
    board_cp = cp.asarray(board, dtype=cp.int8)
    edge_cp = edge_scores_out if isinstance(edge_scores_out, cp.ndarray) else cp.asarray(edge_scores_out, dtype=cp.int32)
    kernel = _raw_kernel(
        "edge_kernel",
        r"""
        extern "C" __global__
        void edge_kernel(
            const signed char* board,
            const int width,
            const int height,
            int* edge_scores_out
        ) {
            int y = blockIdx.x;
            if (y >= height || threadIdx.x != 0) {
                return;
            }
            int prev = -2;
            int gap = 0;
            for (int x = 0; x < width; ++x) {
                int val = (int)board[y * width + x];
                if (val != prev && prev >= 0) {
                    if (gap > 0 && gap < 16) {
                        atomicAdd(&edge_scores_out[gap], 1);
                    }
                    gap = 0;
                }
                ++gap;
                prev = val;
            }
        }
        """,
    )
    kernel((height, 1, 1), (1, 1, 1), (board_cp, np.int32(width), np.int32(height), edge_cp))
    cp.cuda.runtime.deviceSynchronize()
    _assign_array(edge_scores_out, edge_cp)
