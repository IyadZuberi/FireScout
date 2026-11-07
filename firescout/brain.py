"""FireScout robot brain: pure numpy, never imports mujoco or scene ground truth.

Grid convention: occ[i, j] with i = int(y / res) (row, north), j = int(x / res) (col, east).
"""
from __future__ import annotations

import heapq
import itertools
import math

import cv2
import numpy as np

RES = 0.25
NO_GO_C = 250.0
AMBIENT_C = 20.0


# ======================================================================= map

class Grid:
    def __init__(self, width: float, height: float, res: float = RES):
        self.res = res
        self.width, self.height = width, height
        self.rows, self.cols = int(math.ceil(height / res)), int(math.ceil(width / res))
        shape = (self.rows, self.cols)
        self.occ = np.full(shape, -1, np.int8)        # -1 unknown, 0 free, 1 wall
        self.heat = np.full(shape, AMBIENT_C, np.float32)
        self.gas = np.zeros(shape, np.float32)
        self.scanned = np.zeros(shape, bool)

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (min(max(int(y / self.res), 0), self.rows - 1),
                min(max(int(x / self.res), 0), self.cols - 1))

    def to_world(self, i: int, j: int) -> tuple[float, float]:
        return ((j + 0.5) * self.res, (i + 0.5) * self.res)

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.rows and 0 <= j < self.cols


def _line(a, b):
    """Bresenham cells from a to b inclusive."""
    (i0, j0), (i1, j1) = a, b
    di, dj = abs(i1 - i0), abs(j1 - j0)
    si, sj = (1 if i1 > i0 else -1), (1 if j1 > j0 else -1)
    err = dj - di
    out = []
    while True:
        out.append((i0, j0))
        if (i0, j0) == (i1, j1):
            return out
        e2 = 2 * err
        if e2 > -di:
            err -= di
            j0 += sj
        if e2 < dj:
            err += dj
            i0 += si


def integrate_ranges(grid: Grid, pose, ranges, max_range: float = 6.0) -> None:
    x, y, yaw = pose
    a = grid.to_cell(x, y)
    for k, r in enumerate(ranges):
        th = yaw + math.radians(10 * k)
        ex, ey = x + r * math.cos(th), y + r * math.sin(th)
        cells = [c for c in _line(a, (int(ey / grid.res), int(ex / grid.res))) if grid.inside(*c)]
        if not cells:
            continue
        hit = r < max_range - 0.05
        for c in cells[:-1] if hit else cells:
            grid.occ[c] = 0
        if hit:
            grid.occ[cells[-1]] = 1


def cost_map(grid: Grid, no_go_c: float = NO_GO_C) -> np.ndarray:
    """Traversal cost per cell; np.inf = forbidden (wall, inflated wall, or > no_go_c)."""
    heat_eff = np.maximum(grid.heat, cv2.GaussianBlur(grid.heat, (0, 0), 0.5 / grid.res))
    cost = 1.0 + np.maximum(0.0, (heat_eff - 40.0) / 50.0) + grid.gas / 500.0
    cost = np.where(grid.occ == -1, 1.5 + cost - 1.0, cost)
    walls = cv2.dilate((grid.occ == 1).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cost[walls | (heat_eff > no_go_c)] = np.inf
    return cost.astype(np.float32)


_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def astar(cost: np.ndarray, start, goal):
    rows, cols = cost.shape
    if not (np.isfinite(cost[goal])):
        return None
    h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
    g = {start: 0.0}
    came = {}
    pq = [(h(start), 0.0, start)]
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, math.inf):
            continue
        for di, dj, step in _NB:
            n = (cur[0] + di, cur[1] + dj)
            if not (0 <= n[0] < rows and 0 <= n[1] < cols) or not np.isfinite(cost[n]):
                continue
            if di and dj and not (np.isfinite(cost[cur[0] + di, cur[1]]) and np.isfinite(cost[cur[0], cur[1] + dj])):
                continue  # no corner cutting
            ng = gc + step * float(cost[n])
            if ng < g.get(n, math.inf):
                g[n], came[n] = ng, cur
                heapq.heappush(pq, (ng + h(n), ng, n))
    return None


def path_cost(cost: np.ndarray, path) -> float:
    return float(sum(cost[c] for c in path[1:]))


def frontiers(grid: Grid) -> list[tuple[int, int]]:
    free = grid.occ == 0
    unk = grid.occ == -1
    nb = np.zeros_like(unk)
    nb[1:, :] |= unk[:-1, :]
    nb[:-1, :] |= unk[1:, :]
    nb[:, 1:] |= unk[:, :-1]
    nb[:, :-1] |= unk[:, 1:]
    return [(int(i), int(j)) for i, j in np.argwhere(free & nb)]


def rescue_route(cost: np.ndarray, start, targets):
    """Best visiting order over <=4 targets (brute force). Returns (order, path) or None."""
    best = None
    for perm in itertools.permutations(range(len(targets))):
        path, total, cur = [start], 0.0, start
        for k in perm:
            leg = astar(cost, cur, targets[k])
            if leg is None:
                break
            total += path_cost(cost, leg)
            path += leg[1:]
            cur = targets[k]
        else:
            if best is None or total < best[0]:
                best = (total, list(perm), path)
    return None if best is None else (best[1], best[2])
