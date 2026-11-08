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
        self.o2 = np.full(shape, 20.9, np.float32)     # measured O2 % (min seen)
        self.maxz = np.zeros(shape, np.float32)        # highest surface point seen in the cell (depth camera)
        self.topz = np.zeros(shape, np.float32)        # height of a seen object TOP edge (depth jumps behind it)

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
    occ = (grid.occ == 1).astype(np.uint8)
    margin = cv2.dilate(occ, np.ones((5, 5), np.uint8)) > 0
    cost[margin] += 4.0                                   # prefer the middle of corridors and doorways
    touching = cv2.dilate(occ, np.ones((3, 3), np.uint8)) > 0
    cost[touching | (heat_eff > no_go_c)] = np.inf        # robot half-width ~0.2 m: cells next to obstacles forbidden
    return cost.astype(np.float32)


def free_escape(cost: np.ndarray, grid: Grid, here) -> None:
    """If the robot is already inside an inflated cell, let it drive out (never through a real wall or fire)."""
    if np.isfinite(cost[here]):
        return
    i, j = here
    i0, i1, j0, j1 = max(0, i - 2), i + 3, max(0, j - 2), j + 3
    blk = cost[i0:i1, j0:j1]
    wall = grid.occ[i0:i1, j0:j1] == 1
    hot = grid.heat[i0:i1, j0:j1] > NO_GO_C
    blk[~np.isfinite(blk) & ~wall & ~hot] = 25.0


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


# ================================================================ perception
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

HFOV_DEG = 57.0
VIEW_RANGE = 4.0
FIRE_C = 150.0
WARM_MASK_C = 28.0
CAS_LO, CAS_HI = 30.0, 42.0


@dataclass
class Blob:
    kind: Literal['fire', 'warm']
    xy: tuple[float, float]
    temp_c: float
    width_m: float
    height_m: float
    box: tuple[int, int, int, int] = (0, 0, 0, 0)  # u, v, w, h in image px (for overlay)


def detect_blobs(thermal, depth, pose, hfov_deg: float = HFOV_DEG) -> list[Blob]:
    """Connected warm regions (>28 C) -> world-frame blobs; fire if peak > FIRE_C."""
    H, W = thermal.shape
    fx = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    cx = (W - 1) / 2
    x, y, yaw = pose
    n, lab, stats, cent = cv2.connectedComponentsWithStats((thermal > WARM_MASK_C).astype(np.uint8))
    out = []
    for k in range(1, n):
        u, v, w, h, area = (int(s) for s in stats[k])
        if area < 6:
            continue
        mask = lab == k
        dvals = depth[mask]
        dvals = dvals[np.isfinite(dvals) & (dvals > 0)]
        if dvals.size == 0:
            continue
        temps = thermal[mask]
        peak = float(temps.max())
        kind = 'fire' if peak > FIRE_C else 'warm'
        if kind == 'warm':  # a body is a compact warm object, not a smooth heat gradient on a wall
            ring = (cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~mask
            if ring.any() and float(np.median(temps)) - float(np.median(thermal[ring])) < 4.0:
                continue
        if kind == 'fire':
            # range only from the hot core (blur halo spills onto nearer door frames); flames are translucent,
            # so take the near side of the core; ignore fires beyond radar range
            core = mask & (thermal >= max(FIRE_C, 0.7 * peak))
            cd = depth[core]
            cd = cd[np.isfinite(cd) & (cd > 0)]
            if cd.size == 0:
                continue
            r = float(np.percentile(cd, 20)) + 0.3  # near surface -> centre of a ~0.6 m fire seat
            if r > 6.3:
                continue
        else:
            r = float(np.median(dvals))
        bearing = yaw + math.atan((cx - cent[k][0]) / fx)
        hm, wm = r * h / fx, r * w / fx
        if kind == 'warm' and (wm > 2.5 or (hm > 1.0 and not (hm <= 2.0 and wm <= 0.9))):
            continue  # wall-sized warm patch: neither a collapsed nor an upright person
        out.append(Blob(kind, (x + r * math.cos(bearing), y + r * math.sin(bearing)),
                        peak if kind == 'fire' else float(np.median(temps)),
                        r * w / fx, r * h / fx, (u, v, w, h)))
    return out


@dataclass
class Track:
    id: int
    kind: str
    xy: tuple[float, float]
    temps: deque
    hits: deque
    width_m: float
    height_m: float
    breath: float | None = None
    conf: float = 0.0
    state: Literal['candidate', 'verify', 'pinned', 'rejected'] = 'candidate'
    ic: bool | None = None
    first_seen: float = 0.0
    breath_n: int = 0
    responsive: bool = False   # heard calling (mic bearing matched this track)
    verify_tries: int = 0
    instruct: str | None = None   # 'EVACUATE' | 'SHELTER' once the robot has spoken to this person
    verify_from: tuple | None = None  # where the last breath check was taken from
    priority: str = ''


CAM_H, CAM_FWD = 0.33, 0.15  # thermal camera mount on the robot


def project_pixels(depth, pose, step: int = 4, hfov_deg: float = HFOV_DEG):
    """Sampled pixels -> world points (x, y, z) using the depth/radar range; returns points + pixel indices."""
    H, W = depth.shape
    fx = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    v, u = np.mgrid[step // 2:H:step, step // 2:W:step]
    dp = depth[v, u]
    ok = np.isfinite(dp) & (dp > 0.05) & (dp < 6.0)
    v, u, dp = v[ok], u[ok], dp[ok]
    left, up = (W / 2 - u) / fx * dp, (H / 2 - v) / fx * dp
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = x + CAM_FWD * c, y + CAM_FWD * s
    return cx + dp * c - left * s, cy + dp * s + left * c, CAM_H + up, (v, u), dp


def in_view(pose, xy, hfov_deg: float = HFOV_DEG, rng: float = VIEW_RANGE) -> bool:
    x, y, yaw = pose
    dx, dy = xy[0] - x, xy[1] - y
    if math.hypot(dx, dy) > rng:
        return False
    d = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
    return abs(d) <= math.radians(hfov_deg) / 2


class Tracker:
    def __init__(self, gate_m: float = 1.2):
        self.tracks: list[Track] = []
        self.gate = gate_m
        self._next = 1

    def update(self, blobs: list[Blob], t: float, pose) -> list[Track]:
        matched = set()
        for b in blobs:
            best, bd = None, self.gate
            for tr in self.tracks:
                dd = math.dist(tr.xy, b.xy)
                if tr.kind == b.kind and dd <= bd and tr.id not in matched:
                    best, bd = tr, dd
            if best is None:
                best = Track(self._next, b.kind, b.xy, deque(maxlen=5), deque(maxlen=5),
                             b.width_m, b.height_m, first_seen=t)
                self._next += 1
                self.tracks.append(best)
            else:
                best.xy = (0.7 * best.xy[0] + 0.3 * b.xy[0], 0.7 * best.xy[1] + 0.3 * b.xy[1])
                best.width_m, best.height_m = max(best.width_m, b.width_m), max(best.height_m, b.height_m)
            best.temps.append(b.temp_c)
            best.hits.append(True)
            matched.add(best.id)
        for tr in self.tracks:
            if tr.id not in matched and in_view(pose, tr.xy):
                tr.hits.append(False)
        keep = []
        for tr in self.tracks:  # merge duplicates of one body seen from different angles
            dup = next((k for k in keep if k.kind == tr.kind and math.dist(k.xy, tr.xy) < self.gate), None)
            if dup is None:
                keep.append(tr)
            else:
                dup.width_m, dup.height_m = max(dup.width_m, tr.width_m), max(dup.height_m, tr.height_m)
                if tr.state == 'pinned' or (tr.breath or 0) > (dup.breath or 0):
                    dup.state, dup.breath, dup.conf = tr.state, tr.breath, tr.conf
        self.tracks = keep
        return self.tracks


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class Calibrator:
    """Logistic casualty-confidence model; IC feedback = one SGD step (live learning)."""

    def __init__(self, path: str = 'feedback.jsonl', lr: float = 0.5, weights_path: str | None = None):
        self.w = np.array([-4.0, 1.5, 1.5, 1.0, 2.0, 3.0])   # hand-set prior, replaced by trained weights if present
        self.path, self.lr, self.weights_path = path, lr, weights_path
        self.source = 'hand-set prior'
        if weights_path:
            try:
                with open(weights_path, encoding='utf-8') as fh:
                    d = json.load(fh)
                self.w = np.array(d['w'], float)
                self.source = d.get('source', 'saved weights')
            except (OSError, ValueError, KeyError):
                pass

    @staticmethod
    def features(tr: Track) -> np.ndarray:
        temp = float(np.median(tr.temps)) if tr.temps else 0.0
        w, h = tr.width_m, max(tr.height_m, 1e-3)
        return np.array([
            1.0,
            1.0 if CAS_LO <= temp <= CAS_HI else 0.0,
            math.exp(-((max(w, h) - 1.7) / 0.5) ** 2),
            1.0 if w / h > 1.8 else 0.0,
            (sum(tr.hits) / len(tr.hits)) if tr.hits else 0.0,
            tr.breath if tr.breath is not None else 0.5,
        ])

    def conf(self, tr: Track) -> float:
        return _sigmoid(float(self.w @ self.features(tr)))

    def update(self, tr: Track, label: bool) -> None:
        f = self.features(tr)
        self.w += self.lr * ((1.0 if label else 0.0) - _sigmoid(float(self.w @ f))) * f
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'features': f.round(4).tolist(), 'label': bool(label), 'track': tr.id}) + '\n')
        if self.weights_path:  # keep what the IC taught for the next mission
            self.source = 'trained + IC feedback'
            with open(self.weights_path, 'w', encoding='utf-8') as fh:
                json.dump({'w': self.w.round(6).tolist(), 'source': self.source}, fh)


# ================================================================== brain
import time

PIN_CONF = 0.8
VERIFY_CONF = 0.4
REJECT_CONF = 0.2
BREATH_OK = 0.5
BREATH_RANGE = 2.5
BREATH_SAMPLES = 20  # ~1 s of radar micro-motion before deciding
VERIFY_TIMEOUT = 15.0
SELF_TEMP_LIMIT = 60.0
SPEED = 0.4
SLOW_SPEED = 0.2
FAST_SPEED = 0.7          # known, scanned, cool route: move fast
VOICE_MATCH_RAD = 0.35    # mic bearing within this of a warm track = that person is calling
VOICE_MEMORY_S = 25.0
WALL_Z, FURN_MIN_Z = 1.4, 0.1
SPEAKER_RANGE = 5.0       # m: a 10 W speaker is understood through smoke at this range
EVAC_MAX_C = 60.0         # a walking civilian's route must stay below this (measured) temperature
EVAC_FIRE_CLEARANCE = 1.5
SAY_TEXT = {
    'EVACUATE': 'This is the fire service. Stay low, keep one hand on the wall and walk out the way I came in. '
                'Firefighters are at the door.',
    'SHELTER': 'This is the fire service. Stay where you are. Get low, cover your mouth and keep the door shut. '
               'Firefighters are coming to you.',
}


@dataclass
class Card:
    t: float
    kind: str  # REPLAN VERIFY PIN REJECT POSSIBLE RETREAT COMMS LEARN HOLD HEAR PRIORITY IC
    trigger: str
    detail: str
    change: str
    rule: str

    def __str__(self) -> str:
        mm, ss = divmod(int(self.t), 60)
        return f'[{mm:02d}:{ss:02d}] {self.kind} trigger={self.trigger} {self.detail} | {self.change} | rule={self.rule}'


@dataclass
class Command:
    vx: float
    vy: float
    wz: float
    want_breath_xy: tuple[float, float] | None = None


def dijkstra(cost: np.ndarray, start):
    """Distances + parents from start over the cost map (8-connected)."""
    rows, cols = cost.shape
    dist = np.full(cost.shape, np.inf)
    parent = {}
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        dc, cur = heapq.heappop(pq)
        if dc > dist[cur]:
            continue
        for di, dj, step in _NB:
            n = (cur[0] + di, cur[1] + dj)
            if not (0 <= n[0] < rows and 0 <= n[1] < cols) or not np.isfinite(cost[n]):
                continue
            if di and dj and not (np.isfinite(cost[cur[0] + di, cur[1]]) and np.isfinite(cost[cur[0], cur[1] + dj])):
                continue
            nd = dc + step * float(cost[n])
            if nd < dist[n]:
                dist[n], parent[n] = nd, cur
                heapq.heappush(pq, (nd, n))
    return dist, parent


def unwind(parent, goal):
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Brain:
    def __init__(self, width: float, height: float, start_xy, feedback_path: str = 'feedback.jsonl',
                 weights_path: str | None = None):
        self.grid = Grid(width, height)
        self.tracker = Tracker()
        self.cal = Calibrator(feedback_path, weights_path=weights_path)
        self.start_cell = self.grid.to_cell(*start_xy)
        self.mode = 'EXPLORE'
        self.path: list = []
        self.goal = None
        self.verify_id: int | None = None
        self.rescue = None
        self.exit_path = None
        self.cards: list[Card] = []
        self.outbox: list[Card] = []
        self.last_comms_cell = self.start_cell
        self.timings: dict[str, float] = {}
        self.fires: list[tuple[float, float, float]] = []
        self.t = 0.0
        self._comms = True
        self._ticks = 0
        self._last_replan: dict[str, float] = {}
        self._no_target = 0
        self._verify_fail: dict[int, float] = {}
        self._dist = None
        self._parent = None
        self._verify_t0 = 0.0
        self.last_blobs: list[Blob] = []
        self.voice_hint = None           # (x, y, world bearing, t) of an unexplained shout
        self.ic_mode: str | None = None  # None | 'STOP' | 'RECALL'
        self.ic_area = None              # (x0, y0, x1, y1) IC-ordered search area
        self.speed = 0.0
        self.speed_why = ''
        self.guard_stops = 0   # ticks where the proximity guard cut motion (shown in evaluation)
        self.rescue_order: list[Track] = []
        self._last_hear = -99.0
        self.say_out: list[dict] = []   # speaker instructions for the world (sim) to deliver
        ys, xs = np.mgrid[0:self.grid.rows, 0:self.grid.cols]
        self._cx, self._cy = (xs + 0.5) * RES, (ys + 0.5) * RES

    # ------------------------------------------------------------ helpers
    def emit(self, kind, trigger, detail, change, rule):
        c = Card(self.t, kind, trigger, detail, change, rule)
        self.cards.append(c)
        if not self._comms:
            self.outbox.append(c)
        return c

    def cell_name(self, c):
        x, y = self.grid.to_world(*c)
        return f'({x:.1f},{y:.1f})'

    def track(self, tid):
        return next((t for t in self.tracker.tracks if t.id == tid), None)

    def feedback(self, track_id: int, is_person: bool) -> None:
        tr = self.track(track_id)
        if tr is None:
            return
        before = self.cal.conf(tr)
        self.cal.update(tr, is_person)
        tr.ic = is_person
        tr.conf = self.cal.conf(tr)
        if not is_person:
            tr.state = 'rejected'
        self.emit('LEARN', 'IC', f'track#{tr.id} marked {"PERSON" if is_person else "NOT PERSON"}',
                  f'similar conf {before:.2f}->{tr.conf:.2f}', 'ic-feedback')

    def ic(self, cmd: str, area=None) -> None:
        """Incident Commander override. The safety gate still wins over every IC order."""
        cmd = cmd.upper()
        if cmd == 'STOP':
            self.ic_mode = 'STOP'
            self.emit('IC', 'IC', 'STOP ordered', 'holding position, sensors + reporting stay on', 'ic-override')
        elif cmd == 'RESUME':
            self.ic_mode, self.ic_area = None, None
            self.emit('IC', 'IC', 'RESUME ordered', 'back to autonomous search', 'ic-override')
        elif cmd == 'RECALL':
            self.ic_mode = 'RECALL'
            self.emit('IC', 'IC', 'RECALL ordered', 'returning to entry point', 'ic-override')
        elif cmd == 'SEARCH':
            self.ic_mode, self.ic_area = None, tuple(area)
            self.emit('IC', 'IC', f'search area x{area[0]:.0f}-{area[2]:.0f} y{area[1]:.0f}-{area[3]:.0f} first',
                      'exploration restricted to that area', 'ic-priority-area')

    def _around(self, layer, xy, r, fn):
        m = (self._cx - xy[0]) ** 2 + (self._cy - xy[1]) ** 2 <= r * r
        return float(fn(layer[m])) if m.any() else float(fn(layer))

    def priority(self, tr: Track) -> tuple[str, float, str]:
        """Rescue priority for the crew (NOT clinical triage - extrication and triage stay medically led).
        P1 = breathing and (unresponsive or in a hazardous spot); P2 = breathing, calling, spot survivable;
        P? = warm body-shaped, breathing not yet confirmed: crew must check."""
        g = self.grid
        fire_d = min((math.dist(tr.xy, f[:2]) for f in self.fires), default=99.0)
        heat = self._around(g.heat, tr.xy, 1.0, np.max)
        co = self._around(g.gas, tr.xy, 1.5, np.max)
        o2 = self._around(g.o2, tr.xy, 1.5, np.min)
        why = []
        if fire_d < 3.0:
            why.append(f'{fire_d:.1f}m from fire')
        if heat > 60:
            why.append(f'{heat:.0f}C nearby')
        if co > 400:
            why.append(f'CO {co:.0f}ppm')
        if o2 < 17.0:
            why.append(f'O2 {o2:.1f}%')
        hazard = bool(why)
        why.append('calling out' if tr.responsive else 'no voice heard')
        if tr.instruct == 'EVACUATE':
            return 'EVAC', 5.0, 'told to walk out: expect at entry, ' + ', '.join(why)
        if tr.state != 'pinned':
            label, base = 'P?', 50.0
        elif not tr.responsive or hazard:
            label, base = 'P1', 100.0
        else:
            label, base = 'P2', 30.0
        score = (base + (0 if tr.responsive else 15) + max(0.0, 3.0 - fire_d) * 8 + co / 100
                 + max(0.0, 17.0 - o2) * 5 + max(0.0, heat - 40) / 5)
        return label, score, ', '.join(why)

    def semantic(self) -> np.ndarray:
        """0 unknown/free, 1 wall, 2 furniture, 3 person, 4 fire. From the depth camera: an object whose TOP
        edge was seen below 1.4 m is furniture; tall surfaces or radar hits with no top seen are walls."""
        g = self.grid
        sem = np.zeros(g.occ.shape, np.int8)
        sem[(g.occ == 1) | (g.maxz >= WALL_Z)] = 1
        sem[(g.topz > FURN_MIN_Z) & (g.topz < WALL_Z) & (g.maxz < WALL_Z)] = 2
        sem[g.heat > NO_GO_C] = 4
        for tr in self.tracker.tracks:
            if tr.kind == 'warm' and tr.state != 'rejected' and tr.conf >= VERIFY_CONF:
                m = (self._cx - tr.xy[0]) ** 2 + (self._cy - tr.xy[1]) ** 2 <= 0.5 ** 2
                sem[m] = 3
        return sem

    def _hear(self, obs) -> None:
        x, y, yaw = obs.pose
        for rel, db in obs.voices:
            wb = _wrap(yaw + rel)
            match = [tr for tr in self.tracker.tracks if tr.kind == 'warm' and tr.state != 'rejected'
                     and abs(_wrap(math.atan2(tr.xy[1] - y, tr.xy[0] - x) - wb))
                     < (2 * VOICE_MATCH_RAD if tr.responsive else VOICE_MATCH_RAD)]
            if match:
                for tr in match:
                    if not tr.responsive:
                        tr.responsive = True
                        self.emit('HEAR', 'mic', f'voice {db:.0f}dB matches track#{tr.id}', 'marked responsive',
                                  'voice+thermal')
                continue
            new = self.voice_hint is None or abs(_wrap(self.voice_hint[2] - wb)) > 0.8
            self.voice_hint = (x, y, wb, self.t)
            if (new and self.t - self._last_hear >= 3.0) or self.t - self._last_hear >= 12.0:
                self._last_hear = self.t
                self.emit('HEAR', 'mic', f'voice {db:.0f}dB bearing {math.degrees(wb):.0f}deg, no body seen there',
                          'searching toward the sound', 'voice-prior')

    # ------------------------------------------------------------ main loop
    def step(self, obs) -> Command:
        t0 = time.perf_counter()
        self.t = obs.t
        self._ticks += 1
        g = self.grid
        x, y, yaw = obs.pose
        here = g.to_cell(x, y)

        # ---- comms bookkeeping
        if obs.comms:
            if not self._comms:
                self.outbox.clear()  # back in range: queued findings uploaded
            self._comms = True
            self.last_comms_cell = here
        elif self._comms:
            self._comms = False
            self.emit('COMMS', 'radio', 'link lost', f'continuing {self.mode}, logging offline',
                      'comms-loss-continue')

        # ---- map
        integrate_ranges(g, obs.pose, obs.ranges)
        g.occ[here] = 0
        blobs = detect_blobs(obs.thermal, obs.depth, obs.pose)
        self.last_blobs = blobs
        for b in blobs:
            if b.kind == 'fire':
                self._paint_heat(b.xy, b.temp_c)
        near = (self._cx - x) ** 2 + (self._cy - y) ** 2 <= 0.5 ** 2
        g.gas[near] = np.maximum(g.gas[near], obs.gas_ppm)
        g.o2[near] = np.minimum(g.o2[near], obs.o2_pct)
        # measured heatmap + occlusion-aware coverage from the thermal pixels themselves
        px, py, pz, (pv, pu), pd = project_pixels(obs.depth, obs.pose)
        inside = (px >= 0) & (py >= 0) & (px < g.width) & (py < g.height)
        ci, cj = (py[inside] / RES).astype(int), (px[inside] / RES).astype(int)
        temps = obs.thermal[pv[inside], pu[inside]]
        low = pz[inside] < 1.6
        np.maximum.at(g.heat, (ci[low], cj[low]), temps[low].astype(np.float32))
        seen = (pz[inside] < 0.6) & (pd[inside] <= VIEW_RANGE)
        g.scanned[ci[seen], cj[seen]] = True
        zz = pz[inside].astype(np.float32)
        np.maximum.at(g.maxz, (ci, cj), zz)
        above = obs.depth[np.maximum(pv[inside] - 4, 0), pu[inside]]
        edge = ((above > pd[inside] + 0.5) | ~np.isfinite(above)) & (pv[inside] >= 4) & (temps < WARM_MASK_C)
        np.maximum.at(g.topz, (ci[edge], cj[edge]), zz[edge])
        t1 = time.perf_counter()

        # ---- perception -> tracks
        self.tracker.update([b for b in blobs if b.kind == 'warm'], obs.t, obs.pose)
        self._hear(obs)
        vt = self.track(self.verify_id) if self.verify_id else None
        if vt is not None and obs.breath is not None:
            vt.breath_n += 1
            vt.breath = (vt.breath or 0.0) + (obs.breath - (vt.breath or 0.0)) / vt.breath_n
        for tr in self.tracker.tracks:
            if tr.kind != 'warm' or tr.state in ('pinned', 'rejected') or tr.ic is not None:
                continue
            tr.conf = self.cal.conf(tr)
            if len(tr.hits) >= 5 and tr.conf < REJECT_CONF:
                tr.state = 'rejected'
                if self.verify_id == tr.id:
                    self.verify_id, self.mode = None, 'EXPLORE'
                self.emit('REJECT', 'thermal', f'track#{tr.id} {np.median(tr.temps):.0f}°C '
                          f'{tr.width_m:.1f}x{tr.height_m:.1f}m conf {tr.conf:.2f}', 'ignored', 'not-body-like')
            elif tr.breath is not None and tr.breath_n >= BREATH_SAMPLES:
                if self.verify_id == tr.id or tr.state != 'verify':
                    if tr.conf >= PIN_CONF and tr.breath >= BREATH_OK:
                        tr.state = 'pinned'
                        self.emit('PIN', 'thermal+radar', f'casualty#{tr.id} at ({tr.xy[0]:.1f},{tr.xy[1]:.1f}) '
                                  f'conf {tr.conf:.2f} breathing {tr.breath:.2f}', 'added to rescue route',
                                  'two-sensor-confirm')
                    elif tr.verify_tries < 2 and tr.conf >= VERIFY_CONF:  # second opinion from another angle
                        tr.verify_tries += 1
                        tr.verify_from = (obs.pose[0], obs.pose[1])
                        self.emit('POSSIBLE', 'radar', f'track#{tr.id} conf {tr.conf:.2f} breathing {tr.breath:.2f}',
                                  're-checking from another angle', 'never-drop-candidate')
                        tr.state, tr.breath, tr.breath_n = 'candidate', None, 0
                        self._verify_fail[tr.id] = self.t - 8.0
                    else:
                        tr.state = 'verify'
                        self.emit('POSSIBLE', 'radar', f'track#{tr.id} conf {tr.conf:.2f} breathing {tr.breath:.2f}',
                                  'kept as POSSIBLE casualty for crew', 'never-drop-candidate')
                    if self.verify_id == tr.id:
                        self.verify_id = None
                        self.mode = 'EXPLORE'
            elif len(tr.hits) >= 3 and tr.conf >= VERIFY_CONF and self.verify_id is None \
                    and self.mode != 'RETREAT' and tr.state == 'candidate'                     and self.t - self._verify_fail.get(tr.id, -99.0) >= 10.0:
                tr.state = 'verify'
                self.verify_id = tr.id
                self._verify_t0 = self.t
                self.mode = 'VERIFY'
                self.emit('VERIFY', 'thermal', f'track#{tr.id} {np.median(tr.temps):.0f}°C conf {tr.conf:.2f}',
                          'second look + radar breath check', 'uncertain->verify')
        t2 = time.perf_counter()

        # ---- safety gate + planning
        cost = cost_map(g)
        free_escape(cost, g, here)
        blocked = any(not np.isfinite(cost[c]) for c in self.path)
        if self._dist is None or blocked or self._ticks % 5 == 1:  # 4 Hz global plan, instant on block
            self._dist, self._parent = dijkstra(cost, here)
        dist, parent = self._dist, self._parent
        cmd = self._guard(self._plan(obs, cost, dist, parent, here), obs)
        if self._ticks % 10 == 1 and dist is not None:
            self.exit_path = unwind(parent, self.start_cell) if np.isfinite(dist[self.start_cell]) else None
            self._update_rescue(cost, dist, here)
        t3 = time.perf_counter()
        self.timings = {'map': (t1 - t0) * 1e3, 'perceive': (t2 - t1) * 1e3, 'plan': (t3 - t2) * 1e3,
                        'total': (t3 - t0) * 1e3}
        return cmd

    def tracker_xy(self, cell):
        return self.grid.to_world(*cell)

    def _paint_heat(self, xy, temp):
        m = (self._cx - xy[0]) ** 2 + (self._cy - xy[1]) ** 2 <= 0.35 ** 2
        self.grid.heat[m] = np.maximum(self.grid.heat[m], temp)
        for k, (fx, fy, ft) in enumerate(self.fires):
            if math.dist((fx, fy), xy) < 1.2:
                self.fires[k] = (fx, fy, max(ft, temp))
                return
        self.fires.append((xy[0], xy[1], temp))

    def _goto(self, goal, parent, cost):
        """Set self.path toward goal; emit REPLAN (with the sensor reason) if the old route got blocked."""
        old = self.path
        new = unwind(parent, goal)
        if old and self.goal == goal and len(old) > 2:
            blocked = next((c for c in old if not np.isfinite(cost[c])), None)
            if blocked is not None:
                g = self.grid
                hot = float(np.max(g.heat[max(0, blocked[0] - 2):blocked[0] + 3,
                                          max(0, blocked[1] - 2):blocked[1] + 3]))
                if hot > NO_GO_C:
                    trig, det, rule = 'thermal', f'{self.cell_name(blocked)}={hot:.0f}°C>{NO_GO_C:.0f}', 'heat-no-go'
                else:
                    trig, det, rule = 'radar', f'obstacle at {self.cell_name(blocked)}', 'obstacle'
                big = trig == 'thermal' or abs(len(new) - len(old)) * RES >= 1.0
                if big and self.t - self._last_replan.get(trig, -99.0) >= 2.0:
                    self._last_replan[trig] = self.t
                    self.emit('REPLAN', trig, det, f'old {len(old) * RES:.1f}m->new {len(new) * RES:.1f}m '
                              f'({(len(new) - len(old)) * RES / SPEED:+.0f}s)', rule)
        self.goal, self.path = goal, new

    def _update_rescue(self, cost, dist, here) -> None:
        """Rescue route visits confirmed casualties in priority order (P1 first)."""
        pinned = [t for t in self.tracker.tracks if t.state == 'pinned']
        for tr in pinned:
            label, score, why = self.priority(tr)
            if label != tr.priority:
                tr.priority = label
                self.emit('PRIORITY', 'thermal+radar+mic+gas', f'casualty#{tr.id} {label}: {why}',
                          'rescue order updated', 'rescue-priority')
        self._instruct(cost, here)
        ranked = sorted([t for t in pinned if t.instruct != 'EVACUATE'], key=lambda t: -self.priority(t)[1])[:4]
        visit, cur, path = [], here, [here]
        for tr in ranked:
            tgt = self._near(tr.xy, dist) or self.grid.to_cell(*tr.xy)
            if not np.isfinite(dist[tgt]):
                continue
            leg = astar(cost, cur, tgt)
            if leg is None:
                continue
            path += leg[1:]
            visit.append(tr)
            cur = tgt
        self.rescue_order = visit
        self.rescue = (list(range(len(visit))), path) if visit else None

    def _instruct(self, cost, here) -> None:
        """Talk to conscious people in speaker range: walk out if the measured route is survivable, else stay put."""
        g = self.grid
        rxy = g.to_world(*here)
        for tr in self.tracker.tracks:
            if (tr.kind != 'warm' or not tr.responsive or tr.instruct or tr.state == 'rejected'
                    or tr.conf < VERIFY_CONF or math.dist(rxy, tr.xy) > SPEAKER_RANGE):
                continue
            route = astar(cost, g.to_cell(*tr.xy), self.start_cell)
            hot = max((float(g.heat[c]) for c in route), default=999.0) if route else 999.0
            near_fire = min((math.dist(g.to_world(*c), f[:2]) for c in (route or []) for f in self.fires), default=99.0)
            safe = route is not None and hot <= EVAC_MAX_C and near_fire >= EVAC_FIRE_CLEARANCE
            tr.instruct = 'EVACUATE' if safe else 'SHELTER'
            detail = (f'person#{tr.id} conscious; way out {len(route) * RES:.1f}m, max {hot:.0f}C, '
                      f'{near_fire:.1f}m from fire') if route else f'person#{tr.id} conscious; no known way out'
            self.emit('SAY', 'mic+thermal', detail,
                      'told to walk out' if safe else 'told to stay put + stay low; crew priority',
                      'evacuate-if-route-safe' if safe else 'shelter-if-route-unsafe')
            self.say_out.append({'track': tr.id, 'kind': tr.instruct, 'text': SAY_TEXT[tr.instruct], 'xy': tr.xy,
                                 'route': [g.to_world(*c) for c in route] if safe else None})

    def _plan(self, obs, cost, dist, parent, here) -> Command:
        g = self.grid
        x, y, yaw = obs.pose
        heat_here = float(g.heat[here])
        hot_here = max(heat_here, float(cv2.GaussianBlur(g.heat, (0, 0), 0.5 / RES)[here])) > NO_GO_C

        # 1. self-protection (safety gate, fixed rules)
        if obs.self_temp_c >= SELF_TEMP_LIMIT or hot_here or obs.battery < 0.15:
            if self.mode != 'RETREAT':
                rule = 'low-battery' if obs.battery < 0.15 else 'heat-limit'
                self.emit('RETREAT', 'self-temp' if obs.self_temp_c >= SELF_TEMP_LIMIT else 'thermal',
                          f'self {obs.self_temp_c:.0f}°C cell {heat_here:.0f}°C battery {obs.battery:.0%}',
                          'retreating to coolest known area', rule)
                self.mode = 'RETREAT'
                self.verify_id = None
            target = self.start_cell if obs.battery < 0.15 else self._coolest(here)
            return self._drive_to(target, obs, cost, dist, parent)
        if self.mode == 'RETREAT':
            if obs.self_temp_c < SELF_TEMP_LIMIT - 10:
                self.mode = 'EXPLORE'
            else:
                return self._drive_to(self._coolest(here), obs, cost, dist, parent)

        # 2. Incident Commander orders (below safety, above autonomy)
        if self.ic_mode == 'STOP':
            self.mode, self.path = 'IC_STOP', []
            return Command(0.0, 0.0, 0.0)
        if self.ic_mode == 'RECALL':
            self.mode = 'RECALL'
            if math.dist((x, y), g.to_world(*self.start_cell)) < 0.4:
                self.path = []
                return Command(0.0, 0.0, 0.0)
            return self._drive_to(self.start_cell, obs, cost, dist, parent)

        # 3. verify uncertain casualty (second look + radar breath check)
        vt = self.track(self.verify_id) if self.verify_id else None
        if vt is not None and vt.state in ('pinned', 'rejected'):
            self.verify_id, vt = None, None
        if vt is not None and self.t - self._verify_t0 > VERIFY_TIMEOUT and vt.breath_n < BREATH_SAMPLES:
            self.emit('POSSIBLE', 'thermal', f'track#{vt.id} conf {vt.conf:.2f}, no radar line of sight '
                      f'in {VERIFY_TIMEOUT:.0f}s', 'kept as POSSIBLE casualty, exploring on', 'verify-timeout')
            vt.state = 'candidate'
            self._verify_fail[vt.id] = self.t
            self.verify_id, vt = None, None
        if vt is not None:
            self.mode = 'VERIFY'
            if math.dist((x, y), vt.xy) <= BREATH_RANGE and self.grid_los(here, g.to_cell(*vt.xy)):
                face = math.atan2(vt.xy[1] - y, vt.xy[0] - x)
                return Command(0.0, 0.0, float(np.clip(2.0 * _wrap(face - yaw), -1.5, 1.5)), vt.xy)
            target = self._near(vt.xy, dist, avoid=vt.verify_from)
            if target is not None:
                return self._drive_to(target, obs, cost, dist, parent)
            vt.state = 'candidate'  # unreachable for now: retry later, never dropped
            self._verify_fail[vt.id] = self.t
            self.verify_id = None
            self.mode = 'EXPLORE'

        # 3. explore nearest reachable frontier (commit to goal while it stays a frontier)
        if dist is not None:
            # goals: radar frontiers + free cells the thermal camera has not looked at yet
            unk = g.occ == -1
            fmask = np.zeros_like(unk)
            fmask[1:, :] |= unk[:-1, :]
            fmask[:-1, :] |= unk[1:, :]
            fmask[:, 1:] |= unk[:, :-1]
            fmask[:, :-1] |= unk[:, 1:]
            goal_mask = (g.occ == 0) & (fmask | ~g.scanned) & (cost < 6.0) & np.isfinite(dist)                 & (np.hypot(self._cx - x, self._cy - y) > 0.6)
            if self.ic_area is not None:
                ax0, ay0, ax1, ay1 = self.ic_area
                in_area = (self._cx >= ax0) & (self._cx <= ax1) & (self._cy >= ay0) & (self._cy <= ay1)
                if (goal_mask & in_area).any():
                    goal_mask &= in_area
                else:
                    self.emit('IC', 'map', 'IC priority area searched', 'resuming autonomous search',
                              'ic-priority-area')
                    self.ic_area = None
            hint = self.voice_hint if self.voice_hint and self.t - self.voice_hint[3] < VOICE_MEMORY_S else None
            if hint is not None and goal_mask.any():
                ang = np.arctan2(self._cy - hint[1], self._cx - hint[0])
                score = np.where(goal_mask, dist + 8.0 * (1 - np.cos(ang - hint[2])), np.inf)
                target = tuple(int(v) for v in np.unravel_index(np.argmin(score), dist.shape))
            elif self.goal is not None and goal_mask[self.goal]:
                target = self.goal
            elif goal_mask.any():
                target = tuple(int(v) for v in np.unravel_index(np.argmin(np.where(goal_mask, dist, np.inf)),
                                                                 dist.shape))
            else:
                target = None
            if target is not None:
                self._no_target = 0
                self.mode = 'EXPLORE'
                return self._drive_to(target, obs, cost, dist, parent)
        self._no_target += 1
        if self._no_target < 20 and self.mode != 'HOLD':  # 1 s grace: frontiers flicker with sensor noise
            return Command(0.0, 0.0, 0.6)  # turn in place to look around
        if not obs.comms and here != self.last_comms_cell and dist is not None \
                and np.isfinite(dist[self.last_comms_cell]):
            if self.mode != 'RETURN_COMMS':
                self.emit('COMMS', 'radio', 'exploration done offline', 'returning to last radio contact',
                          'upload-findings')
            self.mode = 'RETURN_COMMS'
            return self._drive_to(self.last_comms_cell, obs, cost, dist, parent)
        if self.mode != 'HOLD':
            pct = 100.0 * (g.scanned & (g.occ == 0)).sum() / max(1, (g.occ == 0).sum())
            self.emit('HOLD', 'map', f'no reachable unexplored area (scanned {pct:.0f}%)',
                      'holding position, human search still required', 'never-declare-empty')
            self.mode = 'HOLD'
        self.path = []
        return Command(0.0, 0.0, 0.0)

    def _coolest(self, here):
        g = self.grid
        d = np.hypot(self._cx - self._cx[here], self._cy - self._cy[here])
        heat = np.maximum(g.heat, cv2.GaussianBlur(g.heat, (0, 0), 0.5 / RES))
        cand = np.where((g.occ == 0) & (d <= 6.0), heat + 0.5 * d, np.inf)
        return tuple(int(v) for v in np.unravel_index(np.argmin(cand), cand.shape))

    def grid_los(self, a, b) -> bool:
        """No known wall cell between a and b on the robot's own map."""
        return all(self.grid.occ[c] != 1 for c in _line(a, b)[1:-1] if self.grid.inside(*c))

    def _near(self, xy, dist, avoid=None):
        """Reachable free cell 1-2 m from xy with line of sight to it (for a second look), >1.2 m from `avoid`."""
        if dist is None:
            return None
        d = np.hypot(self._cx - xy[0], self._cy - xy[1])
        ok = (d >= 1.0) & (d <= 2.0) & np.isfinite(dist)
        if avoid is not None:
            ok &= np.hypot(self._cx - avoid[0], self._cy - avoid[1]) > 1.2
        cand = np.where(ok, dist, np.inf)
        tgt = self.grid.to_cell(*xy)
        order = np.argsort(cand, axis=None)[:80]
        for flat in order:
            c = tuple(int(v) for v in np.unravel_index(flat, cand.shape))
            if not np.isfinite(cand[c]):
                return None
            if self.grid_los(c, tgt):
                return c
        return None

    def _guard(self, cmd: Command, obs) -> Command:
        """Reactive proximity stop (radar ring, every tick): never drive into anything closer than 0.45 m from the
        robot centre (hull half-length 0.23 m); push gently away inside 0.32 m. Overrides any plan or IC order."""
        v = np.array([cmd.vx, cmd.vy], float)
        ang = obs.pose[2] + np.radians(np.arange(len(obs.ranges)) * 360.0 / len(obs.ranges))
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        hit = False
        for d, r in zip(dirs, obs.ranges):
            if r >= 0.45:
                continue
            into = float(v @ d)
            if into > 0:
                v -= into * d
                hit = True
            if r < 0.32:
                v -= 0.12 * d
        if hit:
            self.guard_stops += 1
        return Command(float(v[0]), float(v[1]), cmd.wz, cmd.want_breath_xy)

    def _pick_speed(self, x, y) -> tuple[float, str]:
        """Fast where the next 3 m are mapped, scanned and cool; slow near heat or a casualty; normal otherwise."""
        g = self.grid
        ahead = self.path[:12]
        heat = max((float(g.heat[c]) for c in ahead), default=AMBIENT_C)
        near_person = any(t.kind == 'warm' and t.state in ('pinned', 'verify') and math.dist(t.xy, (x, y)) < 1.5
                          for t in self.tracker.tracks)
        if heat > 80 or near_person:
            return SLOW_SPEED, 'slow: heat ahead' if heat > 80 else 'slow: casualty close'
        warm_near = any(t.kind == 'warm' and t.state == 'candidate' and math.dist(t.xy, (x, y)) < 2.5
                        for t in self.tracker.tracks)
        if ahead and heat < 40 and all(g.scanned[c] for c in ahead) and not warm_near:
            return FAST_SPEED, 'fast: known cool route'
        return SPEED, 'normal: searching'

    def _drive_to(self, target, obs, cost, dist, parent) -> Command:
        x, y, yaw = obs.pose
        if dist is None or target is None or not np.isfinite(dist[target]):
            # standing in a no-go cell: back away down the heat gradient
            gy, gx = np.gradient(cv2.GaussianBlur(self.grid.heat, (0, 0), 2))
            i, j = self.grid.to_cell(x, y)
            v = -np.array([gx[i, j], gy[i, j]])
            n = np.linalg.norm(v)
            v = v / n * SPEED if n > 1e-6 else np.array([-SPEED, 0.0])
            return Command(float(v[0]), float(v[1]), 0.0)
        self._goto(target, parent, cost)
        pts = [self.grid.to_world(*c) for c in self.path]
        k0 = min(range(len(pts)), key=lambda k: math.dist(pts[k], (x, y)))
        look = next((p for p in pts[k0:] if math.dist(p, (x, y)) >= 0.5), pts[-1])
        dx, dy = look[0] - x, look[1] - y
        dn = math.hypot(dx, dy)
        if dn < 1e-3:
            return Command(0.0, 0.0, 0.0)
        speed, self.speed_why = self._pick_speed(x, y)
        turn = _wrap(math.atan2(dy, dx) - yaw)
        k = max(0.0, math.cos(turn))  # slow while turning so the thermal camera looks where it goes
        self.speed = speed * k
        return Command(dx / dn * speed * k, dy / dn * speed * k, float(np.clip(2.5 * turn, -1.5, 1.5)))
