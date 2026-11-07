"""Headless evaluation: FireScout vs a wall-following firefighter search on unseen random floors.

Ground truth (Layout) is used ONLY here for scoring; the brain never sees it.
Usage: python eval.py --n 30 --workers 8
"""
from __future__ import annotations

import argparse
import math
import time
from multiprocessing import Pool

import mujoco
import numpy as np

import brain as B
import scene
import sensors

CREW_DELAY_S = 150.0   # assumption: BA don + entry-control briefing before a crew enters (stated in results)
CRAWL_MPS = 0.3        # assumption: crawling wall-following search speed in zero visibility
REACH_M = 1.5          # assumption: arm + tool sweep reach from the wall line
FOUND_M = 1.2          # a pin within this distance of a casualty centre counts as correct
DT = 0.05


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def true_grid(L: scene.Layout) -> np.ndarray:
    g = B.Grid(L.width, L.height)
    occ = np.zeros((g.rows, g.cols), np.int8)
    for x0, y0, x1, y1 in L.walls:
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / (B.RES / 4)) + 1
        for t in np.linspace(0, 1, n):
            occ[g.to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))] = 1
    return occ


def baseline_time(L: scene.Layout, crew_delay_s: float = CREW_DELAY_S, speed_mps: float = CRAWL_MPS):
    """Right-hand wall-following search on the true map; time until within 1 m of the first casualty."""
    cas = [t.xy for t in L.things if t.kind == 'casualty']
    if not cas:
        return None
    occ = true_grid(L)
    rows, cols = occ.shape
    free = lambda c: 0 <= c[0] < rows and 0 <= c[1] < cols and occ[c] == 0
    dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # E, S, W, N  (clockwise -> index+1 is a right turn)
    pos = (int(L.start[1] / B.RES), int(L.start[0] / B.RES))
    h = 0
    steps = 0
    seen = set()
    touching = False  # crawl straight until a wall is found, then keep it on the right hand
    for steps in range(1, 20000):
        if not touching:
            nxt = (pos[0] + dirs[h][0], pos[1] + dirs[h][1])
            if free(nxt):
                pos = nxt
                xy = ((pos[1] + 0.5) * B.RES, (pos[0] + 0.5) * B.RES)
                if any(math.dist(xy, c) <= REACH_M for c in cas):
                    return crew_delay_s + steps * B.RES / speed_mps
                continue
            touching, h = True, (h - 1) % 4  # wall ahead: turn left so it is on the right
        if (pos, h) in seen:  # searched every wall reachable from the entry
            return None
        seen.add((pos, h))
        for turn in (1, 0, -1, 2):  # right, straight, left, back
            nh = (h + turn) % 4
            nxt = (pos[0] + dirs[nh][0], pos[1] + dirs[nh][1])
            if free(nxt):
                pos, h = nxt, nh
                break
        xy = ((pos[1] + 0.5) * B.RES, (pos[0] + 0.5) * B.RES)
        if any(math.dist(xy, c) <= REACH_M for c in cas):
            return crew_delay_s + steps * B.RES / speed_mps
    return None


def run_episode(seed: int, t_max: float = 180.0) -> dict:
    L = scene.generate_layout(seed)
    m, d = scene.load(L)
    s = sensors.Sensors(m, d, L, seed=seed, render_rgb=False)
    b = B.Brain(L.width, L.height, L.start, feedback_path='nul')
    cas = [t.xy for t in L.things if t.kind == 'casualty']
    cmd = B.Command(0, 0, 0)
    steps = int(round(DT / m.opt.timestep))
    lat, breaches, t_first = [], 0, None
    while d.time < t_max:
        o = s.observe(cmd.want_breath_xy)
        cmd = b.step(o)
        lat.append(b.timings['total'])
        rxy = s._robot_xy()
        if any(math.dist(rxy, f) < 0.8 for f, _ in s._fires()):
            breaches += 1
        if t_first is None and any(tr.state == 'pinned' and any(math.dist(tr.xy, c) <= FOUND_M for c in cas)
                                   for tr in b.tracker.tracks):
            t_first = round(d.time, 1)
        d.ctrl[:] = [cmd.vx, cmd.vy, cmd.wz]
        for _ in range(steps):
            mujoco.mj_step(m, d)
    pins = [tr for tr in b.tracker.tracks if tr.state == 'pinned']
    found = sum(any(math.dist(tr.xy, c) <= FOUND_M for tr in pins) for c in cas)
    false_pins = sum(not any(math.dist(tr.xy, c) <= FOUND_M for c in cas) for tr in pins)
    possible = sum(tr.kind == 'warm' and tr.state in ('verify', 'candidate') and tr.conf >= 0.4
                   for tr in b.tracker.tracks)
    return dict(seed=seed, t_first_casualty=t_first, found=found, total_casualties=len(cas),
                false_pins=false_pins, possible=possible, heat_breaches=breaches,
                scanned_pct=round(100 * (b.grid.scanned & (b.grid.occ == 0)).sum() / max(1, (b.grid.occ == 0).sum()), 1),
                loop_ms_p50=round(float(np.percentile(lat, 50)), 1), loop_ms_p95=round(float(np.percentile(lat, 95)), 1),
                baseline_t=(lambda bt: None if bt is None else round(bt, 1))(baseline_time(L)),
                never_clear=not any('clear' in str(c).lower() for c in b.cards))


def summarize(rows: list[dict]) -> str:
    with_cas = [r for r in rows if r['total_casualties'] > 0]
    tf = [r['t_first_casualty'] for r in with_cas if r['t_first_casualty'] is not None]
    tb = [r['baseline_t'] for r in with_cas if r['baseline_t'] is not None]
    found = sum(r['found'] for r in rows)
    total = sum(r['total_casualties'] for r in rows)
    pins = found + sum(r['false_pins'] for r in rows)
    rlo, rhi = wilson(found, total)
    plo, phi = wilson(found, pins)
    med = lambda v: f'{np.median(v):.0f} s (IQR {np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f})' if v else 'n/a'
    lines = [
        '# FireScout evaluation (simulation, unseen procedurally generated floors)', '',
        f'Episodes: {len(rows)} floors · {total} casualties · 180 s robot mission each.',
        f'Baseline assumptions: crew enters after **{CREW_DELAY_S:.0f} s** (BA don + entry control), '
        f'right-hand wall-following crawl at **{CRAWL_MPS} m/s** with a {REACH_M} m sweep reach '
        '(zero visibility, no thermal camera; a crew with a TIC would be faster).',
        'Ground truth is used only for scoring; the robot never sees it. These are simulated results, '
        'not real-world performance.', '',
        '| Metric | Result |', '|---|---|',
        f'| Time to first confirmed casualty, FireScout | {med(tf)} ({len(tf)}/{len(with_cas)} floors) |',
        f'| Time to first casualty, wall-following crew baseline | {med(tb)} |',
        f'| Casualty recall (confirmed pins) | {found}/{total} = {found / max(1, total):.0%} (95% CI {rlo:.0%}-{rhi:.0%}) |',
        f'| Pin precision | {found}/{pins} = {found / max(1, pins):.0%} (95% CI {plo:.0%}-{phi:.0%}) |',
        f'| Unconfirmed "POSSIBLE" casualties flagged for crew | {sum(r["possible"] for r in rows)} |',
        f'| Heat-limit breaches (robot within 0.8 m of a fire) | {sum(r["heat_breaches"] for r in rows)} ticks |',
        f'| Any card/report claiming an area is clear | {"0 (by design)" if all(r["never_clear"] for r in rows) else "VIOLATION"} |',
        f'| Thermal scan coverage (median) | {np.median([r["scanned_pct"] for r in rows]):.0f}% of mapped free space |',
        f'| Brain loop latency p50 / p95 (laptop CPU) | {np.median([r["loop_ms_p50"] for r in rows]):.1f} / '
        f'{np.median([r["loop_ms_p95"] for r in rows]):.1f} ms |', '',
        '## Per floor', '',
        '| seed | casualties | found | FireScout t (s) | baseline t (s) | false pins | possible | breaches | scanned % |',
        '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['total_casualties']} | {r['found']} | {r['t_first_casualty']} | "
                     f"{r['baseline_t']} | {r['false_pins']} | {r['possible']} | {r['heat_breaches']} | {r['scanned_pct']} |")
    return '\n'.join(lines) + '\n'


def main(n: int = 30, workers: int = 8, t_max: float = 180.0, out: str = 'results.md'):
    t0 = time.time()
    with Pool(workers) as p:
        rows = p.starmap(run_episode, [(s, t_max) for s in range(n)])
    text = summarize(sorted(rows, key=lambda r: r['seed']))
    open(out, 'w', encoding='utf-8').write(text)
    print(text)
    print(f'wall time {time.time() - t0:.0f} s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--t-max', type=float, default=180.0)
    a = ap.parse_args()
    main(a.n, a.workers, a.t_max)
