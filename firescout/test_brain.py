import math

import numpy as np

from brain import Grid, astar, cost_map, frontiers, integrate_ranges, rescue_route


# ---------------- Task 3: map + planners

def open_grid(w=4, h=4):
    g = Grid(w, h)
    g.occ[:] = 0
    return g


def test_astar_avoids_no_go():
    g = open_grid()
    g.heat[6:10, 0:12] = 600  # hot band with a gap on the right (x 3..4 m)
    c = cost_map(g)
    p = astar(c, g.to_cell(0.5, 0.5), g.to_cell(0.5, 3.5))
    assert p is not None and all(np.isfinite(c[i, j]) for i, j in p)
    assert max(j for _, j in p) >= 12  # went through the gap


def test_astar_none_when_fully_blocked():
    g = open_grid()
    g.heat[6:10, :] = 600
    assert astar(cost_map(g), g.to_cell(0.5, 0.5), g.to_cell(0.5, 3.5)) is None


def test_fire_cell_itself_is_no_go():
    g = open_grid()
    g.heat[8, 8] = 600  # a single small fire blob
    assert not np.isfinite(cost_map(g)[8, 8])


def test_integrate_ranges_marks_wall():
    g = Grid(6, 6)
    r = np.full(36, 6.0)
    r[0] = 2.0  # ray 0 points along robot heading (+x)
    integrate_ranges(g, (1.0, 3.0, 0.0), r)
    assert g.occ[g.to_cell(3.0, 3.0)] == 1
    assert g.occ[g.to_cell(2.0, 3.0)] == 0


def test_frontiers_found_at_known_unknown_border():
    g = Grid(4, 4)
    g.occ[0:4, 0:4] = 0
    f = frontiers(g)
    assert f and all(i <= 3 and j <= 3 for i, j in f)


def test_rescue_route_orders_targets():
    g = open_grid(8, 4)
    c = cost_map(g)
    start = g.to_cell(0.5, 2.0)
    near, far = g.to_cell(3.0, 2.0), g.to_cell(7.0, 2.0)
    order, path = rescue_route(c, start, [far, near])
    assert order == [1, 0] and path[0] == start and path[-1] == far


# ---------------- Task 4: perception
from collections import deque

from brain import Blob, Calibrator, Track, Tracker, detect_blobs


def test_detect_fire_and_warm_blob_projection():
    th = np.full((120, 160), 20, np.float32)
    th[50:70, 70:90] = 500
    th[60:66, 10:40] = 36
    dp = np.full((120, 160), 3.0, np.float32)
    bs = detect_blobs(th, dp, (0.0, 0.0, 0.0))
    fire = [b for b in bs if b.kind == 'fire'][0]
    assert abs(fire.xy[0] - 3.0) < 0.2 and abs(fire.xy[1]) < 0.2
    warm = [b for b in bs if b.kind == 'warm'][0]
    assert warm.xy[1] > 0  # left of image => +y
    assert warm.width_m > warm.height_m


def test_blob_at_edge_is_finite():
    th = np.full((120, 160), 20, np.float32)
    th[0:10, 150:160] = 36
    bs = detect_blobs(th, np.full((120, 160), 2.0, np.float32), (1, 1, 0))
    assert bs and all(np.isfinite(b.xy).all() for b in bs)


def test_blob_with_no_depth_is_skipped():
    th = np.full((120, 160), 20, np.float32)
    th[40:60, 40:60] = 36
    assert detect_blobs(th, np.full((120, 160), np.inf, np.float32), (0, 0, 0)) == []


def test_decoy_scores_below_casualty():
    cal = Calibrator()
    cas = Track(1, 'warm', (0, 0), deque([36] * 5), deque([True] * 5), 1.7, 0.4, breath=0.85)
    rad = Track(2, 'warm', (1, 0), deque([50] * 5), deque([True] * 5), 1.0, 0.6, breath=0.1)
    assert cal.conf(cas) >= 0.8 > 0.4 > cal.conf(rad)


def test_feedback_lowers_similar_track(tmp_path):
    cal = Calibrator(path=str(tmp_path / 'fb.jsonl'))
    t = Track(3, 'warm', (0, 0), deque([38] * 5), deque([True] * 5), 0.4, 0.3)
    before = cal.conf(t)
    cal.update(t, False)
    assert cal.conf(t) < before
    assert (tmp_path / 'fb.jsonl').read_text().count('\n') == 1


def test_tracker_associates_and_counts_misses():
    tr = Tracker()
    b = Blob('warm', (2.0, 0.0), 36.0, 1.7, 0.4)
    tr.update([b], 0.0, (0, 0, 0))
    tracks = tr.update([Blob('warm', (2.3, 0.1), 36.0, 1.7, 0.4)], 0.05, (0, 0, 0))
    assert len(tracks) == 1 and list(tracks[0].hits) == [True, True]
    tracks = tr.update([], 0.1, (0, 0, 0))  # still in view, not seen -> miss
    assert list(tracks[0].hits) == [True, True, False]
    tracks = tr.update([], 0.15, (0, 0, math.pi))  # looking away -> no miss recorded
    assert len(tracks[0].hits) == 3
