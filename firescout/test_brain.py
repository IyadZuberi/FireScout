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
    assert abs(fire.xy[0] - 3.3) < 0.2 and abs(fire.xy[1]) < 0.2  # near surface 3.0 m + 0.3 m to seat centre
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


# ---------------- Task 5: brain loop
from brain import Brain, Card, Command
from sensors import Observation


def obs(**kw):
    base = dict(t=0.0, pose=(0.6, 4.0, 0.0), thermal=np.full((120, 160), 20, np.float32),
                depth=np.full((120, 160), 5.0, np.float32), ranges=np.full(36, 6.0, np.float32),
                gas_ppm=0.0, comms=True, self_temp_c=25.0, battery=1.0)
    base.update(kw)
    return Observation(**base)


def person_frame():
    th = np.full((120, 160), 20, np.float32)
    th[60:66, 20:140] = 36  # wide, low warm blob straight ahead
    return th, np.full((120, 160), 2.0, np.float32)


def test_card_format():
    c = Card(34.0, 'REPLAN', 'thermal', 'C2=310°C(.92)>250', 'old 6.1m→new 10.3m (+10s)', 'heat-no-go')
    assert str(c).startswith('[00:34] REPLAN trigger=thermal C2=310')
    assert str(c).endswith('| rule=heat-no-go')


def test_retreat_when_self_hot():
    b = Brain(12, 8, (0.6, 4.0))
    b.step(obs(self_temp_c=65))
    assert b.mode == 'RETREAT' and b.cards[-1].kind == 'RETREAT'


def test_retreat_when_fire_dragged_onto_robot():
    b = Brain(12, 8, (2.0, 4.0))
    b.step(obs(pose=(2.0, 4.0, 0.0)))
    i, j = b.grid.to_cell(2.0, 4.0)
    b.grid.heat[i - 1:i + 2, j - 1:j + 2] = 600
    b.step(obs(t=0.05, pose=(2.0, 4.0, 0.0)))
    assert b.mode == 'RETREAT'


def test_hold_not_crash_when_all_blocked():
    b = Brain(4, 4, (0.5, 0.5))
    b.grid.occ[:] = 0
    b.grid.heat[8:, :] = 600  # cool pocket rows 0-7 (y<2 m), rest no-go, no frontiers
    b.grid.scanned[:] = True  # already looked at everything reachable
    for k in range(21):  # 1 s grace before HOLD
        cmd = b.step(obs(t=k * 0.05, pose=(0.5, 0.5, 0)))
    assert b.mode == 'HOLD' and b.cards[-1].kind == 'HOLD'
    assert abs(cmd.vx) < 1e-6 and abs(cmd.vy) < 1e-6


def test_no_card_says_clear():
    b = Brain(12, 8, (0.6, 4.0))
    for k in range(50):
        b.step(obs(t=k * 0.05))
    assert not any('clear' in str(c).lower() for c in b.cards)


def test_comms_loss_queues_and_continues():
    b = Brain(12, 8, (0.6, 4.0))
    cmd = b.step(obs(comms=False))
    assert b.mode == 'EXPLORE' and b.outbox and b.cards[-1].kind == 'COMMS'
    assert abs(cmd.vx) + abs(cmd.vy) > 0


def test_verify_requests_breath_then_pins():
    b = Brain(12, 8, (2.0, 4.0))
    th, dp = person_frame()
    cmd = None
    for k in range(4):
        cmd = b.step(obs(t=k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp))
    assert b.mode == 'VERIFY' and cmd.want_breath_xy is not None
    for k in range(20):  # ~1 s of radar breathing samples
        b.step(obs(t=0.25 + k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp, breath=0.9))
    assert any(c.kind == 'PIN' for c in b.cards)
    assert any(tr.state == 'pinned' for tr in b.tracker.tracks)


def test_no_breath_keeps_possible_not_dropped():
    b = Brain(12, 8, (2.0, 4.0))
    th, dp = person_frame()
    for k in range(4):
        b.step(obs(t=k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp))
    for k in range(20):
        b.step(obs(t=0.25 + k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp, breath=0.1))
    tr = [t for t in b.tracker.tracks if t.kind == 'warm'][0]
    assert tr.state in ('candidate', 'verify') and tr.verify_tries == 1  # kept, re-checked from another angle
    assert any(c.kind == 'POSSIBLE' and 'another angle' in c.change for c in b.cards)
    assert b._near(tr.xy, b._dist, avoid=tr.verify_from) is None or         math.dist(b.grid.to_world(*b._near(tr.xy, b._dist, avoid=tr.verify_from)), tr.verify_from) > 1.2


# ---------------- Task 9: should-tier
def test_gas_raises_cost():
    g = open_grid()
    base = cost_map(g)[8, 8]
    g.gas[8, 8] = 1000
    assert cost_map(g)[8, 8] > base + 1.5


def test_comms_lost_from_start_returns_when_done():
    b = Brain(4, 4, (0.5, 0.5))
    b.grid.occ[:] = 0
    b.grid.scanned[:] = True
    b.step(obs(pose=(0.5, 0.5, 0.0)))  # in radio range here -> last good comms cell
    for k in range(25):  # moved away, radio lost, nothing left to explore
        b.step(obs(t=0.05 * (k + 1), pose=(3.0, 3.0, 0.0), comms=False))
    assert b.mode == 'RETURN_COMMS' and b.outbox
    assert any(c.kind == 'COMMS' and 'returning' in c.change for c in b.cards)


def test_ic_feedback_rejects_and_logs(tmp_path):
    b = Brain(12, 8, (2.0, 4.0), feedback_path=str(tmp_path / 'fb.jsonl'))
    th, dp = person_frame()
    for k in range(3):
        b.step(obs(t=k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp))
    tr = [t for t in b.tracker.tracks if t.kind == 'warm'][0]
    b.feedback(tr.id, False)
    assert tr.state == 'rejected' and b.cards[-1].kind == 'LEARN'
    assert (tmp_path / 'fb.jsonl').exists()


# ---------------------------------------------------------------- smart search + IC control
def test_voice_steers_search_toward_bearing():
    b = Brain(12, 8, (6.0, 4.0))
    b.step(obs(pose=(6.0, 4.0, 0.0)))
    b.step(obs(t=0.1, pose=(6.0, 4.0, 0.0), voices=[(math.pi / 2, 50.0)]))  # shout from the robot's left = north
    assert any(c.kind == 'HEAR' for c in b.cards)
    gx, gy = b.grid.to_world(*b.goal)
    assert abs(math.atan2(gy - 4.0, gx - 6.0) - math.pi / 2) < math.radians(30)  # heading for the voice


def test_voice_marks_track_responsive():
    b = Brain(12, 8, (2.0, 4.0))
    th, dp = person_frame()
    for k in range(4):
        b.step(obs(t=k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp, voices=[(0.05, 55.0)]))
    assert any(tr.responsive for tr in b.tracker.tracks if tr.kind == 'warm')


def test_ic_stop_resume_recall():
    b = Brain(12, 8, (1.0, 4.0))
    b.grid.occ[:] = 0
    b.ic('STOP')
    cmd = b.step(obs(pose=(6.0, 4.0, math.pi)))
    assert b.mode == 'IC_STOP' and abs(cmd.vx) + abs(cmd.vy) < 1e-6 and b.cards[-1].kind == 'IC'
    b.ic('RECALL')
    b.step(obs(t=0.1, pose=(6.0, 4.0, math.pi)))
    assert b.mode == 'RECALL' and b.goal == b.start_cell
    b.step(obs(t=0.2, pose=(6.0, 4.0, math.pi), self_temp_c=70))  # safety gate beats any IC order
    assert b.mode == 'RETREAT'


def test_ic_search_area_restricts_goal():
    b = Brain(12, 8, (6.0, 4.0))
    b.ic('SEARCH', area=(9.0, 0.0, 12.0, 3.0))
    b.step(obs(pose=(6.0, 4.0, 0.0)))
    gx, gy = b.grid.to_world(*b.goal)
    assert 9.0 <= gx <= 12.0 and 0.0 <= gy <= 3.0


def _pinned(b, xy, responsive):
    tr = Track(len(b.tracker.tracks) + 1, 'warm', xy, deque([36.0]), deque([True]), 1.7, 0.3,
               breath=0.9, conf=0.9, state='pinned', responsive=responsive)
    b.tracker.tracks.append(tr)
    return tr


def test_priority_unresponsive_or_near_fire_is_p1():
    b = Brain(12, 8, (1.0, 4.0))
    b.fires = [(10.0, 6.0, 600.0)]
    near = _pinned(b, (9.0, 6.0), responsive=True)
    calm = _pinned(b, (3.0, 2.0), responsive=True)
    limp = _pinned(b, (3.0, 6.0), responsive=False)
    assert b.priority(near)[0] == 'P1' and b.priority(limp)[0] == 'P1' and b.priority(calm)[0] == 'P2'
    assert b.priority(near)[1] > b.priority(calm)[1]


def test_rescue_route_visits_highest_priority_first():
    b = Brain(12, 8, (1.0, 4.0))
    b.grid.occ[:] = 0
    b.fires = [(10.0, 6.0, 600.0)]
    _pinned(b, (3.0, 4.0), responsive=True)   # close but low priority
    hi = _pinned(b, (9.0, 6.0), responsive=False)
    b.step(obs(pose=(1.0, 4.0, 0.0)))
    first = b.rescue[0][0]
    assert b.rescue_order[first].id == hi.id


def test_adaptive_speed_fast_when_known_and_cool_slow_near_heat():
    b = Brain(12, 8, (1.0, 4.0))
    b.grid.occ[:] = 0
    b.grid.scanned[:] = True
    b.ic('RECALL')
    cmd = b.step(obs(pose=(8.0, 4.0, math.pi)))
    assert math.hypot(cmd.vx, cmd.vy) > 0.55
    i, j = b.grid.to_cell(7.0, 4.0)
    b.grid.heat[i - 1:i + 2, j - 1:j + 2] = 120
    cmd = b.step(obs(t=0.1, pose=(8.0, 4.0, math.pi)))
    assert math.hypot(cmd.vx, cmd.vy) < 0.3


def test_depth_edge_tells_furniture_from_wall():
    b = Brain(12, 8, (2.0, 4.0))
    dp = np.full((120, 160), 5.5, np.float32)
    dp[75:, 60:100] = 1.2   # low box 1.2 m ahead, far wall visible over its top
    b.step(obs(pose=(2.0, 4.0, 0.0), depth=dp))
    sem = b.semantic()
    i, j = b.grid.to_cell(3.3, 4.0)
    assert (sem[i - 1:i + 2, j - 1:j + 3] == 2).any()      # furniture
    i, j = b.grid.to_cell(8.0, 4.0)
    assert (sem[i - 2:i + 3, j - 2:j + 3] == 1).any()      # wall (tall, top never seen)


def test_speaker_evacuates_when_route_safe_else_shelters():
    b = Brain(12, 8, (1.0, 4.0))
    b.grid.occ[:] = 0
    tr = _pinned(b, (5.0, 4.0), responsive=True)
    b.step(obs(pose=(3.5, 4.0, 0.0)))
    assert tr.instruct == 'EVACUATE' and b.say_out and b.say_out[-1]['route']
    assert b.priority(tr)[0] == 'EVAC' and tr not in b.rescue_order

    b2 = Brain(12, 8, (1.0, 4.0))
    b2.grid.occ[:] = 0
    b2.grid.heat[:, 10:13] = 300          # wall of fire between the person and the door (x 2.5-3.25 m)
    b2.fires = [(2.9, 4.0, 600.0)]
    tr2 = _pinned(b2, (6.0, 4.0), responsive=True)
    b2.step(obs(pose=(4.5, 4.0, 0.0)))
    assert tr2.instruct == 'SHELTER' and b2.say_out[-1]['route'] is None
    assert any(c.kind == 'SAY' and 'stay put' in c.change for c in b2.cards)


def test_proximity_guard_stops_motion_into_close_obstacle():
    b = Brain(12, 8, (2.0, 4.0))
    rng = np.full(36, 6.0, np.float32)
    rng[0] = 0.3                                  # wall 0.3 m straight ahead (ray 0 = heading)
    cmd = b._guard(Command(0.4, 0.0, 0.0), obs(pose=(2.0, 4.0, 0.0), ranges=rng))
    assert cmd.vx <= 0.0                          # never pushes into it
    side = b._guard(Command(0.0, 0.4, 0.0), obs(pose=(2.0, 4.0, 0.0), ranges=rng))
    assert side.vy > 0.3                          # free directions untouched


def test_planner_keeps_robot_radius_from_walls():
    from brain import cost_map
    g = Grid(4, 4)
    g.occ[:] = 0
    g.occ[8, :] = 1                               # wall across the middle (y = 2.0-2.25 m)
    c = cost_map(g)
    assert not np.isfinite(c[7, 3]) and not np.isfinite(c[9, 3])   # cells touching the wall are forbidden
    assert np.isfinite(c[6, 3])
