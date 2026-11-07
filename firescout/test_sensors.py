import math

import mujoco
import numpy as np

from scene import Layout, Thing, load
from sensors import Sensors

OUTER = [(0, 0, 12, 0), (0, 8, 12, 8), (0, 0, 0, 8), (12, 0, 12, 8)]


def open_layout():
    things = [
        Thing('fire0', 'fire', (6.0, 4.0), (0.6, 0.6, 0.9), 600.0),
        Thing('casualty1', 'casualty', (6.0, 6.5), (1.7, 0.5, 0.3), 36.0, breathing=True, responsive=True),
        Thing('decoy2', 'decoy', (3.0, 6.5), (1.0, 0.1, 0.6), 55.0, style='radiator'),
        Thing('shelf3', 'shelf', (9.0, 1.5), (0.9, 0.35, 1.2), 20.0),
    ]
    return Layout(12, 8, OUTER, things, (1.0, 4.0), 0)


def face(m, d, target, dist, from_angle):
    """Put the robot `dist` m from target at `from_angle`, looking at it."""
    x = target[0] + dist * math.cos(from_angle)
    y = target[1] + dist * math.sin(from_angle)
    d.qpos[0:3] = [x - 1.0, y - 4.0, math.atan2(target[1] - y, target[0] - x)]  # joints are offsets from start
    mujoco.mj_forward(m, d)


def setup():
    L = open_layout()
    m, d = load(L)
    return L, m, d, Sensors(m, d, L, seed=0)


def test_thermal_sees_fire_hotter_than_ambient():
    L, m, d, s = setup()
    face(m, d, (6.0, 4.0), 2.0, math.pi)
    o = s.observe()
    assert o.thermal.shape == (120, 160) and o.thermal.dtype == np.float32
    assert o.thermal.max() > 150 and np.percentile(o.thermal, 10) < 25  # fire hot, background stays near ambient
    assert o.depth.shape == (120, 160)


def test_ranges_shape_and_bounds():
    L, m, d, s = setup()
    o = s.observe()
    assert o.ranges.shape == (36,) and (o.ranges > 0).all() and (o.ranges <= 6.0).all()
    assert abs(o.ranges[18] - 1.0) < 0.3  # pointing west, outer wall 1 m away


def test_breath_only_for_breathing_casualty():
    L, m, d, s = setup()
    face(m, d, (6.0, 6.5), 2.0, -math.pi / 2)
    assert s.observe(want_breath_xy=(6.0, 6.5)).breath > 0.6
    face(m, d, (3.0, 6.5), 2.0, -math.pi / 2)
    assert s.observe(want_breath_xy=(3.0, 6.5)).breath < 0.4
    assert s.observe().breath is None


def test_breath_needs_range():
    L, m, d, s = setup()
    face(m, d, (6.0, 6.5), 4.5, -math.pi / 2 - 0.3)
    assert s.observe(want_breath_xy=(6.0, 6.5)).breath < 0.4


def test_comms_drop_flag():
    L, m, d, s = setup()
    assert s.observe().comms is True
    s.force_comms_drop = True
    assert s.observe().comms is False


def test_gas_and_self_temp_rise_near_fire():
    L, m, d, s = setup()
    far = s.observe().gas_ppm
    face(m, d, (6.0, 4.0), 0.9, math.pi)
    near = s.observe().gas_ppm
    assert near > far + 300
    t0 = s.observe().self_temp_c
    for k in range(200):
        d.time += 0.05
        o = s.observe()
    assert o.self_temp_c > t0 + 5


def test_ignite_shelf_makes_it_hot():
    L, m, d, s = setup()
    s.ignite(3, 450.0)
    face(m, d, (9.0, 1.5), 2.0, math.pi)
    assert s.observe().thermal.max() > 150


def test_o2_drops_near_fire():
    L, m, d, s = setup()
    face(m, d, (6.0, 4.0), 5.0, math.pi)
    far = s.observe().o2_pct
    face(m, d, (6.0, 4.0), 0.9, math.pi)
    near = s.observe().o2_pct
    assert 20.5 <= far <= 21.0 and near < 19.0


def test_voice_heard_with_relative_bearing():
    L, m, d, s = setup()
    face(m, d, (6.0, 6.5), 3.0, -math.pi / 2)   # casualty straight ahead
    heard = []
    for k in range(80):
        d.time = k * 0.1
        heard += s.observe().voices
    assert heard, 'responsive casualty never heard'
    assert all(abs(b) < 0.5 for b, db in heard) and all(db > 30 for b, db in heard)
    L.things[1].responsive = False
    assert not any(s.observe().voices for k in range(5))
