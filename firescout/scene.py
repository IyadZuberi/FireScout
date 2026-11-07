"""Procedural smoke-filled floor + MJCF builder for FireScout.

World frame: x east (0..width), y north (0..height), metres. Ground truth lives
here and in sensors.py only; the robot brain never sees a Layout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

WALL_H = 2.0
WALL_T = 0.1
CORRIDOR = (3.2, 4.8)  # y-span of the central corridor
DOOR_W = 1.0


@dataclass
class Thing:
    name: str
    kind: Literal['fire', 'casualty', 'decoy', 'shelf']
    xy: tuple[float, float]
    size: tuple[float, float, float]
    temp_c: float
    breathing: bool = False
    yaw: float = 0.0
    style: str = ''  # visual style: radiator / pipe / pet / ''


@dataclass
class Layout:
    width: float
    height: float
    walls: list[tuple[float, float, float, float]]
    things: list[Thing]
    start: tuple[float, float]
    seed: int
    rooms: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0,y0,x1,y1


def _split(rng, x0, x1, n, min_w=2.4):
    """Split [x0,x1] into n spans of at least min_w."""
    free = (x1 - x0) - n * min_w
    cuts = np.sort(rng.uniform(0, free, n - 1)) if n > 1 else np.array([])
    widths = np.diff(np.concatenate([[0], cuts, [free]])) + min_w
    edges = x0 + np.concatenate([[0], np.cumsum(widths)])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]


def _wall_with_gap(a, b, fixed, gap_c, horizontal):
    """Segment from a to b along one axis at `fixed`, with a DOOR_W gap centred at gap_c."""
    g0, g1 = gap_c - DOOR_W / 2, gap_c + DOOR_W / 2
    segs = []
    for s, e in ((a, g0), (g1, b)):
        if e - s > 0.05:
            segs.append((s, fixed, e, fixed) if horizontal else (fixed, s, fixed, e))
    return segs


def generate_layout(seed: int, n_rooms: int = 5) -> Layout:
    rng = np.random.default_rng(seed)
    W, H = 12.0, 8.0
    c0, c1 = CORRIDOR
    n_top = n_rooms // 2 + n_rooms % 2
    n_bot = n_rooms - n_top
    walls: list[tuple[float, float, float, float]] = [
        (0, 0, W, 0), (0, H, W, H), (0, 0, 0, H), (W, 0, W, H)]
    rooms = []
    for (y0, y1, wall_y, n) in ((c1, H, c1, n_top), (0, c0, c0, n_bot)):
        spans = _split(rng, 0.0, W, n)
        for k, (x0, x1) in enumerate(spans):
            rooms.append((x0, y0, x1, y1))
            door = float(rng.uniform(x0 + 0.8, x1 - 0.8))
            walls += _wall_with_gap(x0, x1, wall_y, door, horizontal=True)
            if k > 0:  # dividing wall with an inter-room door -> alternative routes
                walls += _wall_with_gap(y0, y1, x0, float(rng.uniform(y0 + 0.8, y1 - 0.8)), horizontal=False)
    start = (1.0, (c0 + c1) / 2)

    things: list[Thing] = []
    taken: list[tuple[float, float]] = [start]

    def place(kind, size, temp, room_ok=True, **kw):
        for _ in range(500):
            if room_ok:
                x0, y0, x1, y1 = rooms[int(rng.integers(len(rooms)))]
                m = 0.5 + max(size[0], size[1]) / 2
                if x1 - x0 < 2 * m or y1 - y0 < 2 * m:
                    continue
                p = (float(rng.uniform(x0 + m, x1 - m)), float(rng.uniform(y0 + m, y1 - m)))
            else:  # corridor, against the south corridor wall
                p = (float(rng.uniform(3.0, W - 2.0)), c0 + 0.1 + size[1] / 2)
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= 1.0 for q in taken):
                taken.append(p)
                things.append(Thing(f'{kind}{len(things)}', kind, p, size, temp, **kw))
                return
        raise RuntimeError(f'could not place {kind} (seed {seed})')

    for _ in range(int(rng.integers(1, 3))):
        place('fire', (0.6, 0.6, 0.9), float(rng.uniform(400, 700)))
    for _ in range(int(rng.integers(0, 3))):
        place('casualty', (1.7, 0.5, 0.3), float(rng.uniform(34, 37)), breathing=True,
              yaw=float(rng.uniform(0, math.pi)))
    decoys = [('radiator', (1.0, 0.1, 0.6), float(rng.uniform(45, 60))),
              ('pipe', (1.2, 0.12, 0.12), 70.0),
              ('pet', (0.4, 0.3, 0.3), 38.0)]
    for i in rng.permutation(3)[: int(rng.integers(2, 4))]:
        style, size, temp = decoys[int(i)]
        place('decoy', size, temp, style=style)
    place('shelf', (0.9, 0.35, 1.2), 20.0, room_ok=False)
    return Layout(W, H, walls, things, start, seed, rooms)


# ---------------------------------------------------------------- MJCF


def _f(*v):
    return ' '.join(f'{x:.4g}' for x in v)


def _thing_geoms(i: int, t: Thing) -> str:
    """Visual + collision geoms for one Thing; every geom named thing{i}_g{k}."""
    sx, sy, sz = t.size
    g = []

    def add(**a):
        k = len(g)
        attrs = ' '.join(f'{key}="{val}"' for key, val in a.items())
        g.append(f'<geom name="thing{i}_g{k}" {attrs}/>')

    if t.kind == 'fire':
        add(type='box', size=_f(0.3, 0.3, 0.05), pos=_f(0, 0, 0.05), material='ember')
        for k, (r, z, mat) in enumerate(((0.28, 0.25, 'flame0'), (0.2, 0.55, 'flame1'), (0.12, 0.8, 'flame2'))):
            add(type='ellipsoid', size=_f(r, r, r * 1.4), pos=_f(0, 0, z), material=mat,
                contype='0', conaffinity='0', group='1')
    elif t.kind == 'casualty':  # lying figure along local x
        add(type='capsule', size=_f(0.13, 0.28), pos=_f(0.05, 0, 0.14), euler='0 90 0', material='cloth')
        add(type='sphere', size='0.11', pos=_f(0.55, 0, 0.12), material='skin')
        for s in (-1, 1):
            add(type='capsule', size=_f(0.06, 0.3), pos=_f(-0.55, 0.1 * s, 0.07), euler='0 90 0', material='denim')
            add(type='capsule', size=_f(0.045, 0.24), pos=_f(0.1, 0.2 * s, 0.06), euler='0 90 0', material='skin')
    elif t.style == 'radiator':
        for k in range(6):
            add(type='box', size=_f(0.07, 0.04, sz / 2), pos=_f(-0.4 + k * 0.16, 0, sz / 2 + 0.05), material='metal')
    elif t.style == 'pipe':
        add(type='cylinder', size=_f(0.06, sx / 2), pos=_f(0, 0, 0.1), euler='0 90 0', material='copper')
    elif t.style == 'pet':
        add(type='capsule', size=_f(0.1, 0.12), pos=_f(0, 0, 0.11), euler='0 90 0', material='fur')
        add(type='sphere', size='0.08', pos=_f(0.22, 0, 0.16), material='fur')
    elif t.kind == 'shelf':
        for s in (-1, 1):
            add(type='box', size=_f(0.02, sy / 2, sz / 2), pos=_f(s * (sx / 2 - 0.02), 0, sz / 2), material='wood')
        for z in (0.05, 0.45, 0.85, 1.18):
            add(type='box', size=_f(sx / 2, sy / 2, 0.02), pos=_f(0, 0, z), material='wood')
    return '\n      '.join(g)


def build_mjcf(layout: Layout) -> str:
    W, H = layout.width, layout.height
    walls = []
    for k, (x0, y0, x1, y1) in enumerate(layout.walls):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx = max(abs(x1 - x0) / 2, WALL_T / 2)
        hy = max(abs(y1 - y0) / 2, WALL_T / 2)
        walls.append(f'<geom name="wall{k}" type="box" pos="{_f(cx, cy, WALL_H / 2)}" '
                     f'size="{_f(hx, hy, WALL_H / 2)}" material="plaster"/>')
    things = []
    for i, t in enumerate(layout.things):
        light = ''
        if t.kind == 'fire':
            light = (f'<light name="firelight{i}" type="point" pos="0 0 0.9" diffuse="1 0.5 0.1" '
                     f'specular="0.3 0.15 0.05" attenuation="1 0.2 0.6" castshadow="false"/>')
        things.append(
            f'<body name="thing{i}" pos="{_f(t.xy[0], t.xy[1], 0.01)}" euler="0 0 {math.degrees(t.yaw):.3g}">\n'
            f'      <freejoint/>\n      {light}\n      {_thing_geoms(i, t)}\n    </body>')
    sites = '\n      '.join(
        f'<site name="rf{k}" pos="0 0 0.12" zaxis="{_f(math.cos(math.radians(10 * k)), math.sin(math.radians(10 * k)), 0)}" '
        f'size="0.01" rgba="0 1 0 0"/>' for k in range(36))
    sensors = '\n    '.join(f'<rangefinder name="rf{k}" site="rf{k}" cutoff="6"/>' for k in range(36))
    sx, sy = layout.start
    return f"""<mujoco model="firescout">
  <option timestep="0.005"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight ambient="0.2 0.2 0.2" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <map fogstart="1" fogend="7" znear="0.02"/>
    <rgba fog="0.32 0.32 0.34 1" haze="0.3 0.3 0.32 1"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.33 0.32" rgb2="0.08 0.08 0.1" width="512" height="512"/>
    <texture name="concrete" type="2d" builtin="checker" rgb1="0.42 0.41 0.39" rgb2="0.37 0.36 0.35" width="512" height="512" mark="edge" markrgb="0.3 0.3 0.3"/>
    <material name="floor" texture="concrete" texrepeat="12 8" reflectance="0.1"/>
    <material name="plaster" rgba="0.78 0.75 0.7 1" specular="0.05" shininess="0.1"/>
    <material name="ember" rgba="0.35 0.06 0.02 1" emission="0.6"/>
    <material name="flame0" rgba="1 0.35 0.05 0.85" emission="1"/>
    <material name="flame1" rgba="1 0.6 0.1 0.75" emission="1"/>
    <material name="flame2" rgba="1 0.9 0.4 0.6" emission="1"/>
    <material name="skin" rgba="0.85 0.65 0.5 1"/>
    <material name="cloth" rgba="0.2 0.35 0.6 1"/>
    <material name="denim" rgba="0.15 0.18 0.3 1"/>
    <material name="metal" rgba="0.85 0.85 0.85 1" specular="0.6" shininess="0.6"/>
    <material name="copper" rgba="0.72 0.45 0.2 1" specular="0.7" shininess="0.7"/>
    <material name="fur" rgba="0.55 0.4 0.25 1"/>
    <material name="wood" rgba="0.55 0.38 0.22 1"/>
    <material name="chassis" rgba="0.6 0.08 0.06 1" specular="0.4" shininess="0.5"/>
    <material name="track" rgba="0.08 0.08 0.08 1"/>
    <material name="lens" rgba="0.05 0.05 0.08 1" specular="1" shininess="1"/>
  </asset>
  <worldbody>
    <light name="ceil0" type="spot" pos="{_f(W * 0.3, H / 2, 6)}" dir="0 0 -1" diffuse="0.45 0.45 0.45" cutoff="70" castshadow="true"/>
    <light name="ceil1" type="spot" pos="{_f(W * 0.75, H / 2, 6)}" dir="0 0 -1" diffuse="0.45 0.45 0.45" cutoff="70" castshadow="true"/>
    <geom name="floor" type="plane" size="{_f(W / 2 + 1, H / 2 + 1, 0.1)}" pos="{_f(W / 2, H / 2, 0)}" material="floor"/>
    {chr(10).join('    ' + w for w in walls).strip()}
    <body name="scout" pos="{_f(sx, sy, 0.0)}">
      <joint name="x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="y" type="slide" axis="0 1 0" damping="1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.2"/>
      <geom name="chassis" type="box" size="0.2 0.12 0.05" pos="0 0 0.09" material="chassis" mass="3"/>
      <geom name="trackL" type="box" size="0.22 0.04 0.05" pos="0 0.16 0.06" material="track" mass="0.5"/>
      <geom name="trackR" type="box" size="0.22 0.04 0.05" pos="0 -0.16 0.06" material="track" mass="0.5"/>
      <geom name="mast" type="cylinder" size="0.025 0.08" pos="0.08 0 0.22" material="track" mass="0.1"/>
      <geom name="lens" type="cylinder" size="0.03 0.02" pos="0.12 0 0.3" euler="0 90 0" material="lens" mass="0.05"/>
      <camera name="thermal" pos="0.15 0 0.3" xyaxes="0 -1 0 0 0 1" fovy="43.6"/>
      <camera name="chase" mode="trackcom" pos="-0.9 0 2.8" xyaxes="0 -1 0 0.85 0 0.53" fovy="70"/>
      {sites}
    </body>
    {chr(10).join('    ' + t for t in things).strip()}
  </worldbody>
  <actuator>
    <velocity name="ax" joint="x" kv="50" ctrlrange="-1 1"/>
    <velocity name="ay" joint="y" kv="50" ctrlrange="-1 1"/>
    <velocity name="ayaw" joint="yaw" kv="10" ctrlrange="-1.5 1.5"/>
  </actuator>
  <sensor>
    {sensors}
  </sensor>
</mujoco>"""


def load(layout: Layout) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(build_mjcf(layout))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d
