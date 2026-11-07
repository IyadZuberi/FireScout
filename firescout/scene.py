"""Procedural smoke-filled care-home floor + MJCF builder for FireScout.

World frame: x east (0..width), y north (0..height), metres. Ground truth lives here and in
sensors.py only; the robot brain never sees a Layout.
Furniture meshes: NVIDIA SimReady assets (CC-BY-4.0), converted by assets.py into firescout/assets/.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, 'assets')
WALL_H = 2.4
WALL_T = 0.12
CORRIDOR = (3.2, 4.8)  # y-span of the central corridor
DOOR_W = 1.0
N_SMOKE = 36
SMOKE_GROUP = 2  # rendered in the viewer + RGB camera, invisible to the thermal camera (LWIR sees through smoke)

# prop key -> (mesh scale xyz, collider half-size xyz or None, rgb override)
PROP_SPEC = {
    'bed': ((0.65, 1.0, 1.0), (1.05, 0.5, 0.3)),
    'sofa': ((1.0, 1.0, 1.0), (0.35, 0.36, 0.36)),
    'chair': ((1.0, 1.0, 1.0), (0.3, 0.3, 0.25)),
    'desk': ((1.0, 1.0, 1.0), (0.4, 0.9, 0.38)),
    'cabinet': ((1.0, 1.0, 0.55), (0.2, 0.41, 0.68)),
    'radiator': ((1.0, 1.0, 1.0), None),
    'iv': ((1.0, 1.0, 1.0), (0.12, 0.12, 0.9)),
    'extinguisher': ((1.0, 1.0, 1.0), (0.07, 0.09, 0.27)),
    'bin': ((1.0, 1.0, 1.0), (0.16, 0.09, 0.14)),
    'tv': ((1.0, 1.0, 1.0), (0.05, 0.6, 0.38)),
}


def asset_meta() -> dict:
    p = os.path.join(ASSET_DIR, 'meta.json')
    return json.load(open(p)) if os.path.exists(p) else {}


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
class Prop:
    key: str
    xy: tuple[float, float]
    yaw: float = 0.0
    z: float = 0.0


@dataclass
class Layout:
    width: float
    height: float
    walls: list[tuple[float, float, float, float]]
    things: list[Thing]
    start: tuple[float, float]
    seed: int
    rooms: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0,y0,x1,y1
    props: list[Prop] = field(default_factory=list)


def _split(rng, x0, x1, n, min_w=2.6):
    free = (x1 - x0) - n * min_w
    cuts = np.sort(rng.uniform(0, free, n - 1)) if n > 1 else np.array([])
    widths = np.diff(np.concatenate([[0], cuts, [free]])) + min_w
    edges = x0 + np.concatenate([[0], np.cumsum(widths)])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]


def _wall_with_gap(a, b, fixed, gap_c, horizontal):
    g0, g1 = gap_c - DOOR_W / 2, gap_c + DOOR_W / 2
    segs = []
    for s, e in ((a, g0), (g1, b)):
        if e - s > 0.05:
            segs.append((s, fixed, e, fixed) if horizontal else (fixed, s, fixed, e))
    return segs


def _furnish(rng, rooms, doors, has_assets) -> tuple[list[Prop], list[tuple]]:
    """Care-home furniture. Returns props and their footprints (cx, cy, hx, hy) for collision-free placement."""
    props, foot = [], []
    if not has_assets:
        return props, foot

    def add(key, x, y, yaw=0.0, hx=0.3, hy=0.3):
        props.append(Prop(key, (float(x), float(y)), float(yaw)))
        if abs(math.sin(yaw)) > 0.7:
            hx, hy = hy, hx
        foot.append((float(x), float(y), hx, hy))

    kinds = ['bedroom'] * len(rooms)
    order = rng.permutation(len(rooms))
    kinds[int(order[0])] = 'lounge'
    kinds[int(order[1])] = 'office'
    for (x0, y0, x1, y1), door_x, kind in zip(rooms, doors, kinds):
        top = y0 >= CORRIDOR[1] - 1e-6  # room north of the corridor: door on its south wall
        far = y1 - 0.62 if top else y0 + 0.62
        near = y0 + 0.6 if top else y1 - 0.6
        cx = (x0 + x1) / 2
        if kind == 'bedroom':
            bx = x0 + 1.25 if door_x > cx else x1 - 1.25  # bed away from the door
            add('bed', bx, far, 0.0, 1.05, 0.5)
            add('chair', x1 - 0.45 if bx < cx else x0 + 0.45, (far + near) / 2, math.pi / 2 * (1 if top else -1))
            add('cabinet', x0 + 0.3 if bx > cx else x1 - 0.3, far, 0.0, 0.2, 0.41)
            if rng.random() < 0.6:
                add('iv', bx + (1.3 if bx < cx else -1.3), far - (0.3 if top else -0.3), 0.0, 0.15, 0.15)
            add('bin', x1 - 0.25 if door_x < cx else x0 + 0.25, near + (0.05 if top else -0.05), 0.0, 0.16, 0.1)
        elif kind == 'lounge':
            add('sofa', cx - 0.5, far, math.pi if top else 0.0, 0.36, 0.36)
            add('sofa', cx + 0.4, far, math.pi if top else 0.0, 0.36, 0.36)
            add('tv', cx, near + (0.1 if top else -0.1), math.pi / 2, 0.05, 0.6)
        else:
            add('desk', cx, far, math.pi / 2, 0.4, 0.9)
            add('chair', cx, far + (-0.75 if top else 0.75), math.pi / 2 * (-1 if top else 1))
            add('cabinet', x1 - 0.3, (far + near) / 2, 0.0, 0.2, 0.41)
    add('extinguisher', 2.2, CORRIDOR[1] - 0.15, 0.0, 0.07, 0.09)
    return props, foot


def generate_layout(seed: int, n_rooms: int = 5) -> Layout:
    rng = np.random.default_rng(seed)
    W, H = 12.0, 8.0
    c0, c1 = CORRIDOR
    n_top = n_rooms // 2 + n_rooms % 2
    n_bot = n_rooms - n_top
    walls: list[tuple[float, float, float, float]] = [
        (0, 0, W, 0), (0, H, W, H), (0, 0, 0, H), (W, 0, W, H)]
    rooms, doors = [], []
    for (y0, y1, wall_y, n) in ((c1, H, c1, n_top), (0, c0, c0, n_bot)):
        spans = _split(rng, 0.0, W, n)
        for k, (x0, x1) in enumerate(spans):
            rooms.append((x0, y0, x1, y1))
            door = float(rng.uniform(x0 + 0.8, x1 - 0.8))
            doors.append(door)
            walls += _wall_with_gap(x0, x1, wall_y, door, horizontal=True)
            if k > 0:  # dividing wall with an inter-room door -> alternative routes
                walls += _wall_with_gap(y0, y1, x0, float(rng.uniform(y0 + 0.9, y1 - 0.9)), horizontal=False)
    start = (1.0, (c0 + c1) / 2)
    props, foot = _furnish(rng, rooms, doors, bool(asset_meta()))

    things: list[Thing] = []
    taken: list[tuple[float, float]] = [start]

    def clear_of_props(p, r):
        return all(abs(p[0] - fx) > hx + r or abs(p[1] - fy) > hy + r for fx, fy, hx, hy in foot)

    def place(kind, size, temp, room_ok=True, near_bed=False, **kw):
        r = max(size[0], size[1]) / 2
        for _ in range(800):
            if near_bed and props:  # lying on the floor beside a bed
                beds = [p for p in props if p.key == 'bed']
                b = beds[int(rng.integers(len(beds)))]
                top = b.xy[1] > c1
                p = (b.xy[0] + float(rng.uniform(-0.4, 0.4)), b.xy[1] + (-1.0 if top else 1.0))
                rr = 0.35
            elif room_ok:
                x0, y0, x1, y1 = rooms[int(rng.integers(len(rooms)))]
                m = 0.35 + r
                if x1 - x0 < 2 * m or y1 - y0 < 2 * m:
                    continue
                p = (float(rng.uniform(x0 + m, x1 - m)), float(rng.uniform(y0 + m, y1 - m)))
                rr = r
            else:  # corridor, against the south corridor wall
                p = (float(rng.uniform(3.0, W - 2.0)), c0 + 0.1 + size[1] / 2)
                rr = 0.1
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= 1.0 for q in taken) and clear_of_props(p, rr * 0.6):
                taken.append(p)
                things.append(Thing(f'{kind}{len(things)}', kind, p, size, temp, **kw))
                return
            near_bed = near_bed and _ < 200
        raise RuntimeError(f'could not place {kind} (seed {seed})')

    for _ in range(int(rng.integers(1, 3))):
        place('fire', (0.6, 0.6, 0.9), float(rng.uniform(400, 700)))
    for _ in range(int(rng.integers(0, 3))):
        place('casualty', (1.7, 0.5, 0.3), float(rng.uniform(34, 37)), near_bed=rng.random() < 0.7,
              breathing=True, yaw=float(rng.choice([0.0, math.pi]) + rng.uniform(-0.3, 0.3)))
    decoys = [('radiator', (0.08, 0.55, 0.5), float(rng.uniform(45, 60))),
              ('pipe', (1.2, 0.12, 0.12), 70.0),
              ('pet', (0.4, 0.3, 0.3), 38.0)]
    for i in rng.permutation(3)[: int(rng.integers(2, 4))]:
        style, size, temp = decoys[int(i)]
        place('decoy', size, temp, style=style)
    place('shelf', (0.4, 0.82, 1.36), 20.0, room_ok=False)
    return Layout(W, H, walls, things, start, seed, rooms, props)


# ---------------------------------------------------------------- MJCF


def _f(*v):
    return ' '.join(f'{x:.4g}' for x in v)


def _human(add, rng):
    """Adult lying on the floor along local x (head at +x). ~1.75 m."""
    shirt = ['shirt_a', 'shirt_b', 'shirt_c'][int(rng.integers(3))]
    add(type='sphere', size='0.105', pos=_f(0.78, 0, 0.11), material='skin')                     # head
    add(type='capsule', size=_f(0.05, 0.05), pos=_f(0.64, 0, 0.1), euler='0 90 0', material='skin')  # neck
    add(type='ellipsoid', size=_f(0.22, 0.17, 0.11), pos=_f(0.38, 0, 0.12), material=shirt)       # chest
    add(type='ellipsoid', size=_f(0.16, 0.15, 0.1), pos=_f(0.12, 0, 0.11), material=shirt)        # belly
    add(type='ellipsoid', size=_f(0.13, 0.17, 0.1), pos=_f(-0.05, 0, 0.1), material='trousers')   # pelvis
    for s in (-1, 1):
        add(type='capsule', size=_f(0.045, 0.13), pos=_f(0.36, 0.23 * s, 0.07), euler=_f(0, 90, 8 * s), material=shirt)
        add(type='capsule', size=_f(0.038, 0.12), pos=_f(0.1, 0.27 * s, 0.05), euler=_f(0, 90, 15 * s), material='skin')
        add(type='sphere', size='0.045', pos=_f(-0.04, 0.3 * s, 0.045), material='skin')
        add(type='capsule', size=_f(0.07, 0.19), pos=_f(-0.33, 0.1 * s, 0.075), euler=_f(0, 90, 4 * s), material='trousers')
        add(type='capsule', size=_f(0.055, 0.18), pos=_f(-0.72, 0.13 * s, 0.06), euler=_f(0, 90, 3 * s), material='trousers')
        add(type='box', size=_f(0.03, 0.045, 0.08), pos=_f(-0.95, 0.14 * s, 0.08), material='shoe')


def _thing_geoms(i: int, t: Thing, rng, meta) -> str:
    """Visual + collision geoms for one Thing; every geom named thing{i}_g{k}."""
    sx, sy, sz = t.size
    g = []

    def add(**a):
        k = len(g)
        attrs = ' '.join(f'{key}="{val}"' for key, val in a.items())
        g.append(f'<geom name="thing{i}_g{k}" {attrs}/>')

    if t.kind == 'fire':  # burning debris pile + layered flames (animated by run.py)
        add(type='box', size=_f(0.32, 0.28, 0.06), pos=_f(0, 0, 0.06), euler='0 0 12', material='char')
        add(type='box', size=_f(0.2, 0.08, 0.05), pos=_f(0.12, 0.1, 0.15), euler='0 25 40', material='char')
        add(type='ellipsoid', size=_f(0.3, 0.3, 0.06), pos=_f(0, 0, 0.12), material='ember', contype='0', conaffinity='0')
        for k, (r, z, mat) in enumerate(((0.34, 0.32, 'flame0'), (0.26, 0.75, 'flame1'), (0.18, 1.12, 'flame2'),
                                         (0.12, 1.4, 'flame3'))):
            add(type='ellipsoid', size=_f(r, r * 0.9, r * 1.5), pos=_f(0.03 * k, -0.02 * k, z), material=mat,
                contype='0', conaffinity='0', group='1')
    elif t.kind == 'casualty':
        _human(add, rng)
    elif t.style == 'radiator':
        if 'radiator' in meta:
            add(type='mesh', mesh='radiator', material='mradiator', contype='0', conaffinity='0')
            add(type='box', size=_f(0.04, 0.27, 0.24), pos=_f(0, 0, 0.25), group='3', rgba='0 0 0 0')
        else:
            for k in range(6):
                add(type='box', size=_f(0.04, 0.04, 0.24), pos=_f(0, -0.25 + k * 0.1, 0.29), material='metal')
    elif t.style == 'pipe':
        add(type='cylinder', size=_f(0.06, sx / 2), pos=_f(0, 0, 0.1), euler='0 90 0', material='copper')
    elif t.style == 'pet':
        add(type='ellipsoid', size=_f(0.2, 0.1, 0.1), pos=_f(0, 0, 0.11), material='fur')
        add(type='sphere', size='0.07', pos=_f(0.22, 0, 0.16), material='fur')
        add(type='capsule', size=_f(0.02, 0.08), pos=_f(-0.24, 0, 0.1), euler='0 60 0', material='fur')
    elif t.kind == 'shelf':
        if 'cabinet' in meta:
            add(type='mesh', mesh='cabinet', material='mcabinet', contype='0', conaffinity='0')
            add(type='box', size=_f(0.2, 0.41, 0.68), pos=_f(0, 0, 0.68), group='3', rgba='0 0 0 0')
        else:
            add(type='box', size=_f(sx / 2, sy / 2, sz / 2), pos=_f(0, 0, sz / 2), material='wood')
    return '\n      '.join(g)


def _robot() -> str:
    wheels = []
    for s in (-1, 1):
        for k, x in enumerate((-0.17, -0.06, 0.06, 0.17)):
            wheels.append(f'<geom type="cylinder" size="0.045 0.035" pos="{x} {0.165 * s} 0.05" euler="90 0 0" '
                          f'material="wheel" contype="0" conaffinity="0" mass="0.05"/>')
    sites = '\n      '.join(
        f'<site name="rf{k}" pos="0 0 0.14" zaxis="{_f(math.cos(math.radians(10 * k)), math.sin(math.radians(10 * k)), 0)}" '
        f'size="0.01" rgba="0 1 0 0"/>' for k in range(36))
    return f"""<body name="scout" pos="__START__">
      <joint name="x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="y" type="slide" axis="0 1 0" damping="1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.2"/>
      <geom name="chassis" type="box" size="0.21 0.12 0.055" pos="0 0 0.1" material="chassis" mass="3"/>
      <geom name="deck" type="box" size="0.16 0.1 0.015" pos="-0.02 0 0.17" material="deck" mass="0.2"/>
      <geom name="trackL" type="capsule" size="0.054 0.2" pos="0 0.165 0.056" euler="0 90 0" material="track" mass="0.5"/>
      <geom name="trackR" type="capsule" size="0.054 0.2" pos="0 -0.165 0.056" euler="0 90 0" material="track" mass="0.5"/>
      {chr(10).join('      ' + w for w in wheels).strip()}
      <geom name="mast" type="cylinder" size="0.022 0.07" pos="0.06 0 0.25" material="deck" mass="0.1"/>
      <geom name="head" type="box" size="0.05 0.08 0.035" pos="0.08 0 0.33" material="head" mass="0.15"/>
      <geom name="lens" type="cylinder" size="0.022 0.012" pos="0.13 0.035 0.33" euler="0 90 0" material="lens" mass="0.01"/>
      <geom name="radar" type="box" size="0.006 0.03 0.025" pos="0.132 -0.035 0.33" material="radar" mass="0.01"/>
      <geom name="beacon" type="sphere" size="0.022" pos="-0.12 0 0.2" material="beacon" contype="0" conaffinity="0" mass="0.01"/>
      <geom name="antenna" type="cylinder" size="0.004 0.09" pos="-0.15 0.07 0.27" material="track" contype="0" conaffinity="0" mass="0.01"/>
      <light name="searchlight" type="spot" pos="0.14 0 0.3" dir="1 0 -0.25" diffuse="0.9 0.9 0.8" cutoff="32" exponent="8" castshadow="true"/>
      <camera name="thermal" pos="0.15 0 0.33" xyaxes="0 -1 0 0 0 1" fovy="43.6"/>
      <camera name="chase" mode="trackcom" pos="-0.9 0 2.8" xyaxes="0 -1 0 0.85 0 0.53" fovy="70"/>
      {sites}
    </body>"""


def build_mjcf(layout: Layout) -> str:
    W, H = layout.width, layout.height
    meta = asset_meta()
    rng = np.random.default_rng(layout.seed + 1000)
    walls = []
    for k, (x0, y0, x1, y1) in enumerate(layout.walls):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx = max(abs(x1 - x0) / 2, WALL_T / 2)
        hy = max(abs(y1 - y0) / 2, WALL_T / 2)
        walls.append(f'<geom name="wall{k}" type="box" pos="{_f(cx, cy, WALL_H / 2)}" '
                     f'size="{_f(hx, hy, WALL_H / 2)}" material="plaster"/>')
        walls.append(f'<geom name="skirt{k}" type="box" pos="{_f(cx, cy, 0.05)}" size="{_f(hx + 0.005, hy + 0.005, 0.05)}" '
                     f'material="skirting" contype="0" conaffinity="0"/>')
    props = []
    for k, p in enumerate(layout.props):
        if p.key not in meta:
            continue
        scale, col = PROP_SPEC[p.key]
        coll = '' if col is None else (f'<geom name="prop{k}_c" type="box" size="{_f(*col)}" pos="0 0 {col[2]:.3g}" '
                                       f'group="3" rgba="0 0 0 0"/>')
        props.append(f'<body name="prop{k}" pos="{_f(p.xy[0], p.xy[1], p.z)}" euler="0 0 {math.degrees(p.yaw):.4g}">'
                     f'<geom name="prop{k}_v" type="mesh" mesh="{p.key}" material="m{p.key}" contype="0" conaffinity="0"/>'
                     f'{coll}</body>')
    things = []
    for i, t in enumerate(layout.things):
        light = ''
        if t.kind == 'fire':
            light = (f'<light name="firelight{i}" type="point" pos="0 0 1.0" diffuse="1.6 0.75 0.15" '
                     f'specular="0.3 0.15 0.05" attenuation="1 0.25 0.5" castshadow="false"/>')
        things.append(
            f'<body name="thing{i}" pos="{_f(t.xy[0], t.xy[1], 0.01)}" euler="0 0 {math.degrees(t.yaw):.3g}">\n'
            f'      <freejoint/>\n      {light}\n      {_thing_geoms(i, t, rng, meta)}\n    </body>')
    smoke = []
    fires = [t.xy for t in layout.things if t.kind == 'fire'] or [(W / 2, H / 2)]
    for k in range(N_SMOKE):
        f = fires[k % len(fires)]
        x, y = f[0] + rng.normal(0, 1.5), f[1] + rng.normal(0, 1.2)
        r = rng.uniform(0.9, 1.5)
        smoke.append(f'<body name="smoke{k}" mocap="true" pos="{_f(x, y, rng.uniform(1.6, 2.3))}">'
                     f'<geom type="ellipsoid" size="{_f(r, r * 0.9, r * 0.35)}" material="smoke" group="{SMOKE_GROUP}" '
                     f'contype="0" conaffinity="0"/></body>')
    mesh_assets = ''.join(
        f'<mesh name="{k}" file="{k}.obj" scale="{_f(*PROP_SPEC[k][0])}"/>'
        f'<material name="m{k}" rgba="{_f(*v["rgb"])} 1" specular="0.35" shininess="0.4"/>'
        for k, v in meta.items())
    sensors = '\n    '.join(f'<rangefinder name="rf{k}" site="rf{k}" cutoff="6"/>' for k in range(36))
    sx, sy = layout.start
    return f"""<mujoco model="firescout">
  <compiler meshdir="{ASSET_DIR.replace(os.sep, '/')}"/>
  <option timestep="0.005"/>
  <visual>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.16 0.16 0.17" diffuse="0.22 0.22 0.22" specular="0 0 0"/>
    <map fogstart="1.5" fogend="9" znear="0.02" haze="0.25"/>
    <rgba fog="0.3 0.3 0.32 1" haze="0.22 0.21 0.22 1"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.28 0.26 0.25" rgb2="0.05 0.05 0.06" width="512" height="512"/>
    <texture name="vinyl" type="2d" builtin="checker" rgb1="0.46 0.47 0.45" rgb2="0.42 0.43 0.41" width="512" height="512" mark="edge" markrgb="0.36 0.37 0.36"/>
    <material name="floor" texture="vinyl" texrepeat="20 13" reflectance="0.12" specular="0.3"/>
    <material name="plaster" rgba="0.83 0.81 0.76 1" specular="0.05" shininess="0.1"/>
    <material name="skirting" rgba="0.42 0.4 0.37 1"/>
    <material name="char" rgba="0.08 0.06 0.05 1"/>
    <material name="ember" rgba="1 0.25 0.02 1" emission="1"/>
    <material name="flame0" rgba="1 0.32 0.04 0.85" emission="1"/>
    <material name="flame1" rgba="1 0.52 0.08 0.72" emission="1"/>
    <material name="flame2" rgba="1 0.78 0.25 0.55" emission="1"/>
    <material name="flame3" rgba="1 0.92 0.6 0.35" emission="1"/>
    <material name="smoke" rgba="0.24 0.24 0.25 0.07"/>
    <material name="skin" rgba="0.82 0.62 0.5 1" specular="0.15"/>
    <material name="shirt_a" rgba="0.62 0.7 0.82 1"/>
    <material name="shirt_b" rgba="0.75 0.62 0.68 1"/>
    <material name="shirt_c" rgba="0.6 0.68 0.6 1"/>
    <material name="trousers" rgba="0.2 0.22 0.3 1"/>
    <material name="shoe" rgba="0.12 0.1 0.09 1"/>
    <material name="metal" rgba="0.9 0.9 0.9 1" specular="0.6" shininess="0.6"/>
    <material name="copper" rgba="0.72 0.45 0.2 1" specular="0.7" shininess="0.7"/>
    <material name="fur" rgba="0.55 0.4 0.25 1"/>
    <material name="wood" rgba="0.55 0.38 0.22 1"/>
    <material name="chassis" rgba="0.72 0.1 0.06 1" specular="0.5" shininess="0.6"/>
    <material name="deck" rgba="0.18 0.18 0.2 1" specular="0.4"/>
    <material name="head" rgba="0.85 0.85 0.82 1" specular="0.5"/>
    <material name="radar" rgba="0.95 0.95 0.95 1"/>
    <material name="track" rgba="0.06 0.06 0.06 1" specular="0.1"/>
    <material name="wheel" rgba="0.3 0.3 0.32 1" specular="0.4"/>
    <material name="lens" rgba="0.03 0.03 0.06 1" specular="1" shininess="1"/>
    <material name="beacon" rgba="1 0.6 0.05 1" emission="0.9"/>
    {mesh_assets}
  </asset>
  <worldbody>
    <light name="ceil0" type="spot" pos="{_f(W * 0.3, H / 2, 6)}" dir="0 0 -1" diffuse="0.38 0.38 0.4" cutoff="75" castshadow="true"/>
    <light name="ceil1" type="spot" pos="{_f(W * 0.75, H / 2, 6)}" dir="0 0 -1" diffuse="0.38 0.38 0.4" cutoff="75" castshadow="true"/>
    <geom name="floor" type="plane" size="{_f(W / 2 + 1, H / 2 + 1, 0.1)}" pos="{_f(W / 2, H / 2, 0)}" material="floor"/>
    {chr(10).join('    ' + w for w in walls).strip()}
    {_robot().replace('__START__', _f(sx, sy, 0.0))}
    {chr(10).join('    ' + t for t in things).strip()}
    {chr(10).join('    ' + p for p in props).strip()}
    {chr(10).join('    ' + s for s in smoke).strip()}
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
