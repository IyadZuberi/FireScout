"""FireScout live demo - operator console.

Windows:
  1. MuJoCo viewer (main): the 3D care-home floor at 60 fps with a mission HUD, thermal + map
     picture-in-picture, the robot's path / rescue / exit routes, casualty markers and measured heat on the floor.
     Ctrl+right-drag any object = unscripted change.
  2. IC control panel (tkinter): ignite shelf / random object, radio drop, confirm / reject candidate, camera.
  3. Rerun (optional, --rerun): deep-dive dashboard + .rrd recording for replay.

Threads: physics + rendering on the main thread (smooth); sensors + brain at 10 Hz on a worker thread that renders
from a snapshot of the world, so a slow perception tick never stutters the 3D view.
"""
from __future__ import annotations

import argparse
import math
import threading
import time
from collections import deque

import cv2
import mujoco
import mujoco.viewer
import numpy as np

import audio
import brain as B
import report
import scene
import sensors

BRAIN_DT = 0.1      # perception + decision rate (10 Hz)
FRAME_DT = 1 / 60   # viewer rate
SPREAD_EVERY = 25.0
CREW_ENTRY_S = 150.0
SCALE = 6           # map pixels per grid cell
DEBUG = False

# ------------------------------------------------------------------ images


SEM_RGB = {1: (225, 225, 225), 2: (170, 120, 70)}  # wall white, furniture brown
PRIO_RGB = {'P1': (255, 40, 40), 'P2': (255, 170, 0), 'P?': (255, 230, 0)}


def map_image(b: B.Brain, rooms=None) -> np.ndarray:
    g = b.grid
    img = np.zeros((g.rows, g.cols, 3), np.float32)
    img[:] = (18, 18, 24)
    img[g.occ == 0] = (58, 60, 68)
    img[(g.occ == 0) & g.scanned] = (62, 84, 110)
    heat = np.maximum(g.heat, cv2.GaussianBlur(g.heat, (0, 0), 1))
    a = np.clip((heat - 32) / 200, 0, 1)[..., None]
    img = img * (1 - a) + np.array([235, 60, 20]) * a
    gas = np.clip(g.gas / 1500, 0, 0.5)[..., None]
    img = img * (1 - gas) + np.array([230, 200, 40]) * gas
    sem = b.semantic()
    for k, col in SEM_RGB.items():
        img[sem == k] = col
    img = np.clip(img, 0, 255).astype(np.uint8)[::-1]
    out = np.ascontiguousarray(cv2.resize(img, (g.cols * SCALE, g.rows * SCALE), interpolation=cv2.INTER_NEAREST))

    def px(xy):
        return int(xy[0] / B.RES * SCALE), int((g.rows - xy[1] / B.RES) * SCALE)
    for path, col, w in ((b.exit_path, (255, 255, 255), 1), (b.rescue[1] if b.rescue else None, (40, 255, 90), 2),
                         (b.path, (0, 220, 255), 2)):
        if path:
            pts = np.array([px(g.to_world(*c)) for c in path], np.int32)
            cv2.polylines(out, [pts], False, col, w, cv2.LINE_AA)
    for k, (x0, y0, x1, y1, kind) in enumerate(rooms or []):
        cv2.putText(out, f'R{k + 1}', px((x0 + 0.15, y1 - 0.45)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 170, 200), 1,
                    cv2.LINE_AA)
    if b.voice_hint and b.t - b.voice_hint[3] < 4.0:
        hx, hy, hb, _ = b.voice_hint
        cv2.arrowedLine(out, px((hx, hy)), px((hx + 4 * math.cos(hb), hy + 4 * math.sin(hb))), (255, 120, 255), 2,
                        cv2.LINE_AA, tipLength=0.08)
    for tr in b.tracker.tracks:
        if tr.kind == 'warm' and tr.state in ('pinned', 'verify'):
            lab = b.priority(tr)[0] if tr.state == 'pinned' else 'P?'
            cv2.circle(out, px(tr.xy), 7, PRIO_RGB[lab], -1, cv2.LINE_AA)
            cv2.putText(out, f'{lab} #{tr.id}', (px(tr.xy)[0] + 9, px(tr.xy)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        PRIO_RGB[lab], 1, cv2.LINE_AA)
    for f in b.fires:
        cv2.drawMarker(out, px(f[:2]), (255, 140, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)
    return out


def thermal_image(th: np.ndarray, blobs=(), labels=()) -> np.ndarray:
    norm = np.clip((th - 15) / (120 - 15) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    rgb = np.ascontiguousarray(cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_CUBIC))
    for bl, (lab, col) in zip(blobs, labels):
        u, v, w, h = (int(k * 4) for k in bl.box)
        cv2.rectangle(rgb, (u, v), (u + w, v + h), col, 2, cv2.LINE_AA)
        cv2.putText(rgb, lab, (u, max(14, v - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return rgb


def blob_label(b: B.Brain, blob: B.Blob) -> tuple[str, tuple[int, int, int]]:
    if blob.kind == 'fire':
        return f'FIRE {blob.temp_c:.0f}C', (255, 140, 30)
    tr = min(b.tracker.tracks, key=lambda t: math.dist(t.xy, blob.xy), default=None)
    if tr is None or math.dist(tr.xy, blob.xy) > 1.5:
        return f'{blob.temp_c:.0f}C', (210, 210, 210)
    name = {'pinned': 'CASUALTY', 'rejected': 'not a person', 'verify': 'casualty?'}.get(tr.state, 'warm?')
    col = {'pinned': (255, 50, 50), 'rejected': (150, 150, 150), 'verify': (255, 210, 0)}.get(tr.state, (210, 210, 210))
    return f'{name} {tr.conf:.2f}', col


# ------------------------------------------------------------------ world events


FIRE_GROW_S = 180.0    # time for a fire to reach full size (~1.8 m across, ~2x taller)
SMOKE_FILL_S = 240.0   # time for the smoke layer to come down from the ceiling to ~1 m


class World:
    """Things the IC or the fire does to the world (main thread only): fires grow and spread, smoke banks down."""

    def __init__(self, m, d, L, s: sensors.Sensors):
        self.m, self.d, self.L, self.s = m, d, L, s
        self.lights = {i: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, f'firelight{i}')
                       for i, t in enumerate(L.things) if t.kind == 'fire'}
        self.base_diffuse = {i: m.light_diffuse[l].copy() for i, l in self.lights.items()}
        self.flames = [g for g in range(m.ngeom) if m.geom_matid[g] >= 0 and
                       (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[g]) or '').startswith('flame')]
        self.base_size = {g: m.geom_size[g].copy() for g in self.flames}
        self.base_pos = {g: m.geom_pos[g].copy() for g in self.flames}
        self.flame_thing = {g: s.geom_thing[g] for g in self.flames}
        self.char = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, 'char')
        self.bed = {}  # fire thing -> its char/ember geoms (grow with the fire)
        for i, t in enumerate(L.things):
            if t.kind == 'fire':
                gs = [g for g in np.where(m.geom_bodyid == s.thing_body[i])[0] if g not in self.base_size]
                self.bed[i] = {int(g): m.geom_size[g].copy() for g in gs}
        self.base_temp = dict(s.temp_by_thing)
        self.t_lit = {i: 0.0 for i, t in enumerate(L.things) if t.kind == 'fire'}
        self.smoke = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'smoke{k}') for k in range(scene.N_SMOKE)]
        self.smoke_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, 'smoke')
        self.smoke_home = [self.d.mocap_pos[m.body_mocapid[b]].copy() for b in self.smoke]
        self.smoke_size = [m.geom_size[m.body_geomadr[b]].copy() for b in self.smoke]
        self.rng = np.random.default_rng(L.seed)
        self.next_spread = SPREAD_EVERY

    def growth(self, i: int) -> float:
        return min(1.0, max(0.0, (self.d.time - self.t_lit[i]) / FIRE_GROW_S)) if i in self.t_lit else 0.0

    def ignite(self, i: int, temp=450.0) -> str:
        t = self.L.things[i]
        if self.s.temp_by_thing[i] >= sensors.FIRE_MIN_C or t.kind == 'casualty':
            return ''
        self.s.ignite(i, temp)
        self.base_temp[i] = temp
        self.t_lit[i] = self.d.time
        body = self.s.thing_body[i]
        for gid in np.where(self.m.geom_bodyid == body)[0]:
            if self.m.geom_group[gid] == 4:
                self.m.geom_group[gid] = 1  # reveal flames
            elif gid not in self.base_size:
                self.m.geom_matid[gid] = self.char
        return f'{t.name} ignited ({temp:.0f} C)'

    def grow(self):
        """Fire development: hotter and wider over FIRE_GROW_S (heat the thermal camera, gas and O2 cells feel)."""
        for i in self.t_lit:
            if self.s.temp_by_thing[i] >= sensors.FIRE_MIN_C:
                self.s.temp_by_thing[i] = min(950.0, self.base_temp[i] + 350.0 * self.growth(i))
        for i, geoms in self.bed.items():
            k = 1 + 2.2 * self.growth(i)
            for g, base in geoms.items():
                self.m.geom_size[g] = base * np.array([k, k, 1.0])

    def animate(self, t: float):
        for i, lid in self.lights.items():
            self.m.light_diffuse[lid] = self.base_diffuse[i] * (1 + 1.2 * self.growth(i)) * \
                (0.7 + 0.3 * math.sin(t * 17 + i) * self.rng.random())
        for g in self.flames:  # tongues: spread out, stretch and flicker independently
            gr = self.growth(int(self.flame_thing[g]))
            spread = 1 + 3.5 * gr
            ph = g * 1.7
            fl = 0.72 + 0.18 * math.sin(t * (7 + g % 6) + ph) + 0.1 * math.sin(t * 23 + 2 * ph) \
                + 0.06 * self.rng.standard_normal()
            r, _, h = self.base_size[g]
            h2 = h * (1 + 1.1 * gr) * fl
            bp = self.base_pos[g]
            z0 = bp[2] - h * 0.8
            self.m.geom_size[g] = [r * (1 + 0.6 * gr), r * 0.8 * (1 + 0.6 * gr), max(0.05, h2)]
            self.m.geom_pos[g] = [bp[0] * spread + 0.03 * math.sin(t * 5 + ph), bp[1] * spread
                                  + 0.03 * math.cos(t * 6 + ph), z0 + h2 * 0.8]
        # smoke: hot layer under the ceiling that thickens, spreads and banks down over SMOKE_FILL_S
        age = max([t - t0 for i, t0 in self.t_lit.items() if self.s.temp_by_thing[i] >= sensors.FIRE_MIN_C] or [0.0])
        fill = min(1.0, age / SMOKE_FILL_S)
        self.m.mat_rgba[self.smoke_mat][3] = 0.05 + 0.17 * fill
        fires = [self.s._thing_xy(i) for i in self.t_lit if self.s.temp_by_thing[i] >= sensors.FIRE_MIN_C]
        for k, b in enumerate(self.smoke):
            home = self.smoke_home[k]
            f = fires[k % len(fires)] if fires else (self.L.width / 2, self.L.height / 2)
            reach = 0.35 + 1.4 * fill
            p = self.d.mocap_pos[self.m.body_mocapid[b]]
            p[0] = min(max(f[0] + (home[0] - f[0]) * reach + 0.25 * math.sin(t * 0.21 + k), 0.4), self.L.width - 0.4)
            p[1] = min(max(f[1] + (home[1] - f[1]) * reach + 0.2 * math.cos(t * 0.17 + 2 * k), 0.4),
                       self.L.height - 0.4)
            p[2] = max(1.0, home[2] - 1.1 * fill)
            self.m.geom_size[self.m.body_geomadr[b]] = self.smoke_size[k] * (1 + 0.5 * fill)

    def spread(self) -> str:
        if self.d.time < self.next_spread:
            return ''
        self.next_spread += SPREAD_EVERY
        fires = [self.s._thing_xy(i) for i, T in self.s.temp_by_thing.items() if T >= sensors.FIRE_MIN_C]
        cand = [(min(math.dist(self.s._thing_xy(i), f) for f in fires), i) for i, t in enumerate(self.L.things)
                if self.s.temp_by_thing[i] < sensors.FIRE_MIN_C and t.kind != 'casualty' and t.style != 'pet'] \
            if fires else []
        reach = 1.2 + 1.6 * max([self.growth(i) for i in self.t_lit] or [0.0])
        cand = [c for c in cand if c[0] <= reach]
        return self.ignite(min(cand)[1]) if cand else ''


# ------------------------------------------------------------------ brain worker


class BrainWorker(threading.Thread):
    """Brain + labelling + logging at 10 Hz off the render thread. Sensor rendering stays on the main thread
    (OpenGL/GLFW contexts must not be driven from worker threads on Windows)."""

    def __init__(self, m, d, L, seed, use_rerun):
        super().__init__(daemon=True)
        self.m, self.L, self.seed, self.use_rerun = m, L, seed, use_rerun
        self.s = sensors.Sensors(m, d, L, seed=seed)  # created on the main thread
        self.pending = None
        self.go = threading.Event()
        self.ready = threading.Event()
        self.ready.set()
        self.lock = threading.Lock()
        self.cmd = B.Command(0, 0, 0)
        self.obs = None
        self.thermal_rgb = None
        self.ticks = 0
        self.lat: deque = deque(maxlen=300)
        self.b = B.Brain(L.width, L.height, L.start)

    def kick(self):
        """Main thread: read the sensors now, hand the observation to the brain thread."""
        self.s.render_rgb = self.use_rerun and self.ticks % 2 == 0
        self.pending = self.s.observe(self.cmd.want_breath_xy)
        self.ready.clear()
        self.go.set()

    def run(self):
        while True:
            self.go.wait()
            self.go.clear()
            o = self.pending
            with self.lock:
                cmd = self.b.step(o)
                labels = [blob_label(self.b, bl) for bl in self.b.last_blobs]
            self.lat.append(self.b.timings['total'])
            self.thermal_rgb = thermal_image(o.thermal, self.b.last_blobs, labels)
            self.obs, self.cmd = o, cmd
            self.ticks += 1
            if self.use_rerun:
                with self.lock:
                    log_rerun(self.b, o, self.thermal_rgb)
            self.ready.set()


# ------------------------------------------------------------------ viewer overlays


def _geom(scn, gtype, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _line(scn, pts, rgba, width=0.03, z=0.04):
    for a, c in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.array([a[0], a[1], z]), np.array([c[0], c[1], z]))
        scn.ngeom += 1


def draw_overlays(scn, b: B.Brain, t: float, show_heat: bool):
    scn.ngeom = 0
    g = b.grid
    if show_heat:  # measured heat painted on the floor (the robot's own map)
        hot = np.argwhere(g.heat > 45)
        for i, j in hot[:: max(1, len(hot) // 500)]:
            a = min(1.0, (float(g.heat[i, j]) - 45) / 300)
            x, y = g.to_world(i, j)
            _geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [B.RES / 2, B.RES / 2, 0.004], [x, y, 0.006],
                  [1.0, 0.75 - 0.6 * a, 0.1, 0.18 + 0.35 * a])
    for path, rgba, w, z in ((b.exit_path, (1, 1, 1, 0.5), 0.02, 0.03),
                             (b.rescue[1] if b.rescue else None, (0.15, 1, 0.35, 0.85), 0.04, 0.05),
                             (b.path, (0, 0.85, 1, 0.9), 0.025, 0.07)):
        if path:
            _line(scn, [g.to_world(*c) for c in path[::2] + [path[-1]]], rgba, w, z)
    pulse = 0.5 + 0.5 * math.sin(t * 6)
    for tr in b.tracker.tracks:
        if tr.kind != 'warm' or tr.state not in ('pinned', 'verify'):
            continue
        lab = b.priority(tr)[0] if tr.state == 'pinned' else 'P?'
        col = [c / 255 for c in PRIO_RGB[lab]] + [0.9]
        _geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.012, 0.6], [tr.xy[0], tr.xy[1], 0.6], col)
        _geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.07 + 0.03 * pulse] * 3, [tr.xy[0], tr.xy[1], 1.25], col)
        scn.geoms[scn.ngeom - 1].label = f'{lab} #{tr.id}' + (' calling' if tr.responsive else '')
    if b.voice_hint and b.t - b.voice_hint[3] < 3.0:  # where the robot heard a voice from
        hx, hy, hb, _ = b.voice_hint
        _line(scn, [(hx + 0.4 * k * math.cos(hb), hy + 0.4 * k * math.sin(hb)) for k in range(1, 11, 2)
                    for _ in (0,)], (1, 0.45, 1, 0.55), 0.03, 0.5)


def _ascii(txt: str) -> str:
    """MuJoCo's overlay font is ASCII-only."""
    return txt.replace('°', '').replace('→', '->').replace('²', '2')


def hud_texts(b: B.Brain, o, w: BrainWorker, truth: dict, msg: str, building: str = ''):
    g = b.grid
    t = o.t
    pinned = sorted([tr for tr in b.tracker.tracks if tr.state == 'pinned'], key=lambda tr: -b.priority(tr)[1])
    possible = [tr for tr in b.tracker.tracks if tr.kind == 'warm' and tr.state == 'verify']
    first = next((c.t for c in b.cards if c.kind == 'PIN'), None)
    cov = 100 * (g.scanned & (g.occ == 0)).sum() / max(1, (g.occ == 0).sum())
    lat = np.array(w.lat) if w.lat else np.zeros(1)
    saved = (f'first casualty at T+{int(first) // 60}:{int(first) % 60:02d}, {CREW_ENTRY_S - first:.0f}s before crew entry'
             if first is not None and first < CREW_ENTRY_S else 'searching ahead of the crew')
    heard = b.voice_hint is not None and t - b.voice_hint[3] < 4.0
    cas = ', '.join(f'{b.priority(tr)[0]} #{tr.id}' for tr in pinned) or 'none yet'
    left = ('FIRESCOUT\nBuilding\nMission clock\nMode\nSpeed\nRadio\nRobot health\nAir at robot\nMicrophones\n'
            'Brain loop (on-device)\nConfirmed casualties\nPossible (crew check)\nFire seats\nFloor scanned\n'
            'Time saved\nIC event\nGround truth (eval only)')
    right = (f'autonomous fire scout\n{building}\n'
             f'T+{int(t) // 60:02d}:{int(t) % 60:02d}   crew enters in {max(0, CREW_ENTRY_S - t):.0f}s\n'
             f'{b.mode}{"   (IC override)" if b.ic_mode or b.ic_area else ""}\n'
             f'{b.speed:.2f} m/s   {b.speed_why}\n'
             f'{"LINKED" if o.comms else "LOST - working offline (" + str(len(b.outbox)) + " queued)"}\n'
             f'{o.self_temp_c:.0f} C / 60 limit   battery {o.battery:.0%}\n'
             f'CO {o.gas_ppm:.0f} ppm   O2 {o.o2_pct:.1f}%\n'
             f'{"VOICE HEARD - steering toward it" if heard else "listening"}\n'
             f'p50 {np.percentile(lat, 50):.1f} ms   p95 {np.percentile(lat, 95):.1f} ms   (budget 50 ms)\n'
             f'{len(pinned)}: {cas}\n{len(possible)}\n{len(b.fires)}\n{cov:.0f}%\n{saved}\n{msg[:70]}\n'
             f'{truth["n"]} in building, {truth["ok"]} confirmed correctly, {truth["fp"]} false')
    cards = '\n'.join(_ascii(str(c))[:100] for c in b.cards[-6:]) or 'starting mission...'
    return [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, _ascii(right)),
            (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'Decision log', cards)]


FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, CYAN = (235, 235, 235), (150, 156, 168), (90, 200, 250)


def _rows_image(rows, width=520, title='FIRESCOUT', sub_title='') -> np.ndarray:
    """rows: [(label, value, rgb)] -> dark translucent-looking panel."""
    h = 52 + 21 * len(rows)
    img = np.full((h, width, 3), (20, 22, 28), np.uint8)
    cv2.putText(img, title, (12, 27), FONT, 0.72, CYAN, 2, cv2.LINE_AA)
    cv2.putText(img, sub_title, (150, 27), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    for k, (lab, val, col) in enumerate(rows):
        y = 52 + 21 * k
        cv2.putText(img, lab, (12, y), FONT, 0.43, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, _ascii(str(val))[:52], (168, y), FONT, 0.45, col, 1, cv2.LINE_AA)
    return img


def hud_image(b: B.Brain, o, w, truth: dict, msg: str, building: str = '') -> np.ndarray:
    g = b.grid
    t = o.t
    pinned = sorted([tr for tr in b.tracker.tracks if tr.state == 'pinned'], key=lambda tr: -b.priority(tr)[1])
    possible = [tr for tr in b.tracker.tracks if tr.kind == 'warm' and tr.state == 'verify']
    first = next((c.t for c in b.cards if c.kind == 'PIN'), None)
    cov = 100 * (g.scanned & (g.occ == 0)).sum() / max(1, (g.occ == 0).sum())
    lat = np.array(w.lat) if w.lat else np.zeros(1)
    heard = b.voice_hint is not None and t - b.voice_hint[3] < 4.0
    red, amber, green = (255, 80, 70), (255, 190, 40), (90, 230, 120)
    rows = [
        ('Building', building, WHITE),
        ('Mission clock', f'T+{int(t) // 60:02d}:{int(t) % 60:02d}   crew enters in {max(0, CREW_ENTRY_S - t):.0f}s',
         WHITE),
        ('Mode', b.mode + ('  (IC override)' if b.ic_mode or b.ic_area else ''),
         amber if b.mode in ('RETREAT', 'IC_STOP', 'RECALL', 'HOLD') else green),
        ('Speed', f'{b.speed:.2f} m/s  {b.speed_why}', WHITE),
        ('Radio', 'LINKED' if o.comms else f'LOST - offline, {len(b.outbox)} queued', green if o.comms else red),
        ('Robot health', f'{o.self_temp_c:.0f} C / 60 limit   battery {o.battery:.0%}',
         red if o.self_temp_c > 50 else WHITE),
        ('Air at robot', f'CO {o.gas_ppm:.0f} ppm   O2 {o.o2_pct:.1f}%',
         red if o.gas_ppm > 400 or o.o2_pct < 17 else WHITE),
        ('Microphones', 'VOICE HEARD - steering to it' if heard else 'listening', (255, 120, 255) if heard else WHITE),
        ('Brain loop', f'p50 {np.percentile(lat, 50):.1f} ms  p95 {np.percentile(lat, 95):.1f} ms  (on-device)',
         WHITE),
        ('Fire seats', f'{len(b.fires)} located', amber if b.fires else WHITE),
        ('Floor scanned', f'{cov:.0f}%  (never declared empty)', WHITE),
        ('First casualty', 'searching' if first is None else
         f'T+{int(first) // 60}:{int(first) % 60:02d}  ({CREW_ENTRY_S - first:.0f}s before crew)', green),
        ('Confirmed', f'{len(pinned)}  (thermal + radar breathing)', red if pinned else WHITE),
    ]
    for tr in pinned[:4]:
        lab, _, why = b.priority(tr)
        rows.append((f'   {lab}  #{tr.id}', why, PRIO_RGB[lab]))
    rows += [('Possible', f'{len(possible)}  (crew must check)', amber if possible else WHITE),
             ('IC event', msg[-52:], CYAN),
             ('Eval only', f'{truth["n"]} in building / {truth["ok"]} found / {truth["fp"]} false', GREY)]
    return _rows_image(rows, sub_title='autonomous fire scout - live')


def log_image(b: B.Brain, width=900) -> np.ndarray:
    cards = b.cards[-7:]
    img = np.full((40 + 20 * max(1, len(cards)), width, 3), (20, 22, 28), np.uint8)
    cv2.putText(img, 'DECISION LOG  (every action + the sensor reading that caused it)', (12, 24), FONT, 0.5, CYAN,
                1, cv2.LINE_AA)
    col = {'RETREAT': (255, 90, 70), 'PIN': (255, 80, 80), 'PRIORITY': (255, 120, 90), 'REPLAN': (255, 190, 40),
           'HEAR': (255, 120, 255), 'IC': (90, 200, 250), 'COMMS': (255, 190, 40), 'HOLD': (255, 190, 40)}
    for k, c in enumerate(cards):
        cv2.putText(img, _ascii(str(c))[:118], (12, 46 + 20 * k), FONT, 0.42, col.get(c.kind, WHITE), 1, cv2.LINE_AA)
    return img


def truth_score(L, b) -> dict:
    cas = [t.xy for t in L.things if t.kind == 'casualty']
    pins = [tr for tr in b.tracker.tracks if tr.state == 'pinned']
    ok = sum(any(math.dist(tr.xy, c) <= 1.2 for tr in pins) for c in cas)
    fp = sum(not any(math.dist(tr.xy, c) <= 1.2 for c in cas) for tr in pins)
    return {'n': len(cas), 'ok': ok, 'fp': fp}


# ------------------------------------------------------------------ rerun (optional deep-dive)

_rr_card = 0


def log_rerun(b: B.Brain, o, thermal_rgb):
    global _rr_card
    import rerun as rr
    rr.set_time('sim', duration=o.t)
    rr.log('thermal', rr.Image(thermal_rgb).compress(jpeg_quality=85))
    if o.rgb is not None:
        rr.log('rgb', rr.Image(cv2.resize(o.rgb, (640, 480))).compress(jpeg_quality=80))
    rr.log('map', rr.Image(map_image(b)).compress(jpeg_quality=90))
    for c in b.cards[_rr_card:]:
        rr.log('cards', rr.TextLog(str(c), level=rr.TextLogLevel.WARN if c.kind in ('RETREAT', 'COMMS', 'HOLD')
                                   else rr.TextLogLevel.INFO))
    _rr_card = len(b.cards)
    rr.log('metrics/self_temp_C', rr.Scalars(o.self_temp_c))
    rr.log('metrics/CO_ppm_div10', rr.Scalars(o.gas_ppm / 10))
    rr.log('metrics/brain_loop_ms', rr.Scalars(b.timings.get('total', 0.0)))
    if int(o.t * 10) % 20 == 0:
        rr.log('methane', rr.TextDocument('\n\n'.join(f'**{l[:2]}** {l[2:]}' for l in report.methane(b).splitlines()),
                                          media_type=rr.MediaType.MARKDOWN))


def rerun_init(save):
    import rerun as rr
    import rerun.blueprint as rrb
    bp = rrb.Blueprint(rrb.Horizontal(
        rrb.Vertical(rrb.Spatial2DView(origin='thermal', name='Thermal camera'),
                     rrb.Spatial2DView(origin='rgb', name='Normal camera in smoke')),
        rrb.Vertical(rrb.Spatial2DView(origin='map', name='Robot map'),
                     rrb.TimeSeriesView(origin='metrics', name='Health & latency'), row_shares=[3, 1]),
        rrb.Vertical(rrb.TextLogView(origin='cards', name='Decision cards'),
                     rrb.TextDocumentView(origin='methane', name='METHANE')),
        column_shares=[2, 3, 2]), collapse_panels=True)
    rr.init('FireScout', spawn=True, default_blueprint=bp)
    if save:
        rr.set_sinks(rr.GrpcSink(), rr.FileSink(save))
        rr.send_blueprint(bp)


# ------------------------------------------------------------------ control panel


class ControlPanel:
    """Incident Commander control window (tkinter, stdlib): buttons queue actions for the sim loop."""

    def __init__(self, rooms=()):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title('FireScout - IC control')
        self.root.configure(bg='#16181d')
        self.root.attributes('-topmost', True)

        self.actions: list[str] = []
        self.root.protocol('WM_DELETE_WINDOW', lambda: self.actions.append('quit'))
        tk.Label(self.root, text='FireScout · Incident Commander', fg='#5ac8fa', bg='#16181d',
                 font=('Segoe UI', 13, 'bold')).grid(row=0, column=0, columnspan=2, pady=(10, 4))
        btns = [('Ignite shelf (unscripted block)', 'shelf'), ('Ignite random object', 'fire'),
                ('Toggle radio drop', 'comms'), ('METHANE report', 'methane'),
                ('Candidate IS a person', 'yes'), ('Candidate NOT a person', 'no'),
                ('Camera: overview', 'cam_over'), ('Camera: follow robot', 'cam_follow'),
                ('Camera: robot eye', 'cam_eye'), ('Toggle floor heatmap', 'heat'),
                ('IC: STOP robot', 'ic_stop'), ('IC: RESUME autonomy', 'ic_resume'),
                ('IC: RECALL to entry', 'ic_recall'), ('Sound on / off', 'audio')]
        for k, (label, act) in enumerate(btns):
            tk.Button(self.root, text=label, width=28, font=('Segoe UI', 10), bg='#2a2e37', fg='white',
                      activebackground='#3d4350', relief='flat',
                      command=lambda a=act: self.actions.append(a)).grid(row=1 + k // 2, column=k % 2, padx=6, pady=3)
        r0 = 1 + (len(btns) + 1) // 2
        tk.Label(self.root, text='IC: search this room first', fg='#9aa4b2', bg='#16181d',
                 font=('Segoe UI', 9)).grid(row=r0, column=0, columnspan=2, pady=(8, 2))
        for k, (x0, y0, x1, y1, kind) in enumerate(rooms):
            tk.Button(self.root, text=f'R{k + 1} {kind}', width=28, font=('Segoe UI', 9), bg='#243042', fg='white',
                      relief='flat', command=lambda a=f'room{k}': self.actions.append(a)
                      ).grid(row=r0 + 1 + k // 2, column=k % 2, padx=6, pady=2)
        self.status = tk.Label(self.root, text='', fg='#e6e6e6', bg='#16181d', font=('Consolas', 10),
                               justify='left', anchor='w')
        self.status.grid(row=r0 + 2 + len(rooms) // 2, column=0, columnspan=2, sticky='we', padx=10, pady=(6, 10))
        self.root.update_idletasks()  # top-right corner, fully on screen at any DPI
        self.root.geometry(f'+{self.root.winfo_screenwidth() - self.root.winfo_reqwidth() - 12}+40')

    def poll(self, msg: str, mode: str) -> list[str]:
        self.status.config(text=f'mode: {mode}' + chr(10) + msg[-70:])
        try:
            self.root.update()
        except self.tk.TclError:
            return ['quit']
        acts, self.actions = self.actions, []
        return acts


def launcher(preset='care_home', casualties=2, fires=1, seed=11):
    """Scenario picker shown before the run: building, casualties, fires, layout seed."""
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title('FireScout - choose scenario')
    root.configure(bg='#16181d', padx=18, pady=14)
    names = {v['name']: k for k, v in scene.PRESETS.items()}
    vb = tk.StringVar(value=scene.PRESETS[preset]['name'])
    vc, vf, vs = tk.IntVar(value=casualties), tk.IntVar(value=fires), tk.IntVar(value=seed)
    tk.Label(root, text='FireScout scenario', fg='#5ac8fa', bg='#16181d', font=('Segoe UI', 14, 'bold')
             ).grid(row=0, column=0, columnspan=2, pady=(0, 10))
    rows = [('Building', ttk.Combobox(root, textvariable=vb, values=list(names), state='readonly', width=26)),
            ('Casualties (0-4)', tk.Spinbox(root, from_=0, to=4, textvariable=vc, width=6)),
            ('Fires (1-3)', tk.Spinbox(root, from_=1, to=3, textvariable=vf, width=6)),
            ('Layout seed', tk.Spinbox(root, from_=0, to=9999, textvariable=vs, width=8))]
    for k, (lab, wdg) in enumerate(rows):
        tk.Label(root, text=lab, fg='#e6e6e6', bg='#16181d', font=('Segoe UI', 10)).grid(row=k + 1, column=0,
                                                                                         sticky='w', pady=4)
        wdg.grid(row=k + 1, column=1, sticky='w', pady=4)
    out = {}

    def go():
        out.update(preset=names[vb.get()], casualties=int(vc.get()), fires=int(vf.get()), seed=int(vs.get()))
        root.destroy()
    tk.Button(root, text='Start mission', command=go, bg='#2f7d32', fg='white', relief='flat', width=20,
              font=('Segoe UI', 11, 'bold')).grid(row=6, column=0, columnspan=2, pady=(12, 0))
    root.mainloop()
    return out or None


# ------------------------------------------------------------------ main


def main(seed: int = 11, use_rerun: bool = False, save: str | None = None, ignite_shelf_at: float | None = None,
         preset: str = 'care_home', casualties: int | None = None, fires: int | None = None, sound: bool = True,
         quit_after: float | None = None):
    L = scene.generate_layout(seed, preset, casualties, fires)
    rooms = [(*r, k) for r, k in zip(L.rooms, L.room_kinds)]
    building = (f'{scene.PRESETS[preset]["name"]}, {L.width:.0f} x {L.height:.0f} m, {len(L.rooms)} rooms '
                f'(seed {seed})')
    player = audio.Player(sound)
    heard_window: dict[int, int] = {}
    last_breath = -99.0
    m, d = scene.load(L)
    w = BrainWorker(m, d, L, seed, use_rerun)
    if use_rerun:
        rerun_init(save)
    w.start()
    world = World(m, d, L, w.s)
    panel = ControlPanel(rooms)
    shelf = next(i for i, t in enumerate(L.things) if t.kind == 'shelf')
    scout = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'scout')
    eye = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'thermal')
    next_brain, msg, show_heat = 0.0, 'mission started - robot placed at the entry door', True
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        with v.lock():
            v.cam.lookat[:] = [L.width / 2, L.height / 2 - 0.4, 0]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 12.0, -72, 90
            for f in ('mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX'):
                v.user_scn.flags[getattr(mujoco.mjtRndFlag, f)] = 1
            v.opt.geomgroup[scene.MESH_GROUP] = 1  # show detailed furniture meshes
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0  # cleaner view; radar shown on the map
        wall0, sim0 = time.perf_counter(), d.time
        frames = 0
        while v.is_running():
            frame0 = time.perf_counter()
            tm = [frame0]
            if d.time >= next_brain and w.ready.is_set():  # brain tick on the worker at 10 Hz sim time
                if w.obs is not None:
                    d.ctrl[:] = [w.cmd.vx, w.cmd.vy, w.cmd.wz]
                w.kick()
                next_brain = d.time + BRAIN_DT
            target = sim0 + (time.perf_counter() - wall0)  # physics in real time (waits for a slow brain)
            while d.time < target and d.time < next_brain + BRAIN_DT:
                mujoco.mj_step(m, d)
            if d.time < target - 0.5:  # brain fell behind: don't fast-forward later
                wall0, sim0 = time.perf_counter(), d.time
            tm.append(time.perf_counter())
            world.grow()
            msg = world.spread() or msg
            if ignite_shelf_at is not None and d.time >= ignite_shelf_at:
                msg, ignite_shelf_at = world.ignite(shelf) or msg, None
            pip, texts, vstate = None, None, None
            with v.lock():
                world.animate(d.time)
                if w.obs is not None:
                    with w.lock:
                        draw_overlays(v.user_scn, w.b, d.time, show_heat)
                        hud = hud_image(w.b, w.obs, w, truth_score(L, w.b), msg, building)
                        logi = log_image(w.b)
                        texts = True
                        minimap = map_image(w.b, rooms)
                        vstate = (w.b.mode, w.obs.breath, w.cmd.want_breath_xy)
                    pip = v.viewport
            # set_images/set_texts wait for the render thread: never call them while holding v.lock()
            if texts is not None and frames % 3 == 0 and pip is not None:
                sc = pip.width / 1500  # panels scale with the window (high-DPI safe)
                ims = []
                hw = int(520 * sc)
                hh = int(hud.shape[0] * hw / hud.shape[1])
                ims.append((mujoco.MjrRect(8, pip.height - hh - 8, hw, hh), cv2.resize(hud, (hw, hh))))
                lw = int(900 * sc)
                lh = int(logi.shape[0] * lw / logi.shape[1])
                ims.append((mujoco.MjrRect(8, 8, lw, lh), cv2.resize(logi, (lw, lh))))
                if w.thermal_rgb is not None:
                    tw = int(440 * sc)
                    th_h = int(tw * 0.75)
                    mh = int(tw * minimap.shape[0] / minimap.shape[1])
                    ims.append((mujoco.MjrRect(pip.width - tw - 8, 8, tw, th_h), cv2.resize(w.thermal_rgb, (tw, th_h))))
                    ims.append((mujoco.MjrRect(pip.width - tw - 8, 16 + th_h, tw, mh), cv2.resize(minimap, (tw, mh))))
                v.set_images(ims)
                v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                              'Ctrl+right-drag any object', 'unscripted change')])  # images only draw with a text set
            if w.obs is not None and player.on:  # what the robot's mics hear, on the laptop speakers
                for i, th in enumerate(L.things):
                    if th.kind != 'casualty' or not th.responsive or (d.time + 2.3 * i) % 7.0 >= 2.0:
                        continue
                    win = int((d.time + 2.3 * i) // 7.0)
                    if heard_window.get(i) != win and w.obs.voices:
                        heard_window[i] = win
                        sex = th.style[0] if th.style else 'f'
                        player.play(f'{"help" if win % 2 else "here"}_{sex}', max(db for _, db in w.obs.voices))
                if texts is not None and vstate[0] == 'VERIFY' and (vstate[1] or 0) > 0.5 and vstate[2] \
                        and d.time - last_breath > 6.5:
                    dist_c = math.dist(w.s._robot_xy(), vstate[2])
                    if player.play('breath', 60 - 8 * dist_c, 6.4):
                        last_breath = d.time
            tm.append(time.perf_counter())
            v.sync()
            tm.append(time.perf_counter())
            frames += 1
            if DEBUG and frames % 60 == 0:
                print(f'frame {frames} sim {d.time:.2f} ticks {w.ticks} ms: kick+phys '
                      f'{(tm[1]-tm[0])*1e3:.0f} overlays {(tm[2]-tm[1])*1e3:.0f} sync {(tm[3]-tm[2])*1e3:.0f}', flush=True)
            for act in panel.poll(msg, w.b.mode):
                if act == 'fire':
                    choices = [i for i, t in enumerate(L.things)
                               if t.kind != 'casualty' and w.s.temp_by_thing[i] < sensors.FIRE_MIN_C]
                    msg = world.ignite(int(world.rng.choice(choices))) if choices else msg
                elif act == 'shelf':
                    msg = world.ignite(shelf) or msg
                elif act == 'comms':
                    w.s.force_comms_drop = not w.s.force_comms_drop
                    msg = 'IC: radio dropped' if w.s.force_comms_drop else 'IC: radio restored'
                elif act in ('yes', 'no'):
                    with w.lock:
                        cand = [t for t in w.b.tracker.tracks if t.kind == 'warm' and t.ic is None
                                and t.state in ('verify', 'candidate') and t.conf >= 0.4]
                        if cand:
                            tr = max(cand, key=lambda t: t.first_seen)
                            w.b.feedback(tr.id, act == 'yes')
                            msg = f'IC feedback on #{tr.id}: {"person" if act == "yes" else "not a person"}'
                elif act == 'methane':
                    with w.lock:
                        print(report.methane(w.b))
                elif act == 'heat':
                    show_heat = not show_heat
                elif act in ('ic_stop', 'ic_resume', 'ic_recall'):
                    with w.lock:
                        w.b.ic(act[3:])
                    msg = f'IC: {act[3:].upper()} sent to robot'
                elif act.startswith('room'):
                    x0, y0, x1, y1, kind = rooms[int(act[4:])]
                    with w.lock:
                        w.b.ic('SEARCH', area=(x0, y0, x1, y1))
                    msg = f'IC: search R{int(act[4:]) + 1} ({kind}) first'
                elif act == 'audio':
                    player.on = not player.on and player.ws is not None
                    if not player.on:
                        player.stop()
                    msg = f'sound {"on" if player.on else "off"}'
                elif act.startswith('cam_'):
                    with v.lock():
                        if act == 'cam_over':
                            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                            v.cam.lookat[:] = [L.width / 2, L.height / 2 - 0.4, 0]
                            v.cam.distance, v.cam.elevation, v.cam.azimuth = 12.0, -72, 90
                        elif act == 'cam_follow':
                            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                            v.cam.trackbodyid = scout
                            v.cam.distance, v.cam.elevation = 3.2, -38
                        else:
                            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            v.cam.fixedcamid = eye
                elif act == 'quit':
                    v.close()
            if quit_after is not None and d.time >= quit_after:
                v.close()
            time.sleep(max(0.0, FRAME_DT - (time.perf_counter() - frame0)))
    panel.root.destroy()
    print(report.methane(w.b))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--rerun', action='store_true', help='also open the Rerun deep-dive dashboard')
    ap.add_argument('--save', default=None, help='with --rerun: also record to this .rrd file')
    ap.add_argument('--ignite-shelf-at', type=float, default=None, help='sim seconds; for scripted rehearsals')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--preset', choices=list(scene.PRESETS), default=None, help='building; omit to get the picker')
    ap.add_argument('--casualties', type=int, default=None)
    ap.add_argument('--fires', type=int, default=None)
    ap.add_argument('--no-sound', action='store_true')
    ap.add_argument('--quit-after', type=float, default=None, help='sim seconds, then print METHANE and exit')
    a = ap.parse_args()
    DEBUG = a.debug
    cfg = dict(preset=a.preset, casualties=a.casualties, fires=a.fires, seed=a.seed)
    if a.preset is None:
        cfg = launcher(seed=a.seed)
        if cfg is None:
            raise SystemExit(0)
    main(cfg['seed'], a.rerun, a.save, a.ignite_shelf_at, cfg['preset'], cfg['casualties'], cfg['fires'],
         not a.no_sound, a.quit_after)
