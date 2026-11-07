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


def map_image(b: B.Brain) -> np.ndarray:
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
    img[g.occ == 1] = (225, 225, 225)
    img = np.clip(img, 0, 255).astype(np.uint8)[::-1]
    out = np.ascontiguousarray(cv2.resize(img, (g.cols * SCALE, g.rows * SCALE), interpolation=cv2.INTER_NEAREST))

    def px(xy):
        return int(xy[0] / B.RES * SCALE), int((g.rows - xy[1] / B.RES) * SCALE)
    for path, col, w in ((b.exit_path, (255, 255, 255), 1), (b.rescue[1] if b.rescue else None, (40, 255, 90), 2),
                         (b.path, (0, 220, 255), 2)):
        if path:
            pts = np.array([px(g.to_world(*c)) for c in path], np.int32)
            cv2.polylines(out, [pts], False, col, w, cv2.LINE_AA)
    for tr in b.tracker.tracks:
        if tr.kind == 'warm' and tr.state in ('pinned', 'verify'):
            cv2.circle(out, px(tr.xy), 7, (255, 40, 40) if tr.state == 'pinned' else (255, 200, 0), -1, cv2.LINE_AA)
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


class World:
    """Things the IC or the fire does to the world (main thread only)."""

    def __init__(self, m, d, L, s: sensors.Sensors):
        self.m, self.d, self.L, self.s = m, d, L, s
        self.lights = {i: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, f'firelight{i}')
                       for i, t in enumerate(L.things) if t.kind == 'fire'}
        self.base_diffuse = {i: m.light_diffuse[l].copy() for i, l in self.lights.items()}
        self.flames = [g for g in range(m.ngeom) if m.geom_matid[g] >= 0 and
                       (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[g]) or '').startswith('flame')]
        self.base_size = {g: m.geom_size[g].copy() for g in self.flames}
        self.char = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, 'char')
        self.smoke = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'smoke{k}') for k in range(scene.N_SMOKE)]
        self.rng = np.random.default_rng(L.seed)
        self.next_spread = SPREAD_EVERY

    def ignite(self, i: int, temp=450.0) -> str:
        t = self.L.things[i]
        if self.s.temp_by_thing[i] >= sensors.FIRE_MIN_C or t.kind == 'casualty':
            return ''
        self.s.ignite(i, temp)
        body = self.s.thing_body[i]
        for gid in np.where(self.m.geom_bodyid == body)[0]:
            if self.m.geom_group[gid] == 4:
                self.m.geom_group[gid] = 1  # reveal flames
            elif gid not in self.base_size:
                self.m.geom_matid[gid] = self.char
        return f'{t.name} ignited ({temp:.0f} C)'

    def animate(self, t: float):
        for i, lid in self.lights.items():
            self.m.light_diffuse[lid] = self.base_diffuse[i] * (0.75 + 0.25 * math.sin(t * 17 + i) * self.rng.random())
        for g in self.flames:
            k = 1 + 0.12 * math.sin(t * (9 + g % 5) + g) + 0.05 * self.rng.standard_normal()
            self.m.geom_size[g] = self.base_size[g] * np.array([k, k, 1 + 0.2 * (k - 1)])
        for k, b in enumerate(self.smoke):
            p = self.d.mocap_pos[self.m.body_mocapid[b]]
            p += np.array([0.12 * math.sin(t * 0.21 + k), 0.1 * math.cos(t * 0.17 + 2 * k), 0.0]) * FRAME_DT
            p[0] = min(max(p[0], 0.5), self.L.width - 0.5)
            p[1] = min(max(p[1], 0.5), self.L.height - 0.5)

    def spread(self) -> str:
        if self.d.time < self.next_spread:
            return ''
        self.next_spread += SPREAD_EVERY
        fires = [self.s._thing_xy(i) for i, T in self.s.temp_by_thing.items() if T >= sensors.FIRE_MIN_C]
        cand = [(min(math.dist(self.s._thing_xy(i), f) for f in fires), i) for i, t in enumerate(self.L.things)
                if self.s.temp_by_thing[i] < sensors.FIRE_MIN_C and t.kind != 'casualty' and t.style != 'pet'] \
            if fires else []
        cand = [c for c in cand if c[0] <= 1.5]
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
        col = (1, 0.1, 0.1, 0.9) if tr.state == 'pinned' else (1, 0.8, 0, 0.8)
        _geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.012, 0.6], [tr.xy[0], tr.xy[1], 0.6], col)
        _geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.07 + 0.03 * pulse] * 3, [tr.xy[0], tr.xy[1], 1.25], col)


def hud_texts(b: B.Brain, o, w: BrainWorker, truth: dict, msg: str):
    g = b.grid
    t = o.t
    pinned = [tr for tr in b.tracker.tracks if tr.state == 'pinned']
    possible = [tr for tr in b.tracker.tracks if tr.kind == 'warm' and tr.state == 'verify']
    first = next((c.t for c in b.cards if c.kind == 'PIN'), None)
    cov = 100 * (g.scanned & (g.occ == 0)).sum() / max(1, (g.occ == 0).sum())
    lat = np.array(w.lat) if w.lat else np.zeros(1)
    saved = (f'first casualty at T+{int(first) // 60}:{int(first) % 60:02d}, {CREW_ENTRY_S - first:.0f}s before crew entry'
             if first is not None and first < CREW_ENTRY_S else 'searching ahead of the crew')
    left = ('FIRESCOUT\nMission clock\nMode\nRadio\nRobot health\nBrain loop (on-device)\n'
            'Casualties\nFire seats\nFloor scanned\nTime saved')
    right = (f'autonomous fire scout\nT+{int(t) // 60:02d}:{int(t) % 60:02d}   crew enters in {max(0, CREW_ENTRY_S - t):.0f}s\n'
             f'{b.mode}\n{"LINKED" if o.comms else "LOST - working offline (" + str(len(b.outbox)) + " queued)"}\n'
             f'{o.self_temp_c:.0f} C / 60 limit   battery {o.battery:.0%}   CO {o.gas_ppm:.0f} ppm\n'
             f'p50 {np.percentile(lat, 50):.1f} ms   p95 {np.percentile(lat, 95):.1f} ms   (budget 50 ms)\n'
             f'{len(pinned)} CONFIRMED (thermal+radar)   {len(possible)} possible\n{len(b.fires)}\n{cov:.0f}%\n{saved}')
    cards = '\n'.join(str(c)[:118] for c in b.cards[-6:]) or 'starting mission...'
    eval_txt = (f'EVALUATION OVERLAY (robot never sees this)\ncasualties in building: {truth["n"]}   '
                f'correctly confirmed: {truth["ok"]}   false pins: {truth["fp"]}\n{msg}')
    return [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
            (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 'Decision log', cards),
            (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, eval_txt, '')]


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

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title('FireScout - IC control')
        self.root.configure(bg='#16181d')
        self.root.attributes('-topmost', True)
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f'+{sw - 560}+40')  # top-right corner, clear of the viewer HUD
        self.actions: list[str] = []
        self.root.protocol('WM_DELETE_WINDOW', lambda: self.actions.append('quit'))
        tk.Label(self.root, text='FireScout · Incident Commander', fg='#5ac8fa', bg='#16181d',
                 font=('Segoe UI', 13, 'bold')).grid(row=0, column=0, columnspan=2, pady=(10, 4))
        btns = [('Ignite shelf (unscripted block)', 'shelf'), ('Ignite random object', 'fire'),
                ('Toggle radio drop', 'comms'), ('METHANE report', 'methane'),
                ('Candidate IS a person', 'yes'), ('Candidate NOT a person', 'no'),
                ('Camera: overview', 'cam_over'), ('Camera: follow robot', 'cam_follow'),
                ('Camera: robot eye', 'cam_eye'), ('Toggle floor heatmap', 'heat')]
        for k, (label, act) in enumerate(btns):
            tk.Button(self.root, text=label, width=28, font=('Segoe UI', 10), bg='#2a2e37', fg='white',
                      activebackground='#3d4350', relief='flat',
                      command=lambda a=act: self.actions.append(a)).grid(row=1 + k // 2, column=k % 2, padx=6, pady=3)
        self.status = tk.Label(self.root, text='', fg='#e6e6e6', bg='#16181d', font=('Consolas', 10),
                               justify='left', anchor='w')
        self.status.grid(row=7, column=0, columnspan=2, sticky='we', padx=10, pady=(6, 10))

    def poll(self, msg: str, mode: str) -> list[str]:
        self.status.config(text=f'mode: {mode}' + chr(10) + msg[-70:])
        try:
            self.root.update()
        except self.tk.TclError:
            return ['quit']
        acts, self.actions = self.actions, []
        return acts


# ------------------------------------------------------------------ main


def main(seed: int = 11, use_rerun: bool = False, save: str | None = None, ignite_shelf_at: float | None = None):
    L = scene.generate_layout(seed)
    m, d = scene.load(L)
    w = BrainWorker(m, d, L, seed, use_rerun)
    if use_rerun:
        rerun_init(save)
    w.start()
    world = World(m, d, L, w.s)
    panel = ControlPanel()
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
            msg = world.spread() or msg
            if ignite_shelf_at is not None and d.time >= ignite_shelf_at:
                msg, ignite_shelf_at = world.ignite(shelf) or msg, None
            pip, texts = None, None
            with v.lock():
                world.animate(d.time)
                if w.obs is not None:
                    with w.lock:
                        draw_overlays(v.user_scn, w.b, d.time, show_heat)
                        texts = hud_texts(w.b, w.obs, w, truth_score(L, w.b), msg)
                        minimap = map_image(w.b)
                    pip = v.viewport
            # set_images/set_texts wait for the render thread: never call them while holding v.lock()
            if texts is not None and frames % 3 == 0:
                if pip is not None and w.thermal_rgb is not None:
                    th = cv2.resize(w.thermal_rgb, (384, 288))
                    mh = int(384 * minimap.shape[0] / minimap.shape[1])
                    mm = cv2.resize(minimap, (384, mh))
                    v.set_images([(mujoco.MjrRect(pip.width - 394, pip.height - 298, 384, 288), th),
                                  (mujoco.MjrRect(pip.width - 394, pip.height - 308 - mh, 384, mh), mm)])
                v.set_texts(texts)
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
    a = ap.parse_args()
    DEBUG = a.debug
    main(a.seed, a.rerun, a.save, a.ignite_shelf_at)
