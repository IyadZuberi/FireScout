"""FireScout live demo.

Windows:
  1. MuJoCo viewer  - the 3D world (drag any object with Ctrl+right-drag = unscripted change)
  2. Rerun          - Incident Commander dashboard (thermal, smoke camera, layered map, decision cards, METHANE)
  3. "IC control"   - button panel: ignite shelf/object, radio drop, METHANE, confirm/reject candidate
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import brain as B
import report
import scene
import sensors

DT_CTRL = 0.05
SPREAD_EVERY = 20.0
SCALE = 6  # map pixels per grid cell


# ------------------------------------------------------------------ dashboard

def blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin='thermal', name='Thermal camera (Lepton 160x120)'),
                rrb.Spatial2DView(origin='rgb', name='Normal camera in smoke'),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin='map', name='Robot map: walls · heat · gas · scanned · routes'),
                rrb.TimeSeriesView(origin='metrics', name='Robot health & loop latency'),
                row_shares=[3, 1],
            ),
            rrb.Vertical(
                rrb.TextDocumentView(origin='status', name='Status'),
                rrb.TextLogView(origin='cards', name='Decision cards (why it acted)'),
                rrb.TextDocumentView(origin='methane', name='METHANE report'),
                row_shares=[1, 3, 2],
            ),
            column_shares=[2, 3, 2],
        ),
        collapse_panels=True,
    )


def to_px(xy, g: B.Grid):
    return ((xy[0] / B.RES) * SCALE, (g.rows - xy[1] / B.RES) * SCALE)


def map_image(b: B.Brain) -> np.ndarray:
    g = b.grid
    img = np.zeros((g.rows, g.cols, 3), np.float32)
    img[:] = (18, 18, 24)
    img[g.occ == 0] = (58, 60, 68)
    img[(g.occ == 0) & g.scanned] = (62, 84, 110)
    heat = np.maximum(g.heat, cv2.GaussianBlur(g.heat, (0, 0), 2))
    a = np.clip((heat - 40) / 260, 0, 1)[..., None]
    img = img * (1 - a) + np.array([235, 60, 20]) * a
    gas = np.clip(g.gas / 1500, 0, 0.6)[..., None]
    img = img * (1 - gas) + np.array([230, 200, 40]) * gas
    img[g.occ == 1] = (225, 225, 225)
    img = np.clip(img, 0, 255).astype(np.uint8)[::-1]
    return cv2.resize(img, (g.cols * SCALE, g.rows * SCALE), interpolation=cv2.INTER_NEAREST)


def thermal_image(th: np.ndarray) -> np.ndarray:
    norm = np.clip((th - 15) / (120 - 15) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    return cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_CUBIC)


def blob_label(b: B.Brain, blob: B.Blob) -> tuple[str, tuple[int, int, int]]:
    if blob.kind == 'fire':
        return f'FIRE {blob.temp_c:.0f}°C', (255, 120, 20)
    tr = min(b.tracker.tracks, key=lambda t: math.dist(t.xy, blob.xy), default=None)
    if tr is None or math.dist(tr.xy, blob.xy) > 1.5:
        return f'warm {blob.temp_c:.0f}°C', (200, 200, 200)
    name = {'pinned': 'CASUALTY', 'rejected': 'not a person', 'verify': 'casualty?'}.get(tr.state, 'warm?')
    color = {'pinned': (255, 40, 40), 'rejected': (150, 150, 150), 'verify': (255, 200, 0)}.get(tr.state, (200, 200, 200))
    return f'#{tr.id} {name} {tr.conf:.2f}', color


def log_dashboard(b: B.Brain, o: sensors.Observation, lat: deque, last_card: int, extra_status: str) -> int:
    rr.set_time('sim', duration=o.t)
    g = b.grid
    rr.log('thermal', rr.Image(thermal_image(o.thermal)).compress(jpeg_quality=85))
    if b.last_blobs:
        boxes, labels, colors = [], [], []
        for bl in b.last_blobs:
            u, v, w, h = bl.box
            boxes.append([u * 4, v * 4, w * 4, h * 4])
            lab, col = blob_label(b, bl)
            labels.append(lab)
            colors.append(col)
        rr.log('thermal/blobs', rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYWH,
                                           labels=labels, colors=colors, show_labels=True))
    else:
        rr.log('thermal/blobs', rr.Clear(recursive=False))
    if o.rgb is not None:
        rr.log('rgb', rr.Image(cv2.resize(o.rgb, (640, 480), interpolation=cv2.INTER_CUBIC)).compress(jpeg_quality=80))

    rr.log('map', rr.Image(map_image(b)).compress(jpeg_quality=90))
    for name, path, color in (('path', b.path, (0, 220, 255)),
                              ('rescue', b.rescue[1] if b.rescue else None, (40, 255, 90)),
                              ('exit', b.exit_path, (255, 255, 255))):
        if path:
            pts = [to_px(g.to_world(*c), g) for c in path]
            rr.log(f'map/{name}', rr.LineStrips2D([pts], colors=[color], radii=1.5 if name == 'path' else 2.5,
                                                   labels=[name.upper()]))
        else:
            rr.log(f'map/{name}', rr.Clear(recursive=False))
    x, y, yaw = o.pose
    rr.log('map/robot', rr.Points2D([to_px((x, y), g)], radii=7, colors=[(60, 140, 255)], labels=['FireScout']))
    rr.log('map/heading', rr.Arrows2D(origins=[to_px((x, y), g)],
                                      vectors=[[math.cos(yaw) * 25, -math.sin(yaw) * 25]], colors=[(60, 140, 255)]))
    pins, plabels, pcolors = [], [], []
    for tr in b.tracker.tracks:
        if tr.kind != 'warm' or (tr.state == 'candidate' and tr.conf < 0.4):
            continue
        pins.append(to_px(tr.xy, g))
        plabels.append({'pinned': f'CASUALTY #{tr.id} {tr.conf:.2f}', 'rejected': f'#{tr.id} ✗',
                        }.get(tr.state, f'POSSIBLE #{tr.id} {tr.conf:.2f}'))
        pcolors.append({'pinned': (255, 40, 40), 'rejected': (120, 120, 120)}.get(tr.state, (255, 200, 0)))
    for f in b.fires:
        pins.append(to_px(f[:2], g))
        plabels.append(f'FIRE {f[2]:.0f}°C')
        pcolors.append((255, 120, 20))
    rr.log('map/pins', rr.Points2D(pins, radii=6, colors=pcolors, labels=plabels) if pins else rr.Clear(recursive=False))

    for c in b.cards[last_card:]:
        lvl = rr.TextLogLevel.WARN if c.kind in ('RETREAT', 'COMMS', 'HOLD') else \
            rr.TextLogLevel.ERROR if c.kind == 'PIN' else rr.TextLogLevel.INFO
        rr.log('cards', rr.TextLog(str(c), level=lvl))
    lat.append(b.timings.get('total', 0.0))
    rr.log('metrics/self_temp_C', rr.Scalars(o.self_temp_c))
    rr.log('metrics/CO_ppm_div10', rr.Scalars(o.gas_ppm / 10))
    rr.log('metrics/brain_loop_ms', rr.Scalars(b.timings.get('total', 0.0)))
    p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)
    rr.log('status', rr.TextDocument(
        f'**Mode:** `{b.mode}`  ·  **Radio:** {"🟢 linked" if o.comms else "🔴 LOST (logging offline, " + str(len(b.outbox)) + " queued)"}\n\n'
        f'**Self-temp:** {o.self_temp_c:.0f} °C / 60 limit  ·  **Battery:** {o.battery:.0%}  ·  **CO:** {o.gas_ppm:.0f} ppm\n\n'
        f'**Brain loop:** p50 {p50:.1f} ms · p95 {p95:.1f} ms (budget 50 ms, on-device)  ·  **Scanned:** '
        f'{100 * (g.scanned & (g.occ == 0)).sum() / max(1, (g.occ == 0).sum()):.0f}%\n\n{extra_status}',
        media_type=rr.MediaType.MARKDOWN))
    return len(b.cards)


def log_methane(b: B.Brain):
    rr.log('methane', rr.TextDocument('\n\n'.join(f'**{l[:2]}** {l[2:]}' for l in report.methane(b).splitlines()),
                                      media_type=rr.MediaType.MARKDOWN))


# ------------------------------------------------------------------ world events

class World:
    def __init__(self, m, d, L, s: sensors.Sensors):
        self.m, self.d, self.L, self.s = m, d, L, s
        self.lights = {i: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, f'firelight{i}')
                       for i, t in enumerate(L.things) if t.kind == 'fire'}
        self.base_diffuse = {i: m.light_diffuse[l].copy() for i, l in self.lights.items()}
        self.flame_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, 'flame0')
        self.rng = np.random.default_rng(L.seed)
        self.next_spread = SPREAD_EVERY

    def ignite(self, i: int, temp=450.0) -> str:
        if self.s.temp_by_thing[i] >= sensors.FIRE_MIN_C or self.L.things[i].kind == 'casualty':
            return ''
        self.s.ignite(i, temp)
        body = self.s.thing_body[i]
        for gid in np.where(self.m.geom_bodyid == body)[0]:
            self.m.geom_matid[gid] = self.flame_mat
        return f'{self.L.things[i].name} ignited ({temp:.0f}°C)'

    def tick(self):
        for i, lid in self.lights.items():
            self.m.light_diffuse[lid] = self.base_diffuse[i] * self.rng.uniform(0.6, 1.0)
        if self.d.time >= self.next_spread:  # fire spread to the nearest flammable object
            self.next_spread += SPREAD_EVERY
            fires = [self.s._thing_xy(i) for i, T in self.s.temp_by_thing.items() if T >= sensors.FIRE_MIN_C]
            cand = [(min(math.dist(self.s._thing_xy(i), f) for f in fires), i)
                    for i, t in enumerate(self.L.things)
                    if self.s.temp_by_thing[i] < sensors.FIRE_MIN_C and t.kind != 'casualty' and t.style != 'pet'] if fires else []
            cand = [c for c in cand if c[0] <= 1.5]
            if cand:
                return self.ignite(min(cand)[1])
        return ''


# ------------------------------------------------------------------ main

class ControlPanel:
    """Incident Commander control window (tkinter, stdlib): buttons queue actions for the sim loop."""

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title('FireScout - IC control')
        self.root.configure(bg='#16181d')
        self.actions: list[str] = []
        self.alive = True
        self.root.protocol('WM_DELETE_WINDOW', lambda: self.actions.append('quit'))
        tk.Label(self.root, text='FireScout · Incident Commander', fg='#5ac8fa', bg='#16181d',
                 font=('Segoe UI', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=(10, 4))
        btns = [('🔥 Ignite shelf (unscripted block)', 'shelf'), ('🔥 Ignite random object', 'fire'),
                ('📡 Toggle radio drop', 'comms'), ('📋 METHANE now', 'methane'),
                ('✅ Candidate IS a person', 'yes'), ('❌ Candidate NOT a person', 'no')]
        for k, (label, act) in enumerate(btns):
            tk.Button(self.root, text=label, width=30, font=('Segoe UI', 10), bg='#2a2e37', fg='white',
                      activebackground='#3d4350', relief='flat',
                      command=lambda a=act: self.actions.append(a)).grid(row=1 + k // 2, column=k % 2, padx=6, pady=4)
        self.status = tk.Label(self.root, text='', fg='#e6e6e6', bg='#16181d', font=('Consolas', 10),
                               justify='left', anchor='w')
        self.status.grid(row=4, column=0, columnspan=2, sticky='we', padx=10, pady=(6, 10))
        tk.Label(self.root, text='3D world: Ctrl+right-drag any object to move it', fg='#8a8f98', bg='#16181d',
                 font=('Segoe UI', 9)).grid(row=5, column=0, columnspan=2, pady=(0, 8))

    def poll(self, msg: str, mode: str, comms: bool) -> list[str]:
        radio = 'linked' if comms else 'LOST'
        self.status.config(text=f'mode: {mode:<12} radio: {radio}' + chr(10) + msg[-70:])
        try:
            self.root.update()
        except self.tk.TclError:
            return ['quit']
        acts, self.actions = self.actions, []
        return acts


def main(seed: int = 7, save: str | None = 'firescout_run.rrd', ignite_shelf_at: float | None = None):
    L = scene.generate_layout(seed)
    m, d = scene.load(L)
    s = sensors.Sensors(m, d, L, seed=seed)
    b = B.Brain(L.width, L.height, L.start)
    w = World(m, d, L, s)
    rr.init('FireScout', spawn=True, default_blueprint=blueprint())
    if save:
        rr.set_sinks(rr.GrpcSink(), rr.FileSink(save))
        rr.send_blueprint(blueprint())
    steps = int(round(DT_CTRL / m.opt.timestep))
    cmd = B.Command(0, 0, 0)
    lat: deque = deque(maxlen=400)
    last_card, tick, msg = 0, 0, 'mission started'
    shelf = next(i for i, t in enumerate(L.things) if t.kind == 'shelf')
    panel = ControlPanel()
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        with v.lock():
            v.cam.lookat[:] = [L.width / 2, L.height / 2, 0]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 12.5, -75, 90
            for f in ('mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX'):
                v.user_scn.flags[getattr(mujoco.mjtRndFlag, f)] = 1
        while v.is_running():
            wall0 = time.perf_counter()
            s.render_rgb = tick % 4 == 0
            o = s.observe(cmd.want_breath_xy)
            cmd = b.step(o)
            d.ctrl[:] = [cmd.vx, cmd.vy, cmd.wz]
            ev = w.tick()
            if ignite_shelf_at is not None and d.time >= ignite_shelf_at:
                ev, ignite_shelf_at = w.ignite(shelf), None
            msg = ev or msg
            for _ in range(steps):
                mujoco.mj_step(m, d)
            v.sync()
            last_card = log_dashboard(b, o, lat, last_card, f'_{msg}_')
            if tick % 40 == 0:
                log_methane(b)
            for act in panel.poll(msg, b.mode, o.comms):
                if act == 'fire':
                    choices = [i for i, t in enumerate(L.things)
                               if t.kind != 'casualty' and s.temp_by_thing[i] < sensors.FIRE_MIN_C]
                    msg = w.ignite(int(w.rng.choice(choices))) if choices else msg
                elif act == 'shelf':
                    msg = w.ignite(shelf) or msg
                elif act == 'comms':
                    s.force_comms_drop = not s.force_comms_drop
                    msg = 'radio dropped by IC' if s.force_comms_drop else 'radio restored'
                elif act in ('yes', 'no'):
                    cand = [t for t in b.tracker.tracks if t.kind == 'warm' and t.ic is None and t.state != 'rejected']
                    if cand:
                        tr = max(cand, key=lambda t: t.first_seen)
                        b.feedback(tr.id, act == 'yes')
                        msg = f'IC feedback on #{tr.id}'
                elif act == 'methane':
                    print(report.methane(b))
                    log_methane(b)
                elif act == 'quit':
                    v.close()
            tick += 1
            time.sleep(max(0.0, DT_CTRL - (time.perf_counter() - wall0)))  # real-time pacing
    panel.root.destroy()
    print(report.methane(b))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--ignite-shelf-at', type=float, default=None, help='sim seconds; for scripted rehearsals')
    a = ap.parse_args()
    main(a.seed, None if a.no_save else 'firescout_run.rrd', a.ignite_shelf_at)
