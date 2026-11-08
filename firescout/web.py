"""FireScout IC Console - browser front end for the live MuJoCo simulation.

MuJoCo stays the simulator: physics, every sensor and the agent run here, headless, in real time.
The browser (web/index.html, three.js) only DRAWS the MuJoCo state it is sent and sends Incident Commander
commands back - nothing in the browser decides anything.

  python web.py              then open http://localhost:8765
Endpoints: /api/scene (static layout), /api/state (SSE, 15 Hz), /feed/<thermal|night|colour|map>.jpg,
POST /api/cmd {"action": ...}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import mujoco
import numpy as np

import brain as B
import eval as E
import report
import run as R
import scene
import sensors

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, 'web')
STATE_HZ = 10
FEED_HZ = 4


def _r(v, n=2):
    return round(float(v), n)


class Sim:
    """One mission: MuJoCo world + sensors + agent + world events. Main thread only (OpenGL)."""

    def __init__(self, preset='care_home', casualties=None, fires=None, seed=11):
        self.cfg = dict(preset=preset, casualties=casualties, fires=fires, seed=seed)
        L = self.L = scene.generate_layout(seed, preset, casualties, fires)
        self.m, self.d = scene.load(L)
        self.w = R.BrainWorker(self.m, self.d, L, seed, use_rerun=False)
        self.w.start()
        self.world = R.World(self.m, self.d, L, self.w.s)
        self.rooms = [(*r, k) for r, k in zip(L.rooms, L.room_kinds)]
        self.shelf = next(i for i, t in enumerate(L.things) if t.kind == 'shelf')
        self.next_brain = 0.0
        self.msg = 'Mission started - robot placed at the entry door'
        self.events: list[dict] = []          # sounds etc. for the browser to play
        self.truth_occ = E.true_grid(L)
        dist = cv2.distanceTransform((self.truth_occ == 0).astype(np.uint8), cv2.DIST_L2, 3) * B.RES
        self.truth_dist = dist                 # metres from each cell to the nearest true wall
        self.feed_r = mujoco.Renderer(self.m, 300, 400)
        self.feed_cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, 'thermal')
        self.feeds: dict[str, bytes] = {}
        self.wall0, self.sim0 = time.perf_counter(), self.d.time
        self.heard_window: dict[int, int] = {}
        self._slow, self._slow_t = None, -1.0
        h, w_ = 300, 400
        yy, xx = np.mgrid[0:h, 0:w_]
        self._vig = np.clip(1 - 0.55 * (((xx - w_ / 2) / (w_ / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2), 0.25, 1)
        self._rng = np.random.default_rng(seed)
        self.said: list[tuple[float, str, str]] = []  # (t, kind, text) robot speaker announcements

    # ---------------------------------------------------------------- loop
    def tick(self):
        d, w = self.d, self.w
        if d.time >= self.next_brain and w.ready.is_set():
            if w.obs is not None:
                d.ctrl[:] = [w.cmd.vx, w.cmd.vy, w.cmd.wz]
            w.kick()
            self.next_brain = d.time + R.BRAIN_DT
        target = self.sim0 + (time.perf_counter() - self.wall0)
        while d.time < target and d.time < self.next_brain + R.BRAIN_DT:
            mujoco.mj_step(self.m, d)
        if d.time < target - 0.5:
            self.wall0, self.sim0 = time.perf_counter(), d.time
        self.world.grow()
        self.msg = self.world.spread() or self.msg
        self.world.animate(d.time)
        self._sounds()
        self._speaker()

    def _speaker(self):
        """Deliver the agent's loudspeaker instructions into the world (the person hears and acts)."""
        with self.w.lock:
            says, self.w.b.say_out = self.w.b.say_out, []
        for sy in says:
            heard = self.world.speak(sy, self.w.s._robot_xy())
            self.said.append((self.d.time, sy['kind'], sy['text']))
            self.events.append({'sound': 'robot_evac' if sy['kind'] == 'EVACUATE' else 'robot_shelter', 'db': 70})
            who = f'occupant reacts' if heard is not None else 'no reaction seen'
            self.msg = f'ROBOT SPEAKER to #{sy["track"]}: {"walk out" if sy["kind"] == "EVACUATE" else "stay put"} ({who})'

    def _sounds(self):
        """Voices the robot's microphones hear (played in the browser, louder when nearer)."""
        o = self.w.obs
        if o is None or not o.voices:
            return
        for i, th in enumerate(self.L.things):
            if th.kind != 'casualty' or not th.responsive or (self.d.time + 2.3 * i) % 7.0 >= 2.0:
                continue
            win = int((self.d.time + 2.3 * i) // 7.0)
            if self.heard_window.get(i) != win:
                self.heard_window[i] = win
                sex = th.style[0] if th.style else 'f'
                db = max(db for _, db in o.voices)
                self.events.append({'sound': f'{"help" if win % 2 else "here"}_{sex}', 'db': _r(db, 0)})

    def room_of(self, xy) -> str:
        for k, (x0, y0, x1, y1, kind) in enumerate(self.rooms):
            if x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1:
                return f'R{k + 1} {kind}'
        return 'corridor'

    def now_sentence(self, b, o) -> tuple[str, str]:
        """(what, why) the agent is doing right now, in plain English, from its own state."""
        g = b.grid
        if b.mode == 'IC_STOP':
            return 'Stopped by the Incident Commander', 'sensors and reporting keep running'
        if b.mode == 'RECALL':
            return 'Returning to the entry', 'Incident Commander ordered a recall'
        if b.mode == 'RETREAT':
            return 'Backing away from heat', f'robot body at {o.self_temp_c:.0f} C (limit 60 C) or a no-go cell nearby'
        if b.mode == 'VERIFY' and b.verify_id:
            tr = b.track(b.verify_id)
            if tr is not None:
                return (f'Checking possible person #{tr.id} in {self.room_of(tr.xy)}',
                        f'thermal says body-like ({tr.conf:.0%}); moving for a clear radar view to detect breathing')
        if b.mode == 'RETURN_COMMS':
            return 'Driving back into radio range', 'radio lost: uploading queued findings'
        if b.mode == 'HOLD':
            return 'Holding position', 'nowhere left it can reach; a human search is still required'
        if b.voice_hint and b.t - b.voice_hint[3] < 8:
            return 'Heading towards a voice', f'microphones heard a call at bearing {math.degrees(b.voice_hint[2]):.0f} deg'
        if b.ic_area:
            return 'Searching the room the IC chose first', 'Incident Commander priority'
        tgt = g.to_world(*b.goal) if b.goal else None
        where = self.room_of(tgt) if tgt else 'the building'
        return f'Exploring {where}', 'nearest area its cameras have not checked yet'

    # ---------------------------------------------------------------- IC commands
    def command(self, c: dict):
        a = c.get('action')
        b = self.w.b
        if a in ('stop', 'resume', 'recall'):
            with self.w.lock:
                b.ic(a)
            self.msg = f'IC: {a.upper()} sent to robot'
        elif a == 'room':
            k = int(c['room'])
            if not 0 <= k < len(self.rooms):
                raise IndexError('no such room')
            x0, y0, x1, y1, kind = self.rooms[k]
            with self.w.lock:
                b.ic('SEARCH', area=(x0, y0, x1, y1))
            self.msg = f'IC: search R{int(c["room"]) + 1} ({kind}) first'
        elif a == 'fire_at':
            x, y = float(c['x']), float(c['y'])
            if not (0.3 <= x <= self.L.width - 0.3 and 0.3 <= y <= self.L.height - 0.3):
                raise ValueError('outside the building')
            self.msg = 'TEST: ' + self.world.fire_at((x, y))
        elif a == 'ignite_shelf':
            self.msg = self.world.ignite(self.shelf) or self.msg
        elif a == 'ignite_random':
            ch = [i for i, t in enumerate(self.L.things)
                  if t.kind not in ('casualty',) and self.w.s.temp_by_thing[i] < sensors.FIRE_MIN_C]
            if ch:
                self.msg = self.world.ignite(int(self.world.rng.choice(ch)))
        elif a == 'radio':
            self.w.s.force_comms_drop = not self.w.s.force_comms_drop
            self.msg = 'IC: radio dropped' if self.w.s.force_comms_drop else 'IC: radio restored'
        elif a in ('person_yes', 'person_no'):
            with self.w.lock:
                cand = [t for t in b.tracker.tracks if t.kind == 'warm' and t.ic is None
                        and t.state in ('verify', 'candidate') and t.conf >= 0.4]
                if cand:
                    tr = max(cand, key=lambda t: t.first_seen)
                    b.feedback(tr.id, a == 'person_yes')
                    self.msg = f'IC feedback on #{tr.id}: {"person" if a == "person_yes" else "not a person"}'

    # ---------------------------------------------------------------- browser payloads
    def scene_json(self) -> dict:
        L = self.L
        meta, hm = scene.asset_meta(), scene.human_meta()
        return {
            'preset': L.preset, 'name': scene.PRESETS[L.preset]['name'], 'seed': L.seed,
            'width': L.width, 'height': L.height, 'wall_h': scene.WALL_H, 'wall_t': scene.WALL_T,
            'start': L.start, 'walls': L.walls,
            'rooms': [{'box': r[:4], 'kind': r[4]} for r in self.rooms],
            'props': [{'key': p.key, 'xy': p.xy, 'yaw': p.yaw, 'z': p.z, 'scale': scene.PROP_SPEC[p.key][0],
                       'rgb': meta[p.key]['rgb']} for p in L.props if p.key in meta],
            'things': [{'i': i, 'kind': t.kind, 'style': t.style, 'xy': t.xy, 'yaw': t.yaw, 'size': t.size,
                        'responsive': t.responsive, 'awake': t.awake} for i, t in enumerate(L.things)],
            'hair': hm.get('hair', {}),
            'presets': {k: v['name'] for k, v in scene.PRESETS.items()},
        }

    def map_accuracy(self, g) -> dict:
        """Robot-built wall map (radar only, no floor plan given) vs the true plan."""
        robot = g.occ == 1
        if not robot.any():
            return {'wall_err_m': None, 'walls_found': 0.0, 'precision': None}
        err = self.truth_dist[robot]
        truth = self.truth_occ == 1
        rob_d = cv2.distanceTransform((~robot).astype(np.uint8), cv2.DIST_L2, 3) * B.RES
        explored = cv2.dilate((g.occ == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        tw = truth & explored
        return {'wall_err_m': _r(np.median(err), 2), 'precision': _r(100 * (err <= 0.3).mean(), 0),
                'walls_found': _r(100 * (rob_d[tw] <= 0.3).mean(), 0) if tw.any() else 0.0}

    def state_json(self) -> dict:
        w, b, d, s = self.w, self.w.b, self.d, self.w.s
        o = w.obs
        L = self.L
        things = []
        for i, t in enumerate(L.things):
            x, y = s._thing_xy(i)
            T = s.temp_by_thing[i]
            item = {'i': i, 'xy': (_r(x), _r(y)), 'temp': _r(T, 0), 'burning': T >= sensors.FIRE_MIN_C,
                    'growth': _r(self.world.growth(i)) if i in self.world.t_lit else 0.0}
            if i in self.world.occ:
                oc = self.world.occ[i]
                item.update(mode=oc['mode'], yaw=_r(oc['yaw'], 3),
                            route=[(_r(px), _r(py)) for px, py in oc['path'][oc['k']:]] if oc['mode'] == 'walk' else None)
            things.append(item)
        rx, ry = s._robot_xy()
        out = {'t': _r(d.time, 1), 'robot': {'x': _r(rx, 3), 'y': _r(ry, 3), 'yaw': _r(s._yaw(), 3)},
               'things': things, 'msg': self.msg, 'evacuated': self.world.evacuated,
               'say': [txt for ts, k, txt in self.said if d.time - ts < 9.0]}
        if o is None:
            return out
        with w.lock:
            g = b.grid
            path = lambda p, k=2: [g.to_world(*c) for c in p[::k]] + ([g.to_world(*p[-1])] if p else []) if p else []
            pinned = sorted([tr for tr in b.tracker.tracks if tr.state == 'pinned'], key=lambda tr: -b.priority(tr)[1])
            tracks = []
            for tr in b.tracker.tracks:
                worth = tr.state in ('pinned', 'verify') or (tr.state == 'candidate' and len(tr.hits) >= 3
                                                              and tr.conf >= 0.6 and tr.verify_tries > 0)
                if tr.kind != 'warm' or not worth:
                    continue
                lab, score, why = b.priority(tr) if tr.state == 'pinned' else ('P?', 0, 'breathing not confirmed')
                if tr.instruct == 'EVACUATE':
                    lab, why = 'EVAC', b.priority(tr)[2]
                tracks.append({'id': tr.id, 'xy': (_r(tr.xy[0]), _r(tr.xy[1])), 'state': tr.state, 'prio': lab,
                               'room': self.room_of(tr.xy),
                               'say': {'EVACUATE': 'walk out the way I came in', 'SHELTER': 'stay put, get low'}.get(
                                   tr.instruct or '', ''),
                               'why': why, 'conf': _r(tr.conf), 'responsive': tr.responsive,
                               'rank': pinned.index(tr) + 1 if tr in pinned else None})
            cov = 100 * (g.scanned & (g.occ == 0)).sum() / max(1, (g.occ == 0).sum())
            first = next((c.t for c in b.cards if c.kind == 'PIN'), None)
            lat = np.array(w.lat) if w.lat else np.zeros(1)
            # crew route = entry -> casualties in priority order (what the BA team should do first)
            slow = d.time - self._slow_t >= 0.5 or self._slow is None   # heavy summaries at 2 Hz
            crew = self._slow['crew'] if not slow else None
            if slow and b._dist is not None and b.rescue_order:
                cost = B.cost_map(g)
                cur, pts = b.start_cell, [b.start_cell]
                for tr in b.rescue_order:
                    tgt = b._near(tr.xy, b._dist) or g.to_cell(*tr.xy)
                    leg = B.astar(cost, cur, tgt)
                    if leg:
                        pts += leg[1:]
                        cur = tgt
                crew = path(pts)
            if slow:
                self._slow_t = d.time
                self._slow = {'crew': crew, 'map': self.map_accuracy(g), 'methane': report.methane(b).splitlines(),
                              'truth': R.truth_score(L, b)}
            new_cards = [R._ascii(str(c)) for c in b.cards[-40:]]
            out.update({
                'mode': b.mode, 'ic': bool(b.ic_mode or b.ic_area), 'speed': _r(b.speed), 'speed_why': b.speed_why,
                'path': path(b.path), 'exit': path(b.exit_path or []), 'crew': crew,
                'fires': [(_r(f[0]), _r(f[1]), _r(f[2], 0), self.room_of(f)) for f in b.fires],
                'now': self.now_sentence(b, o),
                'tracks': tracks,
                'robot_room': self.room_of((rx, ry)),
                'voice': (list(map(_r, b.voice_hint[:3])) if b.voice_hint and b.t - b.voice_hint[3] < 4 else None),
                'cards': new_cards,
                'obs': {'self_temp': _r(o.self_temp_c, 0), 'battery': _r(o.battery), 'co': _r(o.gas_ppm, 0),
                        'o2': _r(o.o2_pct, 1), 'comms': o.comms, 'queued': len(b.outbox)},
                'metrics': {'coverage': _r(cov, 0), 'first': first, 'lat50': _r(np.percentile(lat, 50), 1),
                            'lat95': _r(np.percentile(lat, 95), 1), 'truth': self._slow['truth'],
                            'map': self._slow['map'], 'crew_in': _r(max(0, R.CREW_ENTRY_S - d.time), 0)},
                'methane': self._slow['methane'],
            })
        out['events'], self.events = self.events, []
        return out

    def render_feeds(self):
        """Robot camera feeds: colour (blinded by smoke), IR night vision, thermal, robot-built map."""
        r = self.feed_r
        opt = mujoco.MjvOption()
        opt.geomgroup[scene.MESH_GROUP] = 1
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        r.update_scene(self.d, self.feed_cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        colour = r.render().copy()
        opt.geomgroup[scene.SMOKE_GROUP] = 0  # 850 nm IR + illuminator: sees through thin smoke much further
        r.update_scene(self.d, self.feed_cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        ir = r.render().astype(np.float32).mean(2)
        ir = np.clip((ir / max(1.0, np.percentile(ir, 99))) ** 0.6 * 255, 0, 255)
        ir = (ir + self._rng.normal(0, 9, ir.shape).astype(np.float32)) * self._vig
        night = np.stack([ir * 0.35, ir, ir * 0.4], -1).astype(np.uint8)
        enc = lambda im: cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.feeds['colour'] = enc(colour)
        self.feeds['night'] = enc(night)
        if self.w.thermal_rgb is not None:
            self.feeds['thermal'] = enc(self.w.thermal_rgb)
        with self.w.lock:
            mp = R.map_image(self.w.b, self.rooms)
        self.feeds['map'] = enc(mp)

    def close(self):
        self.feed_r.close()


# ==================================================================== HTTP


class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sim: Sim | None = None
        self.cmds: queue.Queue = queue.Queue()
        self.state = b'{}'
        self.scene = b'{}'
        self.feeds: dict[str, bytes] = {}
        self.version = 0
        self.seq = 0


def make_handler(app: App):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=WEB, **k)

        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = urlparse(self.path).path
            if p == '/api/scene':
                return self._send(app.scene, 'application/json')
            if p == '/api/state':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b'data: ' + app.state + b'\n\n')
                        self.wfile.flush()
                        time.sleep(1 / STATE_HZ)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            if p.startswith('/feed/'):
                body = app.feeds.get(p[6:].split('.')[0])
                return self._send(body, 'image/jpeg') if body else self._send(b'', 'image/jpeg', 404)
            if p.startswith('/assets/'):
                fp = os.path.normpath(os.path.join(HERE, p.lstrip('/')))
                if not fp.startswith(os.path.join(HERE, 'assets')) or not os.path.isfile(fp):
                    return self._send(b'not found', 'text/plain', 404)
                ctype = {'.png': 'image/png', '.wav': 'audio/wav', '.json': 'application/json'}.get(
                    os.path.splitext(fp)[1], 'text/plain')
                return self._send(open(fp, 'rb').read(), ctype)
            return super().do_GET()

        def do_POST(self):
            # only the console page itself may command the robot (blocks cross-site requests from other tabs)
            origin = self.headers.get('Origin', '')
            host = self.headers.get('Host', '')
            ok_hosts = (f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}')
            if host not in ok_hosts or (origin and origin.split('//', 1)[-1] not in ok_hosts):
                return self._send(b'forbidden', 'text/plain', 403)
            n = min(int(self.headers.get('Content-Length', 0) or 0), 4096)
            try:
                c = json.loads(self.rfile.read(n) or b'{}')
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._send(b'bad json', 'text/plain', 400)
            if not isinstance(c, dict):
                return self._send(b'bad command', 'text/plain', 400)
            app.cmds.put(c)
            self._send(b'{"ok":true}', 'application/json')
    return H


def clean_scenario(c: dict) -> dict:
    """Clamp scenario fields from the browser to what the generator supports."""
    preset = c.get('preset') if c.get('preset') in scene.PRESETS else 'care_home'
    return {'preset': preset, 'casualties': min(4, max(0, int(c.get('casualties', 2)))),
            'fires': min(3, max(1, int(c.get('fires', 1)))), 'seed': min(99999, max(0, int(c.get('seed', 0))))}


def serve(cfg, port=8765, quit_after=None):
    app = App(cfg)
    srv = ThreadingHTTPServer(('127.0.0.1', port), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f'FireScout IC Console: http://localhost:{port}', flush=True)

    def start(c):
        if app.sim is not None:
            app.sim.close()
        app.sim = Sim(**c)
        app.scene = json.dumps(app.sim.scene_json()).encode()
        app.version += 1

    start(cfg)
    last_state, last_feed = 0.0, 0.0
    t0 = time.perf_counter()
    while quit_after is None or app.sim.d.time < quit_after:
        f0 = time.perf_counter()
        while not app.cmds.empty():
            c = app.cmds.get()
            try:
                if c.get('action') == 'new':
                    start(clean_scenario(c))
                else:
                    app.sim.command(c)
            except (KeyError, ValueError, TypeError, IndexError) as e:  # a bad command never stops the mission
                app.sim.msg = f'ignored bad command: {e}'
        app.sim.tick()
        now = time.perf_counter()
        if now - last_state >= 1 / STATE_HZ:
            st = app.sim.state_json()
            st['version'] = app.version
            app.seq += 1
            st['seq'] = app.seq
            app.state = json.dumps(st).encode()
            last_state = now
        if now - last_feed >= 1 / FEED_HZ:
            try:  # a display bug must never stop the mission
                app.sim.render_feeds()
                app.feeds = dict(app.sim.feeds)
            except Exception as e:  # noqa: BLE001
                print('feed error:', repr(e), flush=True)
            last_feed = now
        time.sleep(max(0.0, 1 / 120 - (time.perf_counter() - f0)))
    with app.sim.w.lock:
        print('\n'.join(report.methane(app.sim.w.b)))
    srv.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=list(scene.PRESETS), default='care_home')
    ap.add_argument('--casualties', type=int, default=3)
    ap.add_argument('--fires', type=int, default=2)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--quit-after', type=float, default=None)
    a = ap.parse_args()
    serve(dict(preset=a.preset, casualties=a.casualties, fires=a.fires, seed=a.seed), a.port, a.quit_after)
