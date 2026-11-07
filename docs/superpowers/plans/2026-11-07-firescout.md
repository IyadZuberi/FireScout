# FireScout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An autonomous MuJoCo scout robot that explores a smoke-filled floor using simulated thermal + radar sensing, pins casualties with confidence, replans live around fire with a visible reason, and gives the Incident Commander rescue/exit routes and a METHANE report.

**Architecture:** `scene.py` generates a random floor and MJCF. `sensors.py` turns MuJoCo state into noisy sensor readings (the only bridge between ground truth and the robot). `brain.py` is pure numpy (no MuJoCo import): map, perception, planning, safety gate, decision cards. `run.py` runs the live demo with viewer + OpenCV overlay; `eval.py` runs headless batches vs a wall-following firefighter baseline.

**Tech Stack:** Python 3.12, mujoco 4.0.0 (pip), rerun-sdk 0.42 (open-source robotics visualiser for the IC dashboard), numpy, opencv-python, pytest. Windows laptop, RTX A2000 8 GB. No cloud in the loop.

**How the simulation is shown (two windows, side by side in the video):**
1. **MuJoCo viewer (3D world):** shadows, reflections, skybox, fog/haze for smoke, emissive flickering fire geoms with orange point lights, capsule-built casualty figures lying on the floor, textured floor/walls (MuJoCo builtin textures, no downloads), chase camera. Ctrl+right-drag = the unscripted change.
2. **Rerun viewer (IC dashboard):** thermal camera (inferno colormap) with blob boxes + confidence; fogged RGB camera ("what a normal camera sees"); 2D layered map (walls, heat, gas, scanned, path, rescue route, exit route, pins); decision cards as a `TextLog`; time-series plots (self-temp, CO ppm, loop ms); METHANE as a `TextDocument`. Timeline scrubbing lets judges replay any decision. Recording saved as `.rrd` for the submission.

**Spec:** `docs/firescout-design.md`

## Global Constraints

- All robot decisions run locally; no network call inside `Brain.step`.
- `brain.py` must never import `mujoco` or `scene.Layout` ground truth; it sees only `Observation`.
- No output string, card, or report ever states an area is "clear". Allowed: `scanned, nothing detected (conf X.XX)`.
- No-go threshold `NO_GO_C = 250.0` °C; fire blob threshold `FIRE_C = 150.0`; casualty band `30.0–42.0` °C; verify band conf `0.4–0.8`; pin at conf `>= 0.8`.
- Grid resolution `RES = 0.25` m. Fast loop target 20 Hz (`DT_CTRL = 0.05` s).
- Thermal camera: 160×120 px, horizontal FOV 57° (MuJoCo `fovy="43.6"`).
- Radar: 36 rangefinders, max range 6.0 m, noise σ 0.05 m.
- Card format: `[MM:SS] KIND trigger=<sensor> <detail> | <plan change> | rule=<rule>`.
- Sim ground truth is used only by `sensors.py` and `eval.py` scoring.
- No API keys in repo; run `agentverse/skills/review-and-push` (secret scan) before every push to the public repo.

## Review Focus

1. Every route blocked by >250 °C → `astar` returns `None`; brain holds/retreats with a card, never crashes (test in Task 5).
2. A fire dragged onto/next to the robot → robot's own cell exceeds limit → RETREAT toward coolest known free cell (test in Task 5).
3. Floor with zero casualties → METHANE says `No casualties detected (scanned NN%)`, never "clear" (test in Task 7).
4. Blob touching image edge or two warm blobs adjacent → projection stays finite, no NaN positions (test in Task 4).
5. Comms lost from t=0 → robot keeps exploring, outbox grows, returns to last-good-comms cell when exploration done (test in Task 9).

## Deck / guide cross-check (what judges want → where it lives)

| Requirement (source) | Task |
|---|---|
| ≥1 simulated sensor; decide from senses, not scripted path; act (slide 6) | 2, 4, 5 |
| Simulator: MuJoCo recommended (email) | 1 |
| "Get the stream running in the first hour" (slide 7) → viewer + screen recording working | 1 |
| "Keep it on the device"; remote decisions score lowest (slide 7) | Global constraint |
| Perception-driven action: every decision traces to perception, corrects mid-run (slide 9, 25%) | 4, 5 (cards, second look) |
| Adaptation: scene changes mid-sequence, visibly replans (25%) | 6 (drag/`f` key), 9 (fire spread) |
| Operational autonomy: stated edge budget, sensible when unsure (15%) | 5 (safety gate), 6 (latency HUD), 10 (Jetson budget in summary) |
| Relevance & impact: evidenced need, critical failure handled on screen (20%) | 7 (never-clear), 8 (eval vs baseline), 10 |
| Ambition & quality (15%) | 8 (eval), tests throughout |
| 1-page PDF: user/problem, features, architecture, where each model runs + HW + latency, limits, critical failure, safety (slide 8) | 10 |
| ≤60 s video from stream, perception next to actions, one unscripted change, single take (slide 8) | 6, 10 |
| 5-min pitch: problem, how it works, can/cannot, why it matters (slide 8) | 10 |
| Public repo written during hackathon, README with user + model IDs + verification loop, no secrets (builder guide) | 0, 10 |
| Best Use of Gemini (optional): 2+ modalities, verified output, human in control (builder guide) | 11 (stretch) |
| Stop Modal apps / GCP budget alert (email) | 11 |

## File Structure

```
firescout/
  scene.py        # Layout dataclasses, generate_layout(seed), build_mjcf(layout), load(layout)
  sensors.py      # Observation dataclass, Sensors(model, data, layout).observe()
  brain.py        # Grid, detect_blobs, Tracker, Calibrator, astar, cost_map, frontiers, rescue_route, Card, Brain
  report.py       # methane(brain) -> str
  run.py          # live demo: viewer + overlay + keys
  eval.py         # headless batch, baseline, results.md
  test_scene.py  test_sensors.py  test_brain.py  test_report.py  test_eval.py
  requirements.txt  README.md
docs/ summary.pdf (Task 10)
```

---

### Task 0: Repo + environment (everyone, 15 min)

**Files:** Create `.gitignore`, `firescout/requirements.txt`

- [ ] **Step 1:** `git init` in `C:\projects\Agentverse`; `.gitignore` contains `__pycache__/`, `.venv/`, `*.mp4`, `feedback.jsonl`, `agentverse.zip`, `.env`.
- [ ] **Step 2:** `requirements.txt`: `mujoco>=3.2`, `numpy`, `opencv-python`, `rerun-sdk>=0.42`, `pytest`. Run `pip install -r firescout/requirements.txt`.
- [ ] **Step 3:** Verify: `python -c "import mujoco; print(mujoco.__version__)"` prints a version; `python -m mujoco.viewer` opens a window.
- [ ] **Step 4:** Copy `agentverse/` kit out of the repo path (keep only for `review-and-push`); create team GitHub repo (public), push.
- [ ] **Step 5:** Commit `chore: init repo`.

### Task 1: Scene generator + MJCF (Person A)

**Files:** Create `firescout/scene.py`, Test `firescout/test_scene.py`

**Interfaces — Produces:**
- `@dataclass Thing: name: str; kind: Literal['fire','casualty','decoy','shelf']; xy: tuple[float,float]; size: tuple[float,float,float]; temp_c: float; breathing: bool = False`
- `@dataclass Layout: width: float; height: float; walls: list[tuple[float,float,float,float]]  # x0,y0,x1,y1 segments; things: list[Thing]; start: tuple[float,float]; seed: int`
- `generate_layout(seed: int, n_rooms: int = 5) -> Layout` — 12×8 m floor, horizontal corridor (y 3.5–4.5), rooms above/below with 1.0 m door gaps; start = corridor west end (0.6, 4.0). Places 1–2 fires (`temp_c` 400–700), 0–2 casualties (`temp_c` 34–37, size 1.7×0.5×0.3 lying, `breathing=True`), 2–3 decoys (radiator 45–60 °C 1.0×0.1×0.6; pipe 70 °C; pet-size blob 38 °C 0.4×0.3×0.3, `breathing=False`), 1 shelf (temp 20, becomes "burning shelf" in demo by setting 450). Deterministic per seed; no thing overlaps a wall or the start within 1.0 m.
- `build_mjcf(layout: Layout) -> str` — walls as static boxes (h 2.0 m); each Thing a body with `<freejoint/>` named `thing{i}` and one geom `thing{i}_g` (so it can be dragged in the viewer); robot body `scout` (box 0.4×0.3×0.15) with joints `x`,`y` (slide) and `yaw` (hinge), `<velocity>` actuators `ax`,`ay`,`ayaw` (kv 50/50/10, ctrlrange ±1.0/±1.0/±1.5); camera `thermal` on scout facing +x, `fovy="43.6"`; 36 sites `rf{k}` on scout at 10° steps with zaxis radial in xy; `<rangefinder site="rf{k}"/>` sensors; `<visual><map fogstart="1" fogend="6"/><rgba fog="0.35 0.35 0.35 1"/></visual>`.
- `load(layout) -> tuple[mujoco.MjModel, mujoco.MjData]`
- Visual quality (in `build_mjcf`, builtin assets only): `<texture type="skybox" builtin="gradient">`, floor `builtin="checker"` concrete tones with `reflectance=0.1`, walls matte plaster material, headlight dimmed + 2 ceiling spotlights with `castshadow="true"`; each fire = 3 stacked emissive ellipsoids (`emission=1`, orange→yellow) + a `<light>` on the fire body (diffuse 1 0.5 0.1) whose intensity run.py flickers; casualty = capsules (head sphere, torso, 2 arms, 2 legs) lying on the floor, skin/clothing colours, all geoms share the casualty temp; decoys look like what they are (radiator = ribbed boxes, pipe = cylinder, pet = small capsule body); shelf = box frame with boards; scout = dark-red chassis + 2 track boxes + sensor mast with a small "thermal" lens cylinder; `<visual><quality shadowsize="4096"/><map fogstart="1" fogend="7"/><rgba fog="0.32 0.32 0.34 1" haze="0.3 0.3 0.32 1"/></visual>`; tracking camera `chase` (mode="trackcom") on scout.

- [ ] **Step 1: Write failing tests**
```python
def test_layout_deterministic():
    assert generate_layout(3) == generate_layout(3)
def test_layout_has_required_things():
    L = generate_layout(1)
    kinds = [t.kind for t in L.things]
    assert 1 <= kinds.count('fire') <= 2 and kinds.count('shelf') == 1 and kinds.count('decoy') >= 2
def test_mjcf_loads_and_steps():
    m, d = load(generate_layout(1))
    assert m.nsensor == 36 and m.ncam == 1
    for _ in range(10): mujoco.mj_step(m, d)
def test_robot_moves_with_ctrl():
    m, d = load(generate_layout(1)); x0 = d.qpos[0]
    d.ctrl[0] = 0.5
    for _ in range(200): mujoco.mj_step(m, d)
    assert d.qpos[0] > x0 + 0.3
```
- [ ] **Step 2:** `pytest firescout/test_scene.py -v` → FAIL (import error).
- [ ] **Step 3:** Implement the three functions per Interfaces. Robot qpos order must be `x, y, yaw` first (declare scout body before things).
- [ ] **Step 4:** `pytest firescout/test_scene.py -v` → PASS.
- [ ] **Step 5 (first-hour "stream" check):** `python -c "import mujoco.viewer, scene; m,d=scene.load(scene.generate_layout(1)); mujoco.viewer.launch(m,d)"` opens; start Win+Alt+R (Xbox Game Bar) recording for 5 s and confirm an mp4 is saved. Ctrl+right-drag on a thing moves it.
- [ ] **Step 6:** Commit `feat: procedural floor + mjcf`.

### Task 2: Sensor models (Person A)

**Files:** Create `firescout/sensors.py`, Test `firescout/test_sensors.py`

**Interfaces — Consumes:** Task 1 `Layout`, `load`, body/geom names. **Produces:**
- `@dataclass Observation: t: float; pose: tuple[float,float,float]  # x,y,yaw with odometry noise; thermal: np.ndarray  # (120,160) float32 °C; depth: np.ndarray  # (120,160) float32 m, radar range-at-bearing stand-in; ranges: np.ndarray  # (36,) m, 6.0 = no hit; gas_ppm: float; comms: bool; self_temp_c: float; battery: float; rgb: np.ndarray | None  # fogged, overlay only; breath: float | None  # set when breath_check requested`
- `class Sensors: __init__(self, model, data, layout, seed: int = 0); observe(self, want_breath_xy: tuple[float,float] | None = None) -> Observation; force_comms_drop: bool`
- Thermal: render segmentation from camera `thermal` at 160×120 with `mujoco.Renderer`; map geom id → temp (`Thing.temp_c` by geom name, walls 22, floor 20, scout 30); `cv2.GaussianBlur(k=5)`; add N(0, 0.5) noise. Temps read live from a dict `self.temp_by_geom` so run.py can ignite the shelf.
- Fire positions for gas/self-temp are read live from `data.xpos` (dragged fires move their heat).
- Gas: `CO = Σ_fires 2000·exp(-d/2.0)` + N(0, 20), clipped ≥ 0.
- Self-temp: `T += DT·0.05·(T_local − T)`, `T_local = 20 + Σ_fires (temp−20)·exp(−d/0.8)`; starts 25 °C. Battery drains 1/600 per second.
- Comms: `False` if `force_comms_drop` or distance from start > 9.0 m.
- Breath check (only when `want_breath_xy` given): if a `breathing` Thing lies within 0.5 m of target AND robot distance ≤ 3.0 m AND `mujoco.mj_ray` line-of-sight unblocked → 0.85 + N(0,0.05) else 0.10 + N(0,0.05), clipped [0,1].

- [ ] **Step 1: Write failing tests**
```python
def test_thermal_sees_fire_hotter_than_ambient():
    L = generate_layout(1); m, d = load(L); s = Sensors(m, d, L)
    # teleport robot to face the first fire from 2 m
    ...  # set d.qpos[0:3], mj_forward
    o = s.observe()
    assert o.thermal.shape == (120, 160) and o.thermal.max() > 150
def test_ranges_shape_and_bounds():
    o = Sensors(*load(L := generate_layout(1)), L).observe()
    assert o.ranges.shape == (36,) and (o.ranges > 0).all() and (o.ranges <= 6.0).all()
def test_breath_only_for_breathing_casualty():
    # robot 2 m from casualty with LOS -> > 0.6 ; aimed at decoy -> < 0.4
def test_comms_drop_flag():
    s.force_comms_drop = True; assert s.observe().comms is False
```
(Write the elided setup with a helper `face(d, m, xy_target, dist)` in the test file.)
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `pytest firescout/test_sensors.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: thermal/radar/gas/comms sensor models`.

### Task 3: Grid map + planners (Person B)

**Files:** Create `firescout/brain.py` (map/planning section), Test `firescout/test_brain.py`

**Interfaces — Produces (all pure numpy):**
- `RES = 0.25; NO_GO_C = 250.0`
- `class Grid: __init__(self, width: float, height: float, res: float = RES)`; arrays `occ` (int8: −1 unknown, 0 free, 1 wall), `heat` (float32 °C, init 20), `gas` (float32), `scanned` (bool); `to_cell(x, y) -> tuple[int,int]`; `to_world(i, j) -> tuple[float,float]`
- `integrate_ranges(grid, pose, ranges, max_range=6.0) -> None` — Bresenham each ray: free along, wall at hit (if < max_range).
- `cost_map(grid, no_go_c=NO_GO_C) -> np.ndarray` — `heat_eff = GaussianBlur(heat, sigma = 1.0 m in cells)`; cost = `1 + max(0,(heat_eff−40)/50) + gas/500`; unknown → 1.5; wall or `heat_eff > no_go_c` → `np.inf`.
- `astar(cost, start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]] | None` — 8-connected, diagonal ×√2, `heapq`.
- `frontiers(grid) -> list[tuple[int,int]]` — free cells with ≥1 unknown 4-neighbour.
- `rescue_route(cost, start, targets: list[tuple[int,int]]) -> tuple[list[int], list[tuple[int,int]]] | None` — brute force over permutations (≤4 targets), minimise summed A* cost.

- [ ] **Step 1: Write failing tests**
```python
def test_astar_avoids_no_go():
    g = Grid(4, 4); g.occ[:] = 0; g.heat[6:10, 0:12] = 400  # hot band with gap at the right
    c = cost_map(g); p = astar(c, g.to_cell(0.5, 0.5), g.to_cell(0.5, 3.5))
    assert p is not None and all(np.isfinite(c[i, j]) for i, j in p)
def test_astar_none_when_fully_blocked():
    g = Grid(4, 4); g.occ[:] = 0; g.heat[6:10, :] = 400
    assert astar(cost_map(g), g.to_cell(0.5, 0.5), g.to_cell(0.5, 3.5)) is None
def test_integrate_ranges_marks_wall():
    g = Grid(6, 6); r = np.full(36, 6.0); r[0] = 2.0
    integrate_ranges(g, (1.0, 3.0, 0.0), r)
    assert g.occ[g.to_cell(3.0, 3.0)] == 1 and g.occ[g.to_cell(2.0, 3.0)] == 0
def test_rescue_route_orders_targets():
    ...  # two targets, nearer one visited first
```
(Index convention: `occ[i, j]` with `i = int(y/res)`, `j = int(x/res)`. State it in a docstring.)
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `pytest firescout/test_brain.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: grid, cost map, A*, frontiers, rescue route`.

### Task 4: Perception — blobs, tracker, confidence (Person C)

**Files:** Modify `firescout/brain.py` (perception section), Test `firescout/test_brain.py`

**Interfaces — Consumes:** `Observation` fields `thermal`, `depth`, `pose`. **Produces:**
- `HFOV_DEG = 57.0; FIRE_C = 150.0; CAS_LO, CAS_HI = 30.0, 42.0`
- `@dataclass Blob: kind: Literal['fire','warm']; xy: tuple[float,float]; temp_c: float; width_m: float; height_m: float`
- `detect_blobs(thermal, depth, pose, hfov_deg=HFOV_DEG) -> list[Blob]` — threshold masks (`>FIRE_C` fire; `CAS_LO..CAS_HI` warm), `cv2.connectedComponentsWithStats`, drop area < 6 px; projection below; skip blobs whose median depth is not finite or ≤ 0.
- `@dataclass Track: id: int; kind: str; xy: tuple[float,float]; temps: deque; hits: deque  # maxlen 5 of bools; width_m: float; height_m: float; breath: float | None = None; conf: float = 0.0; state: Literal['candidate','verify','pinned','rejected'] = 'candidate'; ic: bool | None = None`
- `class Tracker: update(self, blobs: list[Blob], t: float) -> list[Track]` — associate by nearest within 0.75 m; unmatched tracks append a miss only if their xy is inside the current camera view cone (≤ 4 m).
- `class Calibrator: w: np.ndarray; features(track) -> np.ndarray; conf(track) -> float; update(track, label: bool) -> None; save(path='feedback.jsonl')`
  - features = `[1, temp_in_band, body_size_score, aspect_lying, hits_ratio, breath_or_0.5]`; `conf = sigmoid(w·f)`; initial `w = [-4, 1.5, 1.5, 1.0, 2.0, 3.0]`; `update` = one logistic SGD step, lr 0.5, appends `{features, label, t}` to `feedback.jsonl`.
  - `body_size_score = exp(-((max(w,h)-1.7)/0.5)**2)`; `aspect_lying = 1 if w/h > 1.8 else 0`.

Projection (not determined by tests alone):
```python
fx = (W / 2) / tan(radians(hfov) / 2)
bearing = yaw + atan((cx - u_centroid) / fx)      # image left = CCW
r = median(depth[mask])
xy = (x + r*cos(bearing), y + r*sin(bearing))
width_m, height_m = r*w_px/fx, r*h_px/fx
```

- [ ] **Step 1: Write failing tests**
```python
def test_detect_fire_and_warm_blob_projection():
    th = np.full((120,160), 20, np.float32); th[50:70, 70:90] = 500; th[60:66, 10:40] = 36
    dp = np.full((120,160), 3.0, np.float32)
    bs = detect_blobs(th, dp, (0.0, 0.0, 0.0))
    fire = [b for b in bs if b.kind == 'fire'][0]
    assert abs(fire.xy[0] - 3.0) < 0.2 and abs(fire.xy[1]) < 0.2
    warm = [b for b in bs if b.kind == 'warm'][0]
    assert warm.xy[1] > 0          # left of image => +y
def test_blob_at_edge_is_finite():
    th = np.full((120,160), 20, np.float32); th[0:10, 150:160] = 36
    bs = detect_blobs(th, np.full((120,160), 2.0, np.float32), (1, 1, 0))
    assert all(np.isfinite(b.xy).all() for b in bs)
def test_decoy_scores_below_casualty():
    cal = Calibrator()
    cas = Track(1,'warm',(0,0),deque([36]*5),deque([True]*5),1.7,0.4, breath=0.85)
    rad = Track(2,'warm',(1,0),deque([50]*5),deque([True]*5),1.0,0.6, breath=0.1)
    assert cal.conf(cas) >= 0.8 > 0.4 > cal.conf(rad)
def test_feedback_lowers_similar_track():
    cal = Calibrator(); t = Track(3,'warm',(0,0),deque([38]*5),deque([True]*5),0.4,0.3)
    before = cal.conf(t); cal.update(t, False); assert cal.conf(t) < before
```
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** Commit `feat: thermal perception, tracker, calibrated confidence`.

### Task 5: Brain loop — modes, safety gate, cards (Person B + C)

**Files:** Modify `firescout/brain.py` (Brain section), Test `firescout/test_brain.py`

**Interfaces — Consumes:** Tasks 3–4. **Produces:**
- `@dataclass Card: t: float; kind: Literal['REPLAN','VERIFY','PIN','REJECT','RETREAT','COMMS','LEARN','HOLD']; trigger: str; detail: str; change: str; rule: str; __str__ -> '[MM:SS] KIND trigger=... detail | change | rule=...'`
- `@dataclass Command: vx: float; vy: float; wz: float; want_breath_xy: tuple[float,float] | None = None`
- `class Brain: __init__(self, width: float, height: float, start_xy); step(self, obs: Observation) -> Command`; public state: `grid, tracker, cal, mode: Literal['EXPLORE','VERIFY','RETREAT','RETURN_COMMS','HOLD','DONE'], path, rescue, exit_path, cards: list[Card], outbox: list[Card], last_comms_cell, timings: dict[str,float]`; `feedback(track_id: int, is_person: bool) -> None` (emits LEARN card).
- Step order each tick (timed into `timings`): integrate ranges → heat/gas into grid (fire blobs paint `heat` within 0.3 m; robot cell `gas = obs.gas_ppm`) → mark `scanned` cells in view cone ≤ 4 m → tracker → conf → state transitions → **safety gate** → plan → command.
- State transitions: conf ≥ 0.8 → `pinned` + PIN card; 0.4–0.8 and not yet verified → mode VERIFY, goal = cell 1.5 m from track, request breath; conf < 0.2 after ≥5 frames → `rejected` + REJECT card.
- Safety gate (order matters, first match wins):
  1. `obs.self_temp_c ≥ 60` or robot cell `heat_eff > NO_GO_C` → RETREAT to the known free cell with lowest `heat_eff` within 6 m; RETREAT card.
  2. `obs.battery < 0.15` → go to start; card.
  3. `not obs.comms` → keep current mode, queue cards to `outbox`; COMMS card once per drop; when no frontiers left → RETURN_COMMS to `last_comms_cell`.
  4. Planned path crosses `inf` cost → REPLAN.
- REPLAN card whenever the new path differs from the previous beyond 2 cells; `trigger` = the layer that changed cost on the old path (`thermal`/`radar`/`gas`) with value; `change` = `old→new lengths (+Δs at 0.4 m/s)`.
- No reachable frontier and no verify target → HOLD (card) — never "clear".
- Controller: follow `path` with pure pursuit, lookahead 0.5 m, speed 0.4 m/s (0.2 m/s within 1.5 m of a pinned casualty), `wz` turns toward motion direction so the thermal camera looks where it goes.
- After each tick: `rescue = rescue_route(cost, robot_cell, pinned cells)`, `exit_path = astar(cost, robot_cell, start_cell)`.

- [ ] **Step 1: Write failing tests** (build synthetic `Observation`s; helper `obs(**overrides)` in test file)
```python
def test_card_format():
    assert str(Card(34.0,'REPLAN','thermal','C2=310°C(.92)>250','old 6.1m→new 10.3m (+10s)','heat-no-go')) \
        .startswith('[00:34] REPLAN trigger=thermal')
def test_retreat_when_self_hot():
    b = Brain(12, 8, (0.6, 4.0)); b.step(obs(self_temp_c=65))
    assert b.mode == 'RETREAT' and b.cards[-1].kind == 'RETREAT'
def test_hold_not_crash_when_all_blocked():
    b = Brain(4, 4, (0.5, 0.5)); b.grid.occ[:] = 0; b.grid.heat[8:, :] = 600  # cool pocket rows 0-7 (y<2 m), rest no-go, no frontiers
    cmd = b.step(obs(pose=(0.5, 0.5, 0), ranges=np.full(36, 6.0)))
    assert b.mode == 'HOLD' and b.cards[-1].kind == 'HOLD'
    assert abs(cmd.vx) < 1e-6 and abs(cmd.vy) < 1e-6
def test_no_card_says_clear():
    b = Brain(12, 8, (0.6, 4.0))
    for k in range(50): b.step(obs(t=k*0.05))
    assert not any('clear' in str(c).lower() for c in b.cards)
def test_comms_loss_queues_and_continues():
    b = Brain(12, 8, (0.6, 4.0)); b.step(obs(comms=False))
    assert b.mode != 'HOLD' and b.outbox and b.cards[-1].kind in ('COMMS', 'REPLAN')
```
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `pytest firescout -v` → all PASS.
- [ ] **Step 5:** Commit `feat: brain loop, safety gate, decision cards`.

### Task 6: Live demo runner + overlay (Person C) — **Must milestone, record backup video by 22:00**

**Files:** Create `firescout/run.py`

**Interfaces — Consumes:** `scene.load/generate_layout`, `Sensors`, `Brain`, `report.methane`.
- `main(seed: int = 7)`: `mujoco.viewer.launch_passive`; loop: every `DT_CTRL` → `obs = sensors.observe(cmd.want_breath_xy)` → `cmd = brain.step(obs)` → `d.ctrl[:] = [cmd.vx, cmd.vy, cmd.wz]`; physics steps in between; `viewer.sync()`.
- Dashboard: `rr.init("FireScout", spawn=True)` + `rr.save("run.rrd")`; per tick `rr.set_time("sim", duration=t)` and log: `thermal` (`rr.Image` of inferno-mapped thermal, 2×) + `thermal/blobs` (`rr.Boxes2D` with labels `"casualty? 0.58"`), `rgb` (fogged camera), `map` (`rr.Image` RGB composite: occ grey, heat red, gas yellow, scanned tint) + `map/path|rescue|exit` (`rr.LineStrips2D` cyan/green/white) + `map/pins` (`rr.Points2D` with labels), `cards` (`rr.TextLog`, level WARN for RETREAT/COMMS, INFO otherwise), `metrics/self_temp|co_ppm|loop_ms` (`rr.Scalars`), `methane` (`rr.TextDocument`, every 2 s). Send a `rr.blueprint.Blueprint` layout once: left column thermal over rgb, centre map, right cards over METHANE, bottom metrics. Map pixels use the same `occ[i, j]` convention flipped so +y is up.
- Fire flicker: every tick set each fire light `model.light_diffuse` = base × U(0.7, 1.0).
- Keys via `launch_passive(..., key_callback=...)` in the MuJoCo window: `F` = ignite random thing (450 °C), `S` = ignite shelf, `C` = toggle comms drop, `Y`/`N` = IC feedback on newest non-final track, `M` = print METHANE, `Q` = quit. (Use GLFW key codes; letters = ord of uppercase.)
- Fire spread: every 20 s, the nearest non-fire thing within 1.5 m of a fire ignites (450 °C) and emits a card from run.py (`kind='REPLAN'` triggered next tick by brain naturally).

- [ ] **Step 1:** `python firescout/run.py --seed 7` → MuJoCo viewer + Rerun dashboard open, robot explores, cards scroll; enable `mjRND_SHADOW`, `mjRND_REFLECTION`, `mjRND_FOG`, `mjRND_HAZE`, `mjRND_SKYBOX` on `viewer.user_scn.flags`/`scn.flags`.
- [ ] **Step 2:** Manual check list (all must be seen): thermal shows fire hot; a decoy goes VERIFY then REJECT; casualty goes VERIFY → PIN; drag shelf (after `s`) across corridor → REPLAN card within 1 s; `c` → COMMS card, robot keeps moving; p95 loop < 50 ms in HUD.
- [ ] **Step 3:** Record a 60 s Win+Alt+R backup take. Commit `feat: live demo + IC overlay`.

### Task 7: METHANE report (Person D)

**Files:** Create `firescout/report.py`, Test `firescout/test_report.py`

**Interfaces — Produces:** `methane(brain: Brain, t: float) -> str` — seven lines prefixed `M:`, `E:`, `T:`, `H:`, `A:`, `N:`, `E:`. N lists pinned casualties with grid ref + conf and unconfirmed candidates separately; when none: `N: No casualties detected (scanned NN% of reachable area) — human search still required`. H lists fire cells > 250 °C, max gas ppm. A = exit route length + "safe route via ...". Never contains the word "clear" except inside "not cleared".

- [ ] **Step 1:** Tests: `test_methane_has_seven_lines`, `test_zero_casualties_never_clear` (`'clear' not in text.lower().replace('not cleared','')` and `'human search still required' in text`).
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS → **Step 5:** Commit `feat: METHANE report`.

### Task 8: Evaluation vs firefighter baseline (Person D)

**Files:** Create `firescout/eval.py`, Test `firescout/test_eval.py`

**Interfaces — Produces:**
- `run_episode(seed: int, t_max: float = 180.0) -> dict` — headless (no viewer; renderer only); returns `{seed, t_first_casualty or None, found, total_casualties, false_pins, heat_breaches, scanned_pct_at_180s, loop_ms_p50, loop_ms_p95}`. Ground truth (from `Layout`) used **only here** for scoring: a pin is true if within 1.0 m of a casualty.
- `baseline_time(layout: Layout, crew_delay_s: float = 150.0, speed_mps: float = 0.3) -> float | None` — right-hand wall-follow on the true occupancy grid from start; casualty "found" when within 1.0 m; time = `crew_delay_s + path_len / speed_mps`. Both constants are stated assumptions printed in `results.md`.
- `main(n: int = 30)` writes `results.md`: per-seed table + median/IQR time-to-first-casualty for both, recall & precision with Wilson 95% CI, heat breaches total, false-clear = 0 (by construction, stated).

- [ ] **Step 1:** Tests: `test_wilson_ci_bounds` (`wilson(5,5)` upper == 1.0, lower > 0.4), `test_run_episode_keys` (seed 1, `t_max=20`), `test_baseline_finds_reachable_casualty`.
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS.
- [ ] **Step 5:** `python firescout/eval.py --n 30` → `results.md` generated (fallback `--n 15` if > 20 min).
- [ ] **Step 6:** Commit `feat: eval harness + results`.

### Task 9: Should-tier features (Sun 09:00–10:30, split A/B/C)

Most already have hooks; this task finishes and tests them.
- [ ] Breath check wired end-to-end (Brain VERIFY requests `want_breath_xy`; sensors returns `breath`; calibrator uses it). Test: `test_verify_requests_breath`.
- [ ] Gas layer drawn on map and in cost. Test: `test_gas_raises_cost`.
- [ ] Comms-loss return: Review Focus #5 test `test_comms_lost_from_start_returns_when_done`.
- [ ] Fire spread timer in run.py (Task 6) verified on screen.
- [ ] IC feedback `y`/`n` → LEARN card and `feedback.jsonl` line (gitignored).
- [ ] `pytest firescout -v` all PASS; commit `feat: should-tier sensing + learning`.
- [ ] **10:30 feature freeze.**

### Task 10: Deliverables (Person D lead, everyone 11:00–13:30)

- [ ] `README.md`: named user (UK Incident Commander), problem + evidence, architecture diagram, where each part runs (laptop now; Jetson Orin Nano Super budget, measured loop ms), how to run (`pip install -r firescout/requirements.txt; python firescout/run.py --seed 7`), results table from `results.md`, honest limits (simulated sensor models, single floor, no stairs/doors, no real fire physics), safety rules, "ground truth only for scoring" statement, model IDs used (none in loop; list Gemma/Gemini only if Task 11 ships).
- [ ] `docs/summary.pdf` (1 page): sections exactly from slide 8 — user & problem, key features, system architecture, where each model runs with hardware & latency, limits, critical failure mode (false "clear"/missed casualty → never-clear rule + second look + breath check), safety.
- [ ] Video ≤ 60 s, single continuous take of screen (viewer + IC panel): fogged RGB vs thermal → explore + cards → radiator REJECT (+ `n` LEARN) → casualty VERIFY → breath → PIN → teammate drags burning shelf (unscripted) → REPLAN card → `c` comms drop → robot continues → RETREAT near heat → `m` METHANE.
- [ ] Pitch 5 min: survival-vs-minutes story 30 s · live demo 2 min · how/why trust 1 min · eval numbers 45 s · cannot-do + cost (≈£1–2k BOM vs $75k–$210k) + future (stairs, drone partner, hybrid) 45 s.
- [ ] Run `review-and-push` (secret scan) → push to public repo → submit by **13:30**.

### Task 11: Stretch (only if Task 10 drafts exist)

- [ ] `report.py`: optional Gemma 4 E2B rewording via local Ollama, 3 s timeout, template fallback; label output "AI-reworded".
- [ ] Extinguishing puck: `p` key proposes; IC must press `a` to approve; fire within 1.0 m drops to 60 °C; cards PROPOSE/APPROVED.
- [ ] Gemini 4 Argon after-action review script (`review.py`) on recorded mp4 + `log.jsonl`, offline only; env var `GEMINI_API_KEY`, never committed. Set GCP budget alert first.
- [ ] Modal LoRA on `feedback.jsonl` — only describe pipeline unless ≥ 1 h spare; stop Modal apps after.
