# FireScout

**An autonomous scout robot that searches a smoke-filled building for people before firefighters go in.**
Built for the AgentVerse 2026 Robotics track. Simulator: **MuJoCo** (physics, sensors, agent). Live view: a browser
IC console (three.js) that only draws the MuJoCo state.

> **User:** a UK fire service Incident Commander (IC) and the breathing-apparatus (BA) crew about to enter.
> **Problem:** crews search blind in smoke, and survival falls as minutes pass. In ~1,000 recorded rescues, survival was
> ~75% when the victim was reached within 2 min of arrival and ~28% after 8 min (Fire Engineering tally; correlation).
> Existing robots are big, teleoperated and cost $75k–$210k (Spot, Thermite, Colossus).

FireScout is placed at the doorway while the crew puts on BA (2–5 min of otherwise dead time). It has **no floor plan**.
It maps the building with radar and depth, finds people with thermal + radar breathing + microphones, **ranks who to
rescue first**, **talks to conscious people** through its speaker (walk out if the way is safe, otherwise stay put and
stay low), replans live around growing fire, and hands the IC crew routes, an exit route and a **METHANE** report.
**Every decision runs on the robot. Nothing phones home.** The IC can STOP, RECALL or send it to a room at any time;
its safety rules still win.

## Run it

```bash
pip install -r firescout/requirements.txt
cd firescout
python web.py                       # live IC console -> open http://localhost:8765  (pick a scenario, press Sound)
python run.py                       # fallback: native MuJoCo viewer + IC panel (scenario picker first)
python eval.py --n 30 --workers 8   # headless evaluation -> results.md
python -m pytest -q                 # tests
```

## What you see

| Panel | Shows |
|---|---|
| 3D view | building cutaway, real people (unconscious lying, awake standing / walking out / kneeling), growing particle fire + smoke banking down, the robot (sensor head, radar, mics, speaker, beacon) with its camera field of view, routes: robot's next move (cyan), **crew route in rescue order** (green arrows), exit (white), occupant walk-out (yellow), **fire no-go zones** (red), voice bearing (pink) |
| Rescue priority | P1 / P2 / possible per person with the reason (distance to fire, heat, CO, O2, calling or silent) - "HANDLE FIRST" banner |
| Camera feeds | thermal (sees bodies through smoke) · near-IR night vision · colour camera (blinded by smoke) |
| Robot-built map | what the robot mapped with radar + depth (walls white, furniture brown) over the hidden true plan (magenta) + accuracy |
| Decision log | every action with the sensor reading and rule behind it |
| IC controls | STOP / RESUME / RECALL, search a room first, radio drop, ignite shelf or object, confirm / reject a person |
| Robot & agent | the robot with labelled sensors, spec, and the perceive -> understand -> decide -> act -> learn loop |

## How the agent works (brain.py, numpy only, 10 Hz)

```
PERCEIVE   thermal 160x120 LWIR · 60 GHz radar ring (36 rays, 6 m) + breathing micro-motion · depth/IR camera
           4-mic array (voice bearing) · CO + O2 cells · self-temperature · battery · radio
UNDERSTAND 0.25 m map: walls (radar), furniture vs wall (depth: an object whose top edge is seen below 1.4 m),
           heat, CO, O2, scanned area · people vs decoys (radiator, hot pipe, pet): logistic confidence + tracker
DECIDE     fixed priority: safety gate (no-go > 250 C, self 60 C, battery) > IC order > second look at a possible
           person (move for line of sight, 1 s radar breathing, re-check from another angle) > head for a heard voice
           > explore unmapped / unscanned area. Speed adapts: 0.7 m/s on known cool routes, 0.2 m/s near heat/people.
           PIN only with two sensors (thermal >= 0.8 AND breathing >= 0.5). Never declares an area empty.
ACT        drive · speaker instruction to conscious people (evacuate if the measured route stays < 60 C and 1.5 m
           from fire, else shelter in place) · crew route in priority order · METHANE · decision cards
LEARN      IC confirm / reject -> one logistic-regression step, saved to feedback.jsonl
```

Rescue priority is for the BA crew, **not clinical triage** (UK guidance: extrication and triage stay medically led):
P1 = breathing and silent, or in a hazardous spot (< 3 m from fire, > 60 C, CO > 400 ppm, O2 < 17%) · P2 = breathing,
calling out, survivable spot · possible = warm and body-shaped, breathing not yet confirmed (crew must check).

## Fits the brief

| Brief | FireScout |
|---|---|
| Perceive from at least one simulated sensor | 6 sensor types, all simulated with noise from MuJoCo renders and ray casts |
| Decide from what it senses, not a script | every decision is computed from its own map; nothing is scripted (decision log shows why) |
| Act: move / communicate | drives, talks to occupants, reports routes + METHANE to the IC |
| Keep it on the device | whole loop on CPU, p50 ~3 ms (headless); no cloud call anywhere |
| People in control of physical actions | IC STOP / RECALL / room priority; safety gate cannot be overridden |

## Results

Simulated, on 30 procedurally generated floors the robot has never seen, against a wall-following crew baseline.
Full table and assumptions: **[firescout/results.md](firescout/results.md)**.

## Honest limits

- Simulation only. Sensor models are simple and documented in `sensors.py`; fire growth, smoke and heat are
  engineered models (no CFD). Voices are synthesised.
- One floor; no stairs or doors to open; the robot base is planar.
- Breathing detection needs line of sight within ~2.5 m; 60 GHz radar does not see through rubble.
- Ground truth is used only by the sensor models, occupant behaviour and the evaluation scorer. The brain never sees it.
- A "scanned, nothing detected" area still needs a human search.

## Real robot (estimate)

Lepton 3.5 thermal (~$164 + ~$100 board) · 60 GHz radar (~$150–300) · depth + IR camera (~$100) · 4-mic array + 10 W
speaker (~$60) · Jetson Orin Nano Super ($249) · CO/O2 cells (~$100–200) · tracked base, motors, battery (~$300–600)
· heat shield (~$100–300). **≈ $1.3k–2.3k, 3–5 kg.** Prototype estimates.

## Credits

Furniture: **NVIDIA SimReady assets** (CC-BY-4.0), converted by `assets.py`. People: **Quaternius Universal Base
Characters** (CC0), posed and clothed by `humans.py`. Rendering: three.js (MIT). Voices: Windows offline speech.

## Repo map

`firescout/scene.py` building generator + MJCF · `sensors.py` sensor models · `brain.py` the agent ·
`report.py` METHANE · `web.py` + `web/` live IC console · `run.py` native viewer · `audio.py` voices ·
`humans.py` / `assets.py` asset pipelines · `eval.py` evaluation · `test_*.py` tests ·
[docs/firescout-design.md](docs/firescout-design.md) design.
