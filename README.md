# FireScout

**An autonomous scout robot that searches a smoke-filled building for casualties before firefighters go in.**
Built for the AgentVerse 2026 Robotics track (simulation: MuJoCo).

> **User:** a UK fire service Incident Commander (IC) and the breathing-apparatus (BA) crew about to enter.
> **Problem:** crews search blind in smoke, and survival falls as minutes pass. In ~1,000 recorded rescues, survival was
> ~75% when the victim was reached within 2 min of arrival and ~28% after 8 min (Fire Engineering tally; correlation).
> Existing robots are big, teleoperated and cost $75k–$210k (Spot, Thermite, Colossus).

FireScout is placed at the doorway while the crew puts on BA (2–5 min of otherwise dead time). It explores on its own,
pins casualties with a confidence score, replans live around fire with a visible reason for every decision, and hands
the IC a rescue route, an exit route and a **METHANE** report. **Every decision runs on the robot. Nothing phones home.**

## What you see in the demo

| Window | Shows |
|---|---|
| MuJoCo viewer | the 3D world: floor plan, fires (flickering light), casualty, decoys (radiator, hot pipe, pet), the robot and its radar rays. Ctrl+right-drag any object = unscripted change. |
| Rerun dashboard | thermal camera with labelled detections · fogged normal camera (why thermal is needed) · the robot's own layered map (walls, heat, gas, thermally-scanned area, path / rescue / exit routes, pins) · decision cards · live METHANE · health and loop latency |
| IC control panel | ignite the shelf / a random object, drop the radio, confirm or reject a candidate, print METHANE |

## How it works

```
SENSE (simulated, noisy)          real part it stands for
  thermal camera 160x120, 57°     FLIR Lepton 3.5 (~$164)
  36-ray range ring, 6 m          60 GHz mmWave radar (sees through smoke)
  radar breathing check ≤2.5 m    same radar, micro-motion
  CO sensor, IMU/odometry, radio RSSI, self-temperature, battery

FAST LOOP 20 Hz, on-device, deterministic (brain.py, numpy only)
  perceive  thermal blobs → fire (>150°C) / warm candidate; multi-frame tracker
            confidence = logistic(temp-in-body-band, body size, lying shape, hit ratio, radar breathing)
  map       0.25 m grid: occupancy (radar) · heat · gas · scanned
  plan      Dijkstra/A* on cost = distance + heat + gas; goals = unexplored + not-yet-thermally-scanned
            rescue route (best visiting order) + exit route
  verify    0.4 ≤ conf < 0.8 → drive to a spot with line of sight, 1 s radar breathing check
            PIN only if thermal conf ≥ 0.8 AND breathing ≥ 0.5 (two independent sensors)
  safety    fixed rules, never learned: no-go > 250°C · retreat at self-temp 60°C or low battery ·
            radio lost → keep working, queue findings, return to last radio contact · never declares an area clear
  explain   every replan / pin / reject / retreat emits a decision card citing the sensor reading that caused it

IC FEEDBACK  ✅/❌ on a candidate → one logistic-regression step (live learning) + saved to feedback.jsonl
REPORT       METHANE (JESIP) built only from the robot's own map
```

Example decision cards:

```
[00:22] PIN trigger=thermal+radar casualty#1 at (2.0,2.1) conf 0.96 breathing 0.85 | added to rescue route | rule=two-sensor-confirm
[00:38] REJECT trigger=thermal track#86 58°C 0.2x0.6m conf 0.16 | ignored | rule=not-body-like
[00:46] POSSIBLE trigger=radar track#2 conf 0.51 breathing 0.08 | kept as POSSIBLE casualty for crew | rule=never-drop-candidate
[02:52] RETREAT trigger=self-temp self 60°C cell 20°C battery 71% | retreating to coolest known area | rule=heat-limit
```

## Where each part runs

| Part | Runs on | Latency |
|---|---|---|
| Perception, mapping, planning, safety, cards | robot CPU (target: Jetson Orin Nano Super, 7–25 W) | measured on laptop CPU: see [results.md](firescout/results.md) (budget 50 ms) |
| METHANE report | robot CPU, template | < 1 ms |
| Cloud models | **none in the decision loop** | — |

## Results

Simulated, on 30 procedurally generated floors the robot has never seen, against a wall-following crew baseline.
Full table and assumptions: **[firescout/results.md](firescout/results.md)**.

## Run it

```bash
pip install -r firescout/requirements.txt
cd firescout
python run.py --seed 11            # live demo: MuJoCo + Rerun + IC panel (saves firescout_run.rrd)
python eval.py --n 30 --workers 8  # headless evaluation → results.md
python -m pytest -q                # tests
```

## Honest limits

- Simulation only. Sensor models are simple and documented in `sensors.py`; there is no real fire, smoke or heat physics.
- One floor; no stairs or doors to open; the robot's body physics is simplified (planar base).
- Breathing detection assumes line of sight and ≤2.5 m; 60 GHz radar does not see through rubble.
- Ground truth is used only by the sensor models and the evaluation scorer. The brain never sees it.
- A "scanned, nothing detected" area still needs a human search. FireScout never reports an area as clear.

## Real robot (estimate)

Lepton 3.5 thermal (~$164 + ~$100 board) · 60 GHz radar (~$150–300) · Jetson Orin Nano Super ($249, 176 g) ·
CO/O₂/LEL sensors (~$100–200) · tracked base, motors, battery (~$300–600) · heat shield (~$100–300).
**≈ $1.3k–2.3k, 3–5 kg**, placed or slid through a doorway. Prices are prototype estimates.

## Repo map

`firescout/scene.py` floor generator + MJCF · `sensors.py` sensor models · `brain.py` the robot ·
`report.py` METHANE · `run.py` live demo · `eval.py` evaluation · `test_*.py` tests ·
[docs/firescout-design.md](docs/firescout-design.md) design.
