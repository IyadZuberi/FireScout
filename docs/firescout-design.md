# FireScout — Design (approved 2026-11-07)

AgentVerse 2026 · Robotics track (NVIDIA) · Simulator: MuJoCo (organiser-recommended) · All decisions on-device.

## 1. Intent
A small autonomous scout robot is placed at the doorway of a smoke-filled building by a firefighter. During the 2–5 min the crew spends donning breathing apparatus and passing entry control, it explores, finds casualties and hazards, and hands the Incident Commander (IC) a rescue route, an exit route and a METHANE report. It runs in parallel with the crew, never delaying them.

**User:** UK fire service Incident Commander + the BA team about to enter.
**Why:** survival falls as time-to-reach-victim grows (~75% <2 min, ~28% >8 min after arrival, Fire Engineering tally of ~1,000 rescues, correlational). Crews search blind in smoke; IC information is poor/late (Grenfell: radio failures, outdated risk info). Existing robots: Spot ~$75k, Thermite ~$100k, Colossus ~$210k, mostly teleoperated.

## 2. Goals / non-goals / anti-goals
- Goals: autonomous perception-driven exploration in smoke; casualty + hazard pins with confidence; live replanning with a visible reason; rescue + exit routes; METHANE output; measured time saving vs a wall-following firefighter baseline.
- Non-goals: stairs, opening doors, flying/hybrid, real fire physics, multi-robot, real hardware, extinguishing (stretch only, IC-approved).
- Anti-goals (failure even if it "works"): robot ever declares a room "clear"; any decision depending on a cloud call; sim ground truth leaking into the robot's brain; enters >250 °C cells.

## 3. Sensing (real part → sim model)
| Sensor | Real | Sim | Tier |
|---|---|---|---|
| Thermal camera | FLIR Lepton 3.5 160×120, ~57° | segmentation render → per-geom °C + blur + noise | Must |
| mmWave radar 60 GHz | TI IWR6843-class | ring of `rangefinder` rays, coarse + noise | Must |
| Radar breath check | same radar, ≤3 m | casualty `breathing` flag, LOS + range gated, noisy | Should |
| Gas (CO/O2/LEL) | electrochemical | analytic plume from fire sources + noise at robot pos | Should |
| IMU + odometry | standard | MuJoCo sensors / base pose + drift noise | Must |
| RGB camera | optional | rendered with fog (shows failure in smoke) | Must (overlay only) |
| Comms / self-temp / battery | RSSI, thermistor | distance+walls model; heat-exposure integrator | Must/Should |
| Microphone | MEMS | 1/d² sound level | Stretch |

Real prototype BOM estimate: ≈ $1.3k–2.3k, 3–5 kg (Jetson Orin Nano Super $249/176 g; Lepton 3.5 ~$164). Deployed by placing/sliding at door (not thrown — too heavy). Works minutes in ~100–250 °C smoke, never in flames.

## 4. Brain (fast loop 20 Hz, local, deterministic)
1. Perceive: thermal blobs → clue fusion (temp 30–42 °C AND body size AND shape AND ≥3 of last 5 frames) [+ radar breath] → confidence. Fire = >150 °C.
2. Map: 0.25 m grid layers: occupancy (radar), heat, gas, scanned, comms.
3. Plan: frontier exploration → A* with cost = distance + heat + gas; replanned every tick. Rescue route (best casualty order, brute force ≤4) + exit route.
4. Second look: casualty confidence 0.4–0.8 → move closer / breath check before pinning.
5. Safety gate (fixed, never learned): no-go >250 °C; self-temp retreat; never-clear; low battery return; comms-loss → continue, store, return to last good comms; physical actions need IC approval.
6. Decision card logged + shown on every replan/pin/retreat: time, trigger sensor + value + confidence, old vs new plan, cost delta, rule.

Slow loop (~10 s, separate thread): METHANE from template; Gemma 4 E2B rewording optional (stretch). Offline (stretch): Gemini 4 Argon after-action review of video + log; Modal LoRA on saved IC feedback.

IC feedback ✅/❌ on pins → live calibration of casualty confidence (logistic/Bayesian update); stored as dataset.

## 5. Evaluation
30 procedurally generated unseen floors with decoys (radiators, hot pipes, embers, pet-size blob). Metrics: time to first casualty vs wall-following baseline (same start delay rules), casualty precision/recall with Wilson CIs, false pins, heat-limit breaches (target 0), false-clear (0 by design), % mapped by T+3 min, loop latency p50/p95. Ground truth used for scoring only.

## 6. Deliverables (from deck slide 8 + builder guide)
- 1-page PDF: user/problem, features, architecture, where each model runs + hardware + latency, limits, critical failure mode (false clear / missed casualty) + safety.
- ≤60 s video, single continuous take from screen capture: perception overlay next to actions, one unscripted change (teammate drags burning shelf in viewer).
- 5-min pitch: problem, how it works, can/cannot, why it matters.
- Public repo written during the hackathon, README (named user, model IDs, verification loop, results, limits), no secrets (review-and-push secret scan).

## 7. Judging map
| Criterion (weight) | Evidence |
|---|---|
| Perception-driven action (25) | every card cites sensor reading; second look corrects mid-run |
| Adaptation to change (25) | live drag of burning shelf + fire spread → replan card |
| Operational autonomy (15) | on-device, stated Jetson Orin Nano budget, latency on screen, comms-loss + retreat behaviour |
| Relevance & impact (20) | IC user, survival-vs-time evidence, false-clear impossible, eval vs baseline |
| Ambition & quality (15) | sensor models, layered risk map, clue fusion, eval harness, tests |

## 8. Builder's calls
Robot base = box on planar x/y/yaw joints with velocity actuators (honest simplification). Grid 0.25 m. No-go threshold 250 °C (tunable). Files: scene.py, sensors.py, brain.py, run.py, eval.py, report.py, test_brain.py.
