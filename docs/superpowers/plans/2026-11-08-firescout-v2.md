# FireScout v2 - review + build plan (for approval)

Date 2026-11-08 · deadline 13:30 · two demos: **A = MuJoCo live IC console** (judged live), **B = Isaac Sim on Modal**.

## 1. Where we are (verified, not claimed)

| Area | State | Evidence |
|---|---|---|
| Agent loop (perceive -> map -> decide -> act) | works, unscripted | `brain.py` imports no ground truth; 56 tests pass |
| Safety (no-go >250 C, self 60 C, battery, radio loss) | works | 0 heat breaches in 30 floors |
| Never "area clear" | works | 0 violations (was 1, fixed) |
| Detection | **weak** | recall 14/37 = 38% (CI 24-54%), precision 93%, 90 "possible" |
| Time to first confirmed casualty | 48 s vs crew baseline 222 s | baseline is a blind wall-crawl (strawman, say so) |
| Mapping without a plan | works | live % walls found / correct vs hidden plan |
| Rescue priority, speaker evacuate/stay put, IC STOP/RECALL/room | works | tests + live runs |
| Browser IC console | works but **19/40 usability** | design review below |
| Fire / smoke visuals | particles, growth, banking smoke | close-up and robot's-eye view still poor |
| Deliverables (video, PDF, pitch) | PDF draft only | - |

## 2. Roast (what a judge will hit)

1. **Recall 38%.** Most misses are people the robot *saw* (conf 0.7-0.97) but did not confirm in 180 s: breathing check is the bottleneck; some people get absorbed by a rejected decoy track. Hand-tuning is now noise-limited (30 floors, CIs overlap).
2. **"Training" is thin.** Detector weights are hand-set; only learning = 1 SGD step per IC click. Fine to say "learns from IC", not "trained model".
3. **Breathing radar is oracle-like** (true body position + LOS + noise). Precision 93% looks too clean; add realistic false readings.
4. **Baseline is a strawman.** Add a crew-with-TIC baseline.
5. **UI (design review, 19/40, cognitive load 6/8 fails):**
   - P0 camera feed frozen (refresh loop never starts; tabs only change caption).
   - P0 overlays cover the scene at 1280x720; overview crops fires/rooms.
   - P1 radio-lost / stale data nearly invisible.
   - P1 colour collisions (cyan = path/robot/selected/IC; magenta = voice/true plan/speaker).
   - P1 coordinates instead of room names; ~40 equal-weight numbers; no single "what to do now".
   - P2 buttons 33 px (gloves), STOP not dominant, demo buttons (ignite, radio drop) mixed with IC controls.
   - P3 robot's-eye view blown out, fire flat; contrast 4.45:1 on teal buttons.

## 3. Plan

### WS1 - Agent quality with Modal (CPU, ~1.5 h)  [needs: `modal setup`]
- `firescout/modal_eval.py`: one Modal function = one episode (CPU container, MuJoCo EGL/OSMesa), `.map()` over seeds.
- `eval.run_episode(seed, params)`: thresholds become params (VERIFY_CONF, BREATH_SAMPLES/RANGE, beam footprint, reject rule, mission length).
- Sweep ~40 configs x 150 floors on **train seeds 1000-1149**; pick best recall at precision >= 90%; report once on **held-out seeds 0-299**. Stashed fixes (tracker isolation, peak temp, no stranded verify) enter as sweep options, not guesses.
- Realism: breathing false-positive rate near warm decoys + range fall-off (so precision is honest).
- Add crew-with-TIC baseline (0.5 m/s, 4 m view) beside blind crawl.
- Test: `test_eval.py` param plumbing; acceptance = held-out results.md regenerated, numbers quoted everywhere.

### WS2 - Person detector actually trained (Modal, ~1 h, optional)
- Log every thermal blob + track features + truth label from sweep episodes -> dataset (CSV).
- Fit logistic regression offline (numpy), ship weights in `brain.py`, report held-out ROC/precision-recall.
- Pitch line: "trained on N blobs from M simulated floors on Modal; runs on the robot; IC feedback fine-tunes live."

### WS3 - IC console redesign (~2.5 h)  [impeccable: layout, clarify, colorize, animate, harden]
Layout (one hero, two supports):
- Top **command strip**: HANDLE FIRST (big), fire seats, radio state, crew clock. Replaces floating banners.
- Centre 3D view, auto-fit to building on load/resize; room names in all text ("R2 bedroom", not "(5.12, 5.35)").
- Left: STOP full-width solid red at top; RECALL / RESUME; "Search room" list; casualty cards with room name + reason + robot instruction.
- Right: camera (Thermal | Night IR | Colour, working), robot map vs plan, decision log as plain-English sentences (raw card on hover).
- **Demo controls** collapsed `<details>`: ignite, radio drop, ground-truth/eval, sound.
States: radio lost = red frame + desaturate + "data as of T+mm:ss, N queued"; feed/map age stamps; loading skeleton; empty states.
Colour system (single meaning each): fire red · possible amber · P1 red-orange marker · crew route green · robot white + cyan pulse, path dashed white-cyan · voice violet · true plan grey dashed.
Motion (purposeful only): route draws on when replanned; ring ping + card slide-in on new casualty; camera eases between views; log line highlight 1.5 s; fire light flicker tied to growth. `prefers-reduced-motion` respected.
Targets: buttons >= 48 px, text >= 13 px, contrast AA (fix teal), keyboard: S stop, R recall, 1-3 feeds, V view.
Robot's eye: lower exposure + spot, near plane 0.05, or replace with the IR feed full-size.
Fire: bigger sprite flame with texture + glow, smoke column that reaches the ceiling, ember sparks; check in Follow view.
Accept: re-run critique, target >= 30/40; screenshot 1600x900 + 1280x720.

### WS4 - Demo B: Isaac Sim on Modal (~3 h, staged; stop at any stage)
- B1 (must): clone isaac-lab-on-modal, run the Franka smoke test on L40S to prove the pipeline (~15-30 min first build).
- B2 (main): **digital-twin replay** - export a FireScout mission (layout, props, robot/occupant poses, fire growth per frame) from MuJoCo to JSON; Isaac script builds the USD stage with the *native* NVIDIA SimReady USD furniture we already use, walls/floor, robot, people; RTX-renders a follow camera + overhead -> frames -> ffmpeg video. Same agent decisions, photoreal render. Honest label: "replay of the live MuJoCo mission rendered in Isaac Sim".
- B3 (stretch): **closed loop in Isaac** - run `brain.py` inside the container on Isaac camera data (RGB + depth + semantic segmentation -> same thermal model as MuJoCo). Proves the agent is simulator-agnostic. Only if B2 lands before 09:30.
- Budget: L40S ~ $2/h; stop apps after each run; timeout on every function.

### WS5 - Deliverables (Sunday 09:00-12:30)
- Video <= 60 s, single take from the console (Win+Alt+R or OBS): scenario start -> P1 found (thermal + breathing) -> speaker tells awake person to walk out -> IC ignites shelf (unscripted) -> replan -> radio drop -> METHANE. Optional 10 s Isaac clip separately.
- 1-page PDF: fill numbers from WS1, print via Chrome headless.
- Pitch (5 min) in Canva: problem -> live demo -> how it decides -> results + limits -> safety -> ask.
- README + results.md final; secret scan before push.

### WS6 - Gemini (optional, ~1 h)
Off the decision loop: IC types an order -> Gemini -> structured command (robot still applies safety); after-action summary. Works without internet via buttons.

## 4. How we test the agent
- Unit: 56 tests (planner, perception, safety, speaker policy, IC override, scene/sensors).
- Evaluation: held-out floors (never used for tuning), Wilson CIs, two baselines, no-"clear" check, heat breaches, latency.
- Adversarial live checks (in the video/pitch): ignite object on the route, radio drop, IC STOP mid-verify, judge picks building/casualties/seed.
- Hidden challenge: judges choose a seed we never ran.

## 5. Order + timing (proposal)
1. WS3 P0s (feed bug, layout fit) - 30 min
2. WS1 Modal sweep running in background - 1.5 h
3. WS4-B1 Isaac smoke test running in background
4. WS3 rest - 2 h
5. WS4-B2 replay - 2 h
6. WS2 if time; WS6 optional
7. WS5 Sunday morning; feature freeze 11:30

## 6. Install / accounts needed
- `pip install modal` + `modal setup` (login, credits redeemed) - **you**
- Isaac: nothing local (all in the Modal container); optional `pip install uv` for the sample repo
- Video capture: Windows Game Bar (built in) or OBS
- Gemini (only if WS6): API key from AI Studio in an env var, never committed
