# FireScout evaluation (simulation, unseen procedurally generated floors)

Episodes: 30 floors · 37 casualties · 180 s robot mission each.
Baseline assumptions: crew enters after **150 s** (BA don + entry control), right-hand wall-following crawl at **0.3 m/s** with a 1.5 m sweep reach (zero visibility, no thermal camera; a crew with a TIC would be faster).
Ground truth is used only for scoring; the robot never sees it. These are simulated results, not real-world performance.

| Metric | Result |
|---|---|
| Time to first confirmed casualty, FireScout | 37 s (IQR 22-66) (8/21 floors) |
| Time to first casualty, wall-following crew baseline | 229 s (IQR 216-282) |
| Casualty recall (confirmed pins) | 11/37 = 30% (95% CI 17%-46%) |
| Pin precision | 11/12 = 92% (95% CI 65%-99%) |
| Unconfirmed "POSSIBLE" casualties flagged for crew | 83 |
| Heat-limit breaches (robot within 0.8 m of a fire) | 0 ticks |
| Any card/report claiming an area is clear | VIOLATION |
| Thermal scan coverage (median) | 87% of mapped free space |
| Brain loop latency p50 / p95 (laptop CPU) | 3.2 / 50.2 ms |

## Per floor

| seed | casualties | found | FireScout t (s) | baseline t (s) | false pins | possible | breaches | scanned % |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 2 | 58.3 | 282.5 | 0 | 2 | 0 | 86.3 |
| 1 | 0 | 0 | None | None | 0 | 2 | 0 | 93.2 |
| 2 | 1 | 0 | None | 288.3 | 0 | 5 | 0 | 83.0 |
| 3 | 2 | 2 | 22.3 | 242.5 | 1 | 4 | 0 | 97.7 |
| 4 | 2 | 0 | None | 295.0 | 0 | 1 | 0 | 68.9 |
| 5 | 2 | 0 | None | 215.8 | 0 | 2 | 0 | 85.1 |
| 6 | 2 | 1 | 43.3 | 220.0 | 0 | 2 | 0 | 94.7 |
| 7 | 1 | 1 | 88.3 | 321.7 | 0 | 1 | 0 | 98.5 |
| 8 | 2 | 0 | None | 226.7 | 0 | 2 | 0 | 69.3 |
| 9 | 2 | 1 | 21.8 | 240.0 | 0 | 2 | 0 | 93.1 |
| 10 | 2 | 0 | None | 192.5 | 0 | 5 | 0 | 92.9 |
| 11 | 0 | 0 | None | None | 0 | 6 | 0 | 83.3 |
| 12 | 1 | 0 | None | 321.7 | 0 | 4 | 0 | 84.8 |
| 13 | 2 | 0 | None | 210.0 | 0 | 4 | 0 | 77.2 |
| 14 | 1 | 0 | None | 324.2 | 0 | 6 | 0 | 67.9 |
| 15 | 0 | 0 | None | None | 0 | 4 | 0 | 87.8 |
| 16 | 2 | 2 | 29.8 | 246.7 | 0 | 1 | 0 | 91.7 |
| 17 | 2 | 0 | None | 216.7 | 0 | 4 | 0 | 64.7 |
| 18 | 2 | 0 | None | 219.2 | 0 | 4 | 0 | 57.6 |
| 19 | 2 | 0 | None | 229.2 | 0 | 3 | 0 | 86.0 |
| 20 | 0 | 0 | None | None | 0 | 1 | 0 | 95.9 |
| 21 | 2 | 0 | None | 201.7 | 0 | 5 | 0 | 84.0 |
| 22 | 0 | 0 | None | None | 0 | 0 | 0 | 98.6 |
| 23 | 0 | 0 | None | None | 0 | 1 | 0 | 96.3 |
| 24 | 2 | 0 | None | 194.2 | 0 | 3 | 0 | 66.5 |
| 25 | 0 | 0 | None | None | 0 | 0 | 0 | 89.8 |
| 26 | 0 | 0 | None | None | 0 | 0 | 0 | 65.9 |
| 27 | 0 | 0 | None | None | 0 | 4 | 0 | 98.1 |
| 28 | 2 | 1 | 95.4 | 214.2 | 0 | 4 | 0 | 98.6 |
| 29 | 1 | 1 | 8.1 | 235.8 | 0 | 1 | 0 | 96.7 |
