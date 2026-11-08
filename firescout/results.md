# FireScout evaluation (simulation, unseen procedurally generated floors)

Episodes: 30 floors · 37 casualties · 180 s robot mission each.
Baseline assumptions: crew enters after **150 s** (BA don + entry control), right-hand wall-following crawl at **0.3 m/s** with a 1.5 m sweep reach (zero visibility, no thermal camera; a crew with a TIC would be faster).
Ground truth is used only for scoring; the robot never sees it. These are simulated results, not real-world performance.

| Metric | Result |
|---|---|
| Time to first confirmed casualty, FireScout | 48 s (IQR 28-59) (13/21 floors) |
| Time to first casualty, wall-following crew baseline | 222 s (IQR 214-242) |
| Casualty recall (confirmed pins) | 14/37 = 38% (95% CI 24%-54%) |
| Pin precision | 14/15 = 93% (95% CI 70%-99%) |
| Unconfirmed "POSSIBLE" casualties flagged for crew | 90 |
| Heat-limit breaches (robot within 0.8 m of a fire) | 0 ticks |
| Any card/report claiming an area is clear | 0 (by design) |
| Thermal scan coverage (median) | 88% of mapped free space |
| Brain loop latency p50 / p95 (laptop CPU) | 2.6 / 40.4 ms |

## Per floor

| seed | casualties | found | FireScout t (s) | baseline t (s) | false pins | possible | breaches | scanned % |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 1 | 47.6 | 189.2 | 0 | 5 | 0 | 96.9 |
| 1 | 0 | 0 | None | None | 0 | 0 | 0 | 92.2 |
| 2 | 1 | 0 | None | 253.3 | 0 | 7 | 0 | 91.2 |
| 3 | 2 | 1 | 8.7 | 241.7 | 0 | 0 | 0 | 66.1 |
| 4 | 2 | 0 | None | 214.2 | 0 | 4 | 0 | 84.8 |
| 5 | 2 | 1 | 13.2 | 216.7 | 0 | 0 | 0 | 69.4 |
| 6 | 2 | 0 | 22.3 | 221.7 | 1 | 4 | 0 | 86.1 |
| 7 | 1 | 0 | None | 321.7 | 0 | 2 | 0 | 92.9 |
| 8 | 2 | 2 | 59.1 | 251.7 | 0 | 5 | 0 | 90.9 |
| 9 | 2 | 1 | 126.3 | 193.3 | 0 | 3 | 0 | 97.1 |
| 10 | 2 | 0 | None | 226.7 | 0 | 2 | 0 | 47.5 |
| 11 | 0 | 0 | None | None | 0 | 6 | 0 | 83.3 |
| 12 | 1 | 0 | None | 210.8 | 0 | 4 | 0 | 80.8 |
| 13 | 2 | 2 | 49.6 | 210.0 | 0 | 2 | 0 | 82.7 |
| 14 | 1 | 0 | None | 356.7 | 0 | 5 | 0 | 73.0 |
| 15 | 0 | 0 | None | None | 0 | 7 | 0 | 88.4 |
| 16 | 2 | 1 | 33.2 | 230.8 | 0 | 1 | 0 | 97.3 |
| 17 | 2 | 0 | None | 218.3 | 0 | 4 | 0 | 63.9 |
| 18 | 2 | 1 | 35.9 | 219.2 | 0 | 3 | 0 | 86.0 |
| 19 | 2 | 0 | None | 189.2 | 0 | 9 | 0 | 88.7 |
| 20 | 0 | 0 | None | None | 0 | 3 | 0 | 47.4 |
| 21 | 2 | 1 | 50.2 | 241.7 | 0 | 2 | 0 | 86.6 |
| 22 | 0 | 0 | None | None | 0 | 0 | 0 | 98.1 |
| 23 | 0 | 0 | None | None | 0 | 1 | 0 | 97.0 |
| 24 | 2 | 1 | 64.2 | 226.7 | 0 | 2 | 0 | 87.7 |
| 25 | 0 | 0 | None | None | 0 | 0 | 0 | 96.4 |
| 26 | 0 | 0 | None | None | 0 | 0 | 0 | 66.4 |
| 27 | 0 | 0 | None | None | 0 | 3 | 0 | 97.3 |
| 28 | 2 | 1 | 28.5 | 215.0 | 0 | 3 | 0 | 99.2 |
| 29 | 1 | 1 | 77.2 | 234.2 | 0 | 3 | 0 | 89.1 |
