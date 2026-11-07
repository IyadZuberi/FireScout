# FireScout evaluation (simulation, unseen procedurally generated floors)

Episodes: 30 floors · 25 casualties · 180 s robot mission each.
Baseline assumptions: crew enters after **150 s** (BA don + entry control), right-hand wall-following crawl at **0.3 m/s** with a 1.5 m sweep reach (zero visibility, no thermal camera; a crew with a TIC would be faster).
Ground truth is used only for scoring; the robot never sees it. These are simulated results, not real-world performance.

| Metric | Result |
|---|---|
| Time to first confirmed casualty, FireScout | 17 s (IQR 5-25) (10/18 floors) |
| Time to first casualty, wall-following crew baseline | 232 s (IQR 207-285) |
| Casualty recall (confirmed pins) | 9/25 = 36% (95% CI 20%-55%) |
| Pin precision | 9/9 = 100% (95% CI 70%-100%) |
| Unconfirmed "POSSIBLE" casualties flagged for crew | 28 |
| Heat-limit breaches (robot within 0.8 m of a fire) | 0 ticks |
| Any card/report claiming an area is clear | 0 (by design) |
| Thermal scan coverage (median) | 101% of mapped free space |
| Brain loop latency p50 / p95 (laptop CPU) | 2.2 / 48.4 ms |

## Per floor

| seed | casualties | found | FireScout t (s) | baseline t (s) | false pins | possible | breaches | scanned % |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 1 | 18.6 | 190.0 | 0 | 1 | 0 | 67.8 |
| 1 | 0 | 0 | None | None | 0 | 1 | 0 | 102.4 |
| 2 | 0 | 0 | None | None | 0 | 1 | 0 | 106.9 |
| 3 | 0 | 0 | None | None | 0 | 1 | 0 | 111.7 |
| 4 | 2 | 0 | None | 219.2 | 0 | 1 | 0 | 100.7 |
| 5 | 1 | 1 | 25.2 | 340.0 | 0 | 0 | 0 | 113.0 |
| 6 | 0 | 0 | None | None | 0 | 2 | 0 | 104.4 |
| 7 | 1 | 0 | None | 191.7 | 0 | 1 | 0 | 82.1 |
| 8 | 0 | 0 | None | None | 0 | 1 | 0 | 104.6 |
| 9 | 0 | 0 | None | None | 0 | 0 | 0 | 116.3 |
| 10 | 0 | 0 | None | None | 0 | 0 | 0 | 109.1 |
| 11 | 1 | 1 | 24.8 | 287.5 | 0 | 1 | 0 | 108.5 |
| 12 | 0 | 0 | None | None | 0 | 1 | 0 | 111.1 |
| 13 | 2 | 0 | None | 189.2 | 0 | 0 | 0 | 72.4 |
| 14 | 0 | 0 | None | None | 0 | 0 | 0 | 70.1 |
| 15 | 2 | 1 | 4.9 | 232.5 | 0 | 1 | 0 | 109.3 |
| 16 | 1 | 0 | None | 327.5 | 0 | 0 | 0 | 78.7 |
| 17 | 1 | 1 | 2.3 | 252.5 | 0 | 1 | 0 | 72.1 |
| 18 | 2 | 0 | None | 223.3 | 0 | 1 | 0 | 72.9 |
| 19 | 1 | 1 | 5.7 | 301.7 | 0 | 1 | 0 | 80.9 |
| 20 | 1 | 0 | None | 322.5 | 0 | 1 | 0 | 95.4 |
| 21 | 0 | 0 | None | None | 0 | 1 | 0 | 87.9 |
| 22 | 1 | 0 | None | 207.5 | 0 | 1 | 0 | 104.2 |
| 23 | 1 | 1 | 176.7 | 278.3 | 0 | 0 | 0 | 79.4 |
| 24 | 0 | 0 | None | None | 0 | 1 | 0 | 110.1 |
| 25 | 2 | 1 | 51.4 | 201.7 | 0 | 2 | 0 | 101.1 |
| 26 | 1 | 1 | 3.6 | 275.0 | 0 | 0 | 0 | 97.0 |
| 27 | 2 | 0 | 15.4 | 232.5 | 0 | 3 | 0 | 103.3 |
| 28 | 0 | 0 | None | None | 0 | 2 | 0 | 87.0 |
| 29 | 1 | 0 | None | 206.7 | 0 | 2 | 0 | 82.9 |
