from eval import baseline_time, run_episode, wilson
from scene import Layout, Thing

OUTER = [(0, 0, 12, 0), (0, 8, 12, 8), (0, 0, 0, 8), (12, 0, 12, 8)]


def test_wilson_ci_bounds():
    lo, hi = wilson(5, 5)
    assert hi == 1.0 and lo > 0.4
    lo, hi = wilson(0, 5)
    assert lo == 0.0 and hi < 0.6


def test_baseline_finds_reachable_casualty():
    L = Layout(12, 8, OUTER, [Thing('c', 'casualty', (6.0, 7.2), (1.7, 0.5, 0.3), 36, True)], (1.0, 4.0), 0)
    t = baseline_time(L, crew_delay_s=150.0, speed_mps=0.3)
    assert t is not None and t > 150.0


def test_baseline_none_without_casualty():
    L = Layout(12, 8, OUTER, [], (1.0, 4.0), 0)
    assert baseline_time(L) is None


def test_run_episode_keys():
    r = run_episode(1, t_max=3.0)
    for k in ('seed', 't_first_casualty', 'found', 'total_casualties', 'false_pins', 'heat_breaches',
              'scanned_pct', 'loop_ms_p50', 'loop_ms_p95', 'baseline_t'):
        assert k in r
