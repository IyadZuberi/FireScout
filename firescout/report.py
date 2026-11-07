"""METHANE situation report (JESIP format) built only from the robot's own map and tracks."""
from __future__ import annotations

import numpy as np

from brain import NO_GO_C, RES, Brain


def _xy(p):
    return f'({p[0]:.1f},{p[1]:.1f})'


def methane(b: Brain, t: float | None = None) -> str:
    t = b.t if t is None else t
    g = b.grid
    mm, ss = divmod(int(t), 60)
    x, y = g.to_world(*b.start_cell)
    free = max(1, int((g.occ == 0).sum()))
    pct = 100.0 * g.scanned.sum() / free
    fires = sorted(b.fires, key=lambda f: -f[2])
    hot_cells = int((g.heat > NO_GO_C).sum())
    gas_max = float(g.gas.max())
    gas_at = g.to_world(*np.unravel_index(int(np.argmax(g.gas)), g.gas.shape))
    pinned = [tr for tr in b.tracker.tracks if tr.state == 'pinned']
    possible = [tr for tr in b.tracker.tracks
                if tr.kind == 'warm' and tr.state in ('verify', 'candidate') and tr.conf >= 0.4]
    rescue = f'{len(b.rescue[1]) * RES:.1f} m (visit order {[pinned[k].id for k in b.rescue[0] if k < len(pinned)]})' \
        if b.rescue else 'not yet planned'
    exit_ = f'{len(b.exit_path) * RES:.1f} m back to entry' if b.exit_path else 'unknown (no safe route found)'
    if pinned:
        n = (f'N: {len(pinned)} confirmed casualt{"y" if len(pinned) == 1 else "ies"} '
             + '; '.join(f'#{tr.id} at {_xy(tr.xy)} conf {tr.conf:.2f}, breathing {tr.breath:.2f}' for tr in pinned))
    else:
        n = 'N: 0 confirmed casualties'
    if possible:
        n += ' | POSSIBLE (unconfirmed): ' + '; '.join(f'#{tr.id} at {_xy(tr.xy)} conf {tr.conf:.2f}'
                                                       for tr in possible)
    if not pinned and not possible:
        n += f' — none detected (scanned {pct:.0f}% of reachable area) — human search still required'
    else:
        n += f' — scanned {pct:.0f}% of reachable area, human search still required'
    return '\n'.join([
        f'M: FireScout recon report T+{mm:02d}:{ss:02d} (robot sitrep, not a major-incident declaration)',
        f'E: Entry point {_xy((x, y))}, single floor; robot mode {b.mode}',
        'T: Structure fire, ' + (', '.join(f'seat at {_xy(f)} ~{f[2]:.0f}°C' for f in fires[:3]) or 'no fire seat seen yet'),
        f'H: No-go >{NO_GO_C:.0f}°C: {hot_cells * RES * RES:.1f} m²; max CO {gas_max:.0f} ppm at {_xy(gas_at)}',
        f'A: Rescue route {rescue}; exit route {exit_}',
        n,
        f'E: BA crew for rescue; ambulance for {len(pinned)} confirmed + {len(possible)} possible casualties',
    ])
