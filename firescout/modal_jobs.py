"""Offline work on Modal (NOT part of the robot's decision loop - the robot runs on its own computer).

  modal run modal_jobs.py::smoke                  # 2 floors, proves the cloud image works (~2 min, cents)
  modal run modal_jobs.py::sweep --n 120          # tune agent thresholds on TRAIN floors 1000+
  modal run modal_jobs.py::train --n 300          # train the person detector on simulated tracks, writes detector_weights.json
  modal run modal_jobs.py::evaluate --n 300       # final numbers on HELD-OUT floors 0..n-1 -> results_modal.md

CPU containers, MuJoCo rendering headless via OSMesa (the sensors only need 160x120 images). Every function has a
timeout so a stuck job cannot burn credits.
"""
from __future__ import annotations

import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('libosmesa6', 'libgl1', 'libglib2.0-0')
         .pip_install('mujoco==3.3.7', 'numpy<2.3', 'opencv-python-headless')
         .env({'MUJOCO_GL': 'osmesa', 'PYOPENGL_PLATFORM': 'osmesa'})
         .add_local_dir(HERE, '/root/firescout', ignore=['web', '__pycache__', '*.rrd', '*.jsonl', 'assets/audio']))
app = modal.App('firescout-offline', image=image)
TRAIN0, HELD_OUT = 1000, 0   # tuning/training never sees the floors used for the reported numbers


@app.function(cpu=2, memory=2048, timeout=900, retries=1)
def episode(seed: int, params: dict | None = None, weights: list | None = None, dataset: bool = False) -> dict:
    import sys
    sys.path.insert(0, '/root/firescout')
    import eval as E
    return E.run_episode(seed, 180.0, params=params, weights=weights, dataset=dataset)


def _score(rows):
    found = sum(r['found'] for r in rows)
    total = sum(r['total_casualties'] for r in rows)
    fp = sum(r['false_pins'] for r in rows)
    return found / max(1, total), found / max(1, found + fp)


@app.local_entrypoint()
def smoke():
    for r in episode.map([0, 1]):
        print({k: v for k, v in r.items() if k != 'rows'})


@app.local_entrypoint()
def sweep(n: int = 120):
    grid = [{'VERIFY_CONF': vc, 'BREATH_SAMPLES': bs, 'BREATH_RANGE': br, 'VERIFY_TIMEOUT': vt}
            for vc in (0.3, 0.4, 0.5) for bs in (10, 20) for br in (2.5, 3.0) for vt in (10.0, 20.0)]
    seeds = list(range(TRAIN0, TRAIN0 + n))
    best = None
    for p in grid:
        rows = list(episode.map(seeds, kwargs={'params': p}))
        rec, prec = _score(rows)
        print(f'{p} recall {rec:.2f} precision {prec:.2f}')
        if prec >= 0.90 and (best is None or rec > best[1]):
            best = (p, rec, prec)
    print('BEST (precision >= 0.90):', best)
    with open(os.path.join(HERE, 'best_params.json'), 'w') as fh:
        json.dump({'params': best[0] if best else None, 'recall': best and best[1], 'precision': best and best[2],
                   'floors': n, 'seeds': [TRAIN0, TRAIN0 + n]}, fh, indent=1)


@app.local_entrypoint()
def train(n: int = 300):
    import numpy as np
    rows = [x for r in episode.map(range(TRAIN0, TRAIN0 + n), kwargs={'dataset': True}) for x in r['rows']]
    X = np.array([x['f'] for x in rows], float)
    y = np.array([x['y'] for x in rows], float)
    w = np.array([-4.0, 1.5, 1.5, 1.0, 2.0, 3.0])
    for _ in range(3000):  # logistic regression, L2, full-batch gradient descent (tiny model: 6 weights)
        p = 1 / (1 + np.exp(-(X @ w)))
        w -= 0.5 * (X.T @ (p - y) / len(y) + 1e-3 * w)
    p = 1 / (1 + np.exp(-(X @ w)))
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    print(f'{len(y)} tracks from {n} floors, {int(y.sum())} real people; train accuracy {acc:.3f}; w={w.round(3)}')
    with open(os.path.join(HERE, 'detector_weights.json'), 'w') as fh:
        json.dump({'w': w.round(6).tolist(), 'source': f'trained on {len(y)} simulated tracks from {n} floors (Modal)',
                   'train_accuracy': acc}, fh, indent=1)


@app.local_entrypoint()
def evaluate(n: int = 300, use_best: bool = True, use_trained: bool = True):
    import sys
    sys.path.insert(0, HERE)
    import eval as E
    params = weights = None
    if use_best and os.path.exists(os.path.join(HERE, 'best_params.json')):
        params = json.load(open(os.path.join(HERE, 'best_params.json')))['params']
    if use_trained and os.path.exists(os.path.join(HERE, 'detector_weights.json')):
        weights = json.load(open(os.path.join(HERE, 'detector_weights.json')))['w']
    rows = list(episode.map(range(HELD_OUT, HELD_OUT + n), kwargs={'params': params, 'weights': weights}))
    for r in rows:
        r.pop('rows', None)
    md = E.summarize(rows)
    with open(os.path.join(HERE, 'results_modal.md'), 'w', encoding='utf-8') as fh:
        fh.write(f'Held-out floors 0..{n - 1}; params {params}; detector {"trained" if weights else "prior"}\n\n' + md)
    rec, prec = _score(rows)
    print(f'held-out recall {rec:.2f} precision {prec:.2f} -> results_modal.md')
