"""Convert NVIDIA SimReady assets (OpenUSD, CC-BY-4.0) into decimated OBJ meshes MuJoCo can load.

Run once:  python assets.py      -> firescout/assets/<key>.obj + assets/meta.json
Source: https://huggingface.co/datasets/nvidia/simready-assets
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'assets')
CACHE = os.path.join(HERE, '..', '.cache_simready')

# key -> (SimReady asset, rgb, target max triangles)
ASSETS = {
    'bed': ('bed_hospital_standard_a01', (0.86, 0.87, 0.9), 60000),
    'sofa': ('sofa_soft_b01', (0.33, 0.4, 0.52), 15000),
    'chair': ('chair_hospital_a01', (0.28, 0.48, 0.58), 10000),
    'desk': ('desk_a01', (0.56, 0.42, 0.3), 8000),
    'cabinet': ('cabinet_file_a01', (0.58, 0.6, 0.63), 6000),
    'radiator': ('coolingSystem_radiator_a01', (0.93, 0.93, 0.91), 15000),
    'iv': ('IV_pole_a01', (0.82, 0.82, 0.85), 6000),
    'extinguisher': ('fireExtinguisher_abc_a03', (0.78, 0.08, 0.06), 6000),
    'bin': ('bin_waste_a01', (0.24, 0.25, 0.27), 4000),
    'tv': ('TV_a01', (0.08, 0.08, 0.1), 4000),
}


def _download(asset: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download('nvidia/simready-assets', f'{asset}/simready_usd/payloads/geometries.usd',
                           repo_type='dataset', local_dir=CACHE)


def _load_tris(path: str):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(path)
    mpu = UsdGeom.GetStageMetersPerUnit(st) or 1.0
    y_up = UsdGeom.GetStageUpAxis(st) == 'Y'
    verts, tris = [], []
    off = 0
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        counts = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        p = np.c_[np.array(pts, dtype=np.float64), np.ones(len(pts))] @ xf
        verts.append(p[:, :3])
        idx = np.array(idx)
        k = 0
        for c in counts:  # fan triangulation
            for j in range(1, c - 1):
                tris.append((off + idx[k], off + idx[k + j], off + idx[k + j + 1]))
            k += c
        off += len(pts)
    v = np.vstack(verts) * mpu
    if y_up:
        v = v[:, [0, 2, 1]] * np.array([1, -1, 1])
    return v, np.array(tris, dtype=np.int64)


def _decimate(v, f, max_tris):
    """Vertex clustering on a uniform grid until the triangle count fits (lazy but robust)."""
    size = float(np.ptp(v, axis=0).max())
    for div in (400, 250, 160, 110, 80, 55, 40):
        cell = size / div
        q = np.floor((v - v.min(0)) / cell).astype(np.int64)
        keys, inv = np.unique(q, axis=0, return_inverse=True)
        inv = inv.ravel()
        nv = np.zeros((len(keys), 3))
        np.add.at(nv, inv, v)
        nv /= np.bincount(inv, minlength=len(keys))[:, None]
        nf = inv[f]
        nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
        nf = np.unique(np.sort(nf, axis=1), axis=0) if len(nf) > max_tris * 4 else nf
        if len(nf) <= max_tris:
            return nv, nf
    return nv, nf


def convert(key: str) -> dict:
    asset, rgb, max_tris = ASSETS[key]
    v, f = _load_tris(_download(asset))
    v, f = _decimate(v, f, max_tris)
    lo, hi = v.min(0), v.max(0)
    v = v - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])  # centre in xy, sit on floor
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{key}.obj'), 'w') as fh:
        fh.write(f'# {asset} (NVIDIA SimReady, CC-BY-4.0), decimated\n')
        fh.writelines(f'v {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in v)
        fh.writelines(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in f)
    size = (hi - lo).round(3).tolist()
    return {'asset': asset, 'rgb': rgb, 'size': size, 'tris': int(len(f))}


def load_meta() -> dict:
    p = os.path.join(OUT, 'meta.json')
    return json.load(open(p)) if os.path.exists(p) else {}


if __name__ == '__main__':
    meta = load_meta()
    for key in sys.argv[1:] or ASSETS:
        try:
            meta[key] = convert(key)
            print(key, meta[key])
        except Exception as e:  # keep going: a missing asset falls back to a primitive in scene.py
            print(key, 'FAILED', e)
    json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
