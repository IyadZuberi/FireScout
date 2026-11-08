"""Convert Quaternius Universal Base Characters (CC0 glTF) into posed casualty meshes for MuJoCo.

Skinned glTF -> linear-blend skinning with a custom pose -> rotate onto the floor -> OBJ + texture.
Run once: python humans.py "<path to 'Base Characters/Godot - UE'>"
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / 'assets'
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


class Gltf:
    def __init__(self, path: Path):
        self.g = json.loads(path.read_text())
        self.bin = (path.parent / self.g['buffers'][0]['uri']).read_bytes()
        self.parent = {c: i for i, n in enumerate(self.g['nodes']) for c in n.get('children', [])}
        self.names = {n.get('name'): i for i, n in enumerate(self.g['nodes'])}
        self.override: dict[int, np.ndarray] = {}  # accessor index -> replacement data (e.g. loosened clothes)

    def acc(self, i):
        if i in self.override:
            return self.override[i]
        a = self.g['accessors'][i]
        v = self.g['bufferViews'][a['bufferView']]
        n, dt = NC[a['type']], CT[a['componentType']]
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = v.get('byteStride', 0) or n * np.dtype(dt).itemsize
        raw = np.frombuffer(self.bin, np.uint8, count=stride * (a['count'] - 1) + n * np.dtype(dt).itemsize, offset=off)
        out = np.lib.stride_tricks.as_strided(raw, (a['count'], n * np.dtype(dt).itemsize), (stride, 1))
        return np.ascontiguousarray(out).view(dt).reshape(a['count'], n)

    def local(self, i, rot_delta=None):
        n = self.g['nodes'][i]
        M = np.eye(4)
        if 'matrix' in n:
            M = np.array(n['matrix']).reshape(4, 4).T
        else:
            R = quat_mat(n.get('rotation', [0, 0, 0, 1]))
            if rot_delta is not None:
                R = R @ rot_delta
            M[:3, :3] = R * np.array(n.get('scale', [1, 1, 1]))
            M[:3, 3] = n.get('translation', [0, 0, 0])
        return M

    def world(self, pose):
        cache = {}

        def w(i):
            if i not in cache:
                L = self.local(i, pose.get(self.g['nodes'][i].get('name')))
                cache[i] = (w(self.parent[i]) @ L) if i in self.parent else L
            return cache[i]
        return w

    def skinned(self, pose):
        skin = self.g['skins'][0]
        ibm = self.acc(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
        w = self.world(pose)
        J = np.stack([w(j) @ ibm[k] for k, j in enumerate(skin['joints'])])
        parts = []
        for mi, mesh in enumerate(self.g['meshes']):
            for p in mesh['primitives']:
                at = p['attributes']
                P = self.acc(at['POSITION']).astype(float)
                jt = self.acc(at['JOINTS_0']).astype(int)
                wt = self.acc(at['WEIGHTS_0']).astype(float)
                if wt.max() > 1.5:  # normalised ints
                    wt = wt / wt.sum(1, keepdims=True)
                Ph = np.c_[P, np.ones(len(P))]
                M = np.einsum('vk,vkij->vij', wt, J[jt])
                V = np.einsum('vij,vj->vi', M, Ph)[:, :3]
                uv = self.acc(at['TEXCOORD_0']).astype(float)
                F = self.acc(p['indices']).reshape(-1, 3)
                mat = self.g['materials'][p['material']]
                tex = mat['pbrMetallicRoughness'].get('baseColorTexture', {}).get('index')
                img = self.g['images'][self.g['textures'][tex]['source']]['uri'] if tex is not None else None
                parts.append((mat['name'], V, uv, F, img))
        return parts


def joint_pos(g, pose, name):
    return g.world(pose)(g.names[name])[:3, 3]


def _p(*rots):
    """Compose local rotations [(axis, deg), ...] left to right."""
    R = np.eye(3)
    for ax, deg in rots:
        R = R @ axis_angle({'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[ax], deg)
    return R


# Local-frame bone rotations. Found with probe(): upperarm z lowers the arm (left -, right +),
# upperarm x swings it forward, thigh x<0 flexes the hip, calf x>0 bends the knee.
POSES = {
    'supine': {'upperarm_l': _p(('z', -72)), 'upperarm_r': _p(('z', 72)),
               'lowerarm_l': _p(('z', 12)), 'lowerarm_r': _p(('z', -12)),
               'thigh_l': _p(('z', 4)), 'thigh_r': _p(('z', -4))},
    'prone': {'upperarm_l': _p(('z', 55), ('x', 12)), 'lowerarm_l': _p(('z', 35)),
              'upperarm_r': _p(('z', 78), ('x', 8)), 'thigh_l': _p(('z', -18)), 'thigh_r': _p(('z', -3))},
    'recovery': {'upperarm_r': _p(('z', 55), ('x', 75)), 'lowerarm_r': _p(('z', -60)),
                 'upperarm_l': _p(('z', -25), ('x', 85)), 'thigh_r': _p(('x', -75)), 'calf_r': _p(('x', 95)),
                 'thigh_l': _p(('x', -15)), 'calf_l': _p(('x', 25))},
    'standing': {'upperarm_l': _p(('z', -76)), 'upperarm_r': _p(('z', 76)),
                 'lowerarm_l': _p(('x', 15)), 'lowerarm_r': _p(('x', 15))},
    'kneel': {'upperarm_l': _p(('z', -70), ('x', 25)), 'upperarm_r': _p(('z', 70), ('x', 25)),
              'lowerarm_l': _p(('x', 40)), 'lowerarm_r': _p(('x', 40)),
              'thigh_l': _p(('x', -20)), 'thigh_r': _p(('x', -20)), 'calf_l': _p(('x', 105)), 'calf_r': _p(('x', 105))},
}
for _k in range(4):  # 4-frame walk cycle (stooped slightly: staying low under smoke)
    _ph = math.sin(_k * math.pi / 2)
    _pc = math.cos(_k * math.pi / 2)
    POSES[f'walk{_k}'] = {
        'upperarm_l': _p(('z', -74), ('x', 22 * _ph)), 'upperarm_r': _p(('z', 74), ('x', -22 * _ph)),
        'lowerarm_l': _p(('x', 25)), 'lowerarm_r': _p(('x', 25)),
        'thigh_l': _p(('x', -26 * _ph)), 'thigh_r': _p(('x', 26 * _ph)),
        'calf_l': _p(('x', 10 + 30 * max(0.0, -_pc))), 'calf_r': _p(('x', 10 + 30 * max(0.0, _pc))),
        'spine_02': _p(('x', 12))}
UPRIGHT = ('standing', 'kneel') + tuple(f'walk{k}' for k in range(4))
# glTF (x=left, y=up, z=front) -> world with the body lying along +x (head at +x). Rows = world axes.
ORIENT = {
    **{k: np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float) for k in UPRIGHT},  # upright, facing +x
    'supine': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float),    # front faces up
    'prone': np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float),     # front faces down
    'recovery': np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float),  # lying on the left side, facing -y
}
SKIN_BONES = ('Head', 'neck', 'hand', 'index', 'middle', 'pinky', 'ring', 'thumb')
LOWER = ('pelvis', 'thigh', 'calf')
FEET = ('foot', 'ball')
OUTFITS = {  # (top rgb, bottom rgb, sock rgb): nightwear / day clothes seen in care homes
    'm0': ((0.42, 0.52, 0.66), (0.2, 0.24, 0.34), (0.35, 0.33, 0.3)),
    'm1': ((0.52, 0.2, 0.22), (0.45, 0.45, 0.43), (0.2, 0.2, 0.22)),
    'f0': ((0.3, 0.34, 0.48), (0.3, 0.34, 0.48), (0.7, 0.68, 0.65)),  # navy dressing gown
    'f1': ((0.55, 0.5, 0.7), (0.22, 0.22, 0.28), (0.3, 0.28, 0.3)),
}
HAIR = {'grey': (0.78, 0.77, 0.74), 'dark': (0.3, 0.22, 0.16)}


def vertex_class(g, prim_attrs):
    """0 skin, 1 top, 2 bottoms, 3 socks - from each vertex's dominant bone."""
    skin = g.g['skins'][0]
    jt = g.acc(prim_attrs['JOINTS_0']).astype(int)
    wt = g.acc(prim_attrs['WEIGHTS_0']).astype(float)
    names = [g.g['nodes'][skin['joints'][j]]['name'] for j in jt[np.arange(len(jt)), wt.argmax(1)]]
    return np.array([0 if n.startswith(SKIN_BONES) else 3 if n.startswith(FEET) else
                     2 if n.startswith(LOWER) else 1 for n in names])


def paint_texture(src_png: Path, uv, F, cls, outfit, size=1024):
    """Recolour clothing regions of the base texture, keeping its shading (folds, AO)."""
    from PIL import Image, ImageDraw
    img = np.asarray(Image.open(src_png).convert('RGB').resize((size, size), Image.LANCZOS), float) / 255
    lab = Image.new('L', (size, size), 0)
    dr = ImageDraw.Draw(lab)
    tri_cls = np.array([np.bincount(cls[f], minlength=4).argmax() for f in F])
    for c in (1, 2, 3, 0):  # skin last so faces/hands win on shared seams
        for f in F[tri_cls == c]:
            dr.polygon([(uv[v, 0] * size, uv[v, 1] * size) for v in f], fill=c, outline=c)
    lab = np.asarray(lab)
    lum = img.mean(2)
    out = img.copy()
    for c in (1, 2, 3):
        m = lab == c
        if m.any():
            shade = np.clip(lum[m] / np.median(lum[m]), 0.55, 1.35)[:, None]
            out[m] = np.clip(np.array(outfit[c - 1]) * shade, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def loosen(g: Gltf, prim: dict, cls: np.ndarray, iters: int = 30, inflate: float = 0.022) -> None:
    """Clothes hang loose: smooth away body contours under the clothing (Laplacian on welded vertices) and push
    the cloth out a little. Writes the result as an accessor override, so every pose uses it."""
    at = prim['attributes']
    P = g.acc(at['POSITION']).astype(float)
    N = g.acc(at['NORMAL']).astype(float)
    F = g.acc(prim['indices']).reshape(-1, 3)
    key = np.round(P / 1e-4).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.ravel()
    n = weld.max() + 1
    Q = np.zeros((n, 3))
    np.add.at(Q, weld, P)
    Q /= np.bincount(weld, minlength=n)[:, None]
    e = np.concatenate([weld[F[:, [0, 1]]], weld[F[:, [1, 2]]], weld[F[:, [2, 0]]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(float)
    cloth = np.zeros(n, bool)
    cloth[weld[cls > 0]] = True
    for _ in range(iters):
        avg = np.zeros((n, 3))
        np.add.at(avg, e[:, 0], Q[e[:, 1]])
        avg /= np.maximum(deg, 1)[:, None]
        Q[cloth] = 0.5 * Q[cloth] + 0.5 * avg[cloth]
    out = Q[weld] + inflate * N * (cls > 0)[:, None]
    g.override[at['POSITION']] = out.astype(np.float32)


def write_obj(path: Path, V, uv, F):
    with open(path, 'w') as fh:
        fh.write(''.join(f'v {x:.3f} {y:.3f} {z:.3f}\n' for x, y, z in V))
        fh.write(''.join(f'vt {u:.5f} {1 - v:.5f}\n' for u, v in uv))
        fh.write(''.join(f'f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n' for a, b, c in F))


def build(base: Path, hair_dir: Path, out: Path = OUT) -> dict:
    from PIL import Image
    meta = {}
    for sex, hair_name in (('m', 'Hair_SimpleParted'), ('f', 'Hair_Buns')):
        body = Gltf(base / f'Superhero_{"Male" if sex == "m" else "Female"}_FullBody.gltf')
        hair = Gltf(hair_dir / f'{hair_name}.gltf')
        bp = next(p for m in body.g['meshes'] for p in m['primitives']
                  if body.g['materials'][p['material']]['name'].startswith('MI_Superhero'))
        cls = vertex_class(body, bp['attributes'])
        loosen(body, bp, cls)
        stand_shift = None
        for pose_name, pose in POSES.items():
            parts = [p for p in body.skinned(pose) if p[0].startswith('MI_Superhero')]
            parts += [('hair',) + p[1:] for p in hair.skinned(pose)]
            R = ORIENT[pose_name]
            allV = np.vstack([p[1] for p in parts]) @ R.T
            lo, hi = allV.min(0), allV.max(0)
            shift = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            if pose_name in UPRIGHT:  # one origin for all upright frames (feet centre), so animation does not jitter
                stand_shift = stand_shift if stand_shift is not None else shift
                shift = np.array([stand_shift[0], stand_shift[1], lo[2]])
            for name, V, uv, F, img in parts:
                key = f'human_{sex}_{pose_name}' + ('_hair' if name == 'hair' else '')
                write_obj(out / f'{key}.obj', V @ R.T - shift, uv, F)
            meta[f'{sex}_{pose_name}'] = {'size': [round(float(x), 3) for x in hi - lo]}
        tex_src = base.parent / 'Textures' / ('T_Superhero_Male_Ligh.png' if sex == 'm' else
                                              'T_Superhero_Female_Light_BaseColor.png')
        uv = body.acc(bp['attributes']['TEXCOORD_0']).astype(float)
        F = body.acc(bp['indices']).reshape(-1, 3)
        for k, outfit in OUTFITS.items():
            if k[0] == sex:
                paint_texture(tex_src, uv, F, cls, outfit).save(out / f'tex_human_{k}.png', optimize=True)
    Image.open(hair_dir / 'T_Hair_1_BaseColor.png').convert('RGB').resize((256, 256)).save(out / 'tex_hair.png')
    (out / 'humans.json').write_text(json.dumps({'source': 'Quaternius Universal Base Characters (CC0 1.0)',
                                                 'variants': meta, 'outfits': list(OUTFITS), 'hair': HAIR}, indent=1))
    return meta


def probe(g):
    """Print where the hand/foot go when a bone rotates about each local axis (used to fill POSES)."""
    for bone, tip in [('upperarm_l', 'hand_l'), ('upperarm_r', 'hand_r'), ('thigh_l', 'foot_l'),
                      ('calf_l', 'foot_l'), ('lowerarm_l', 'hand_l')]:
        base = joint_pos(g, {}, tip)
        for ax in np.eye(3):
            p = joint_pos(g, {bone: axis_angle(ax, 60)}, tip)
            print(bone, ax, np.round(p - base, 2))


if __name__ == '__main__':
    root = Path(sys.argv[1])  # the inner ".../Universal Base Characters[Standard]" folder
    print(build(root / 'Base Characters' / 'Godot - UE',
                root / 'Hairstyles' / 'Rigged to Head Bone' / 'glTF (Godot -Unreal)'))
