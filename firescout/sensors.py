"""Noisy sensor models: the ONLY bridge between MuJoCo ground truth and the robot brain.

Real-part equivalents: FLIR Lepton 3.5 (thermal, 160x120, 57 deg HFOV), 60 GHz mmWave
radar (ranges + breathing micro-motion), electrochemical CO cell, IMU/odometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from scene import MESH_GROUP, SMOKE_GROUP, Layout

AMBIENT_C = 20.0
WALL_C = 22.0
SCOUT_C = 30.0
FIRE_MIN_C = 150.0
MAX_RANGE = 6.0
BREATH_RANGE = 3.0
COMMS_RANGE = 9.0
H, W = 120, 160


@dataclass
class Observation:
    t: float
    pose: tuple[float, float, float]
    thermal: np.ndarray
    depth: np.ndarray
    ranges: np.ndarray
    gas_ppm: float
    comms: bool
    self_temp_c: float
    battery: float
    rgb: np.ndarray | None = None
    breath: float | None = None
    o2_pct: float = 20.9
    voices: list = None  # [(bearing rad relative to heading, level dB)] from the 4-mic array

    def __post_init__(self):
        if self.voices is None:
            self.voices = []


class Sensors:
    def __init__(self, model, data, layout: Layout, seed: int = 0, render_rgb: bool = True):
        self.m, self.d, self.layout = model, data, layout
        self.rng = np.random.default_rng(seed)
        self.r = mujoco.Renderer(model, H, W)
        self.render_rgb = render_rgb
        self.force_comms_drop = False
        self.self_temp = 25.0
        self.battery = 1.0
        self._last_t = data.time
        self.scout_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'scout')
        self.thing_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'thing{i}')
                           for i in range(len(layout.things))]
        self.temp_by_thing = {i: t.temp_c for i, t in enumerate(layout.things)}
        # geom -> thing index (or -1 wall/floor, -2 scout)
        body_to_thing = {b: i for i, b in enumerate(self.thing_body)}
        gb = model.geom_bodyid
        self.geom_thing = np.array([body_to_thing.get(int(b), -2 if b == self.scout_body else -1)
                                    for b in gb], dtype=np.int32)
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'thermal')
        self.vopt = mujoco.MjvOption()
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0  # sensor sees the world, not debug rays
        self.vopt_rgb = mujoco.MjvOption()
        self.vopt_rgb.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        self.vopt.geomgroup[SMOKE_GROUP] = 0  # long-wave IR sees through smoke; the RGB camera does not
        self.vopt.geomgroup[3] = 1            # thermal/depth: cheap box colliders instead of detailed meshes
        self.vopt.geomgroup[MESH_GROUP] = 0
        self.vopt_rgb.geomgroup[MESH_GROUP] = 1
        self.ray_groups = np.array([1, 0, 0, 1, 0, 0], np.uint8)  # walls/people/debris + box colliders
        fy = (H / 2) / math.tan(math.radians(model.cam_fovy[self.cam]) / 2)
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        self._ray_cam = np.stack([(u - W / 2) / fy, -(v - H / 2) / fy, -np.ones_like(u)], -1)  # per-pixel, z=-1

    # -- ground-truth helpers (never exported to the brain) --
    def ignite(self, i: int, temp_c: float = 450.0) -> None:
        self.temp_by_thing[i] = temp_c

    def _thing_xy(self, i):
        p = self.d.xpos[self.thing_body[i]]
        return float(p[0]), float(p[1])

    def _fires(self):
        return [(self._thing_xy(i), T) for i, T in self.temp_by_thing.items() if T >= FIRE_MIN_C]

    def _robot_xy(self):
        p = self.d.xpos[self.scout_body]
        return float(p[0]), float(p[1])

    def _yaw(self):
        q = self.d.xquat[self.scout_body]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    # -- sensor channels --
    def _geom_temps(self):
        lut = np.full(self.m.ngeom + 1, WALL_C, dtype=np.float32)  # last slot = background
        for g, ti in enumerate(self.geom_thing):
            lut[g] = self.temp_by_thing[ti] if ti >= 0 else (SCOUT_C if ti == -2 else WALL_C)
        lut[self.floor_geom] = AMBIENT_C
        lut[-1] = AMBIENT_C
        return lut

    def _thermal_and_depth(self):
        r = self.r
        r.update_scene(self.d, self.cam, scene_option=self.vopt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0  # no mirror images of fires on the floor
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        r.enable_segmentation_rendering()
        seg = r.render()
        r.disable_segmentation_rendering()
        ids = seg[:, :, 0].astype(np.int64)
        is_geom = seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
        ids = np.where(is_geom & (ids >= 0), ids, -1)
        thermal = self._geom_temps()[ids]
        r.enable_depth_rendering()
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        # surfaces that are not people/objects (walls, floor, furniture) take the local heat field
        surface = np.where(ids >= 0, self.geom_thing[np.clip(ids, 0, None)] < 0, True)
        pts = self.d.cam_xpos[self.cam] + (self._ray_cam * depth[..., None]) @ self.d.cam_xmat[self.cam].reshape(3, 3).T
        thermal = np.where(surface, np.maximum(thermal, self._heat_field(pts)), thermal)
        thermal = cv2.GaussianBlur(thermal.astype(np.float32), (5, 5), 0)
        thermal = (thermal + self.rng.normal(0, 0.5, thermal.shape)).astype(np.float32)
        return thermal, depth

    def _heat_field(self, pts: np.ndarray) -> np.ndarray:
        """Surface temperature near fires: radiant/convective falloff + hot gas layer under the ceiling."""
        field = np.full(pts.shape[:-1], AMBIENT_C, np.float32)
        for (fx, fy), T in self._fires():
            dT = T - AMBIENT_C
            d = np.sqrt((pts[..., 0] - fx) ** 2 + (pts[..., 1] - fy) ** 2 + (pts[..., 2] - 0.5) ** 2)
            dxy = np.hypot(pts[..., 0] - fx, pts[..., 1] - fy)
            layer = np.clip((pts[..., 2] - 1.2) / 1.2, 0, 1)
            field += 0.25 * dT * np.exp(-d / 0.5) + 0.08 * dT * np.exp(-dxy / 2.5) * layer
        return field

    def _rgb(self):
        self.r.update_scene(self.d, self.cam, scene_option=self.vopt_rgb)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        return self.r.render().copy()

    def _ranges(self):
        """60 GHz radar ring: 36 rays at 10 deg, 0.14 m high, walls/furniture colliders/people (not meshes/smoke)."""
        x, y = self._robot_xy()
        a = self._yaw() + np.radians(np.arange(36) * 10.0)
        vec = np.stack([np.cos(a), np.sin(a), np.zeros(36)], 1).ravel()
        gid = np.zeros(36, np.int32)
        dist = np.zeros(36, np.float64)
        mujoco.mj_multiRay(self.m, self.d, np.array([x, y, 0.14]), vec, self.ray_groups, 1, self.scout_body,
                           gid, dist, None, 36, MAX_RANGE)
        raw = dist.astype(np.float32)
        raw = np.where((raw < 0) | (raw > MAX_RANGE), MAX_RANGE, raw)
        noisy = raw + self.rng.normal(0, 0.05, 36)
        return np.clip(noisy, 0.05, MAX_RANGE).astype(np.float32)

    def _gas(self, xy):
        co = sum(2000.0 * math.exp(-math.dist(xy, f) / 2.0) for f, _ in self._fires())
        return max(0.0, co + float(self.rng.normal(0, 20)))

    def _o2(self, xy):
        """Electrochemical O2 cell: combustion depletes oxygen near the fire seat (clean air 20.9%)."""
        dep = sum(min(1.0, (T - AMBIENT_C) / 600) * 4.0 * math.exp(-math.dist(xy, f) / 1.5) for f, T in self._fires())
        return float(np.clip(20.9 - dep + self.rng.normal(0, 0.05), 10.0, 20.95))

    def _voices(self, xy):
        """Responsive casualties shout ~2 s in every 7 s. 70 dB at 1 m, spherical spreading, -12 dB through walls;
        the mic array reports bearing (+/-0.15 rad noise) relative to the robot heading when level > 30 dB."""
        out = []
        t = float(self.d.time)
        for i, th in enumerate(self.layout.things):
            if th.kind != 'casualty' or not th.responsive or (t + 2.3 * i) % 7.0 >= 2.0:
                continue
            p = self._thing_xy(i)
            dist = max(0.3, math.dist(xy, p))
            db = 70.0 - 20 * math.log10(dist) - (0.0 if self._los(xy, p) else 12.0) + float(self.rng.normal(0, 1))
            if db > 30.0:
                b = math.atan2(p[1] - xy[1], p[0] - xy[0]) - self._yaw() + float(self.rng.normal(0, 0.15))
                out.append((math.atan2(math.sin(b), math.cos(b)), db))
        return out

    def _los(self, a, b):
        """True if nothing but the target thing / free space lies between a and b."""
        pnt = np.array([a[0], a[1], 0.15])
        vec = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        dist = float(np.linalg.norm(vec))
        gid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(self.m, self.d, pnt, vec / dist, None, 1, self.scout_body, gid)
        return hit < 0 or hit >= dist - 0.6 or self.geom_thing[gid[0]] >= 0

    def _breath(self, target):
        rxy = self._robot_xy()
        ok = False
        for i, t in enumerate(self.layout.things):
            if t.breathing and math.dist(self._thing_xy(i), target) <= 1.0:  # radar beam footprint on a 1.7 m body
                ok = math.dist(rxy, target) <= BREATH_RANGE and self._los(rxy, target)
        base = 0.85 if ok else 0.10
        return float(np.clip(base + self.rng.normal(0, 0.05), 0, 1))

    def observe(self, want_breath_xy: tuple[float, float] | None = None) -> Observation:
        dt = max(0.0, self.d.time - self._last_t)
        self._last_t = self.d.time
        xy = self._robot_xy()
        t_local = AMBIENT_C + sum((T - AMBIENT_C) * math.exp(-math.dist(xy, f) / 0.8) for f, T in self._fires())
        self.self_temp += dt * 0.05 * (t_local - self.self_temp)
        self.battery = max(0.0, self.battery - dt / 600.0)
        thermal, depth = self._thermal_and_depth()
        pose = (xy[0] + float(self.rng.normal(0, 0.02)), xy[1] + float(self.rng.normal(0, 0.02)),
                self._yaw() + float(self.rng.normal(0, 0.01)))
        comms = not self.force_comms_drop and math.dist(xy, self.layout.start) <= COMMS_RANGE
        return Observation(
            t=float(self.d.time), pose=pose, thermal=thermal, depth=depth, ranges=self._ranges(),
            gas_ppm=self._gas(xy), comms=comms, self_temp_c=self.self_temp, battery=self.battery,
            rgb=self._rgb() if self.render_rgb else None,
            breath=self._breath(want_breath_xy) if want_breath_xy is not None else None,
            o2_pct=self._o2(xy), voices=self._voices(xy))
