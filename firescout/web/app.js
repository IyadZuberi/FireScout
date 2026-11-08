// FireScout IC Console - draws the live MuJoCo simulation streamed by web.py. No decisions are made here.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const $ = (s) => document.querySelector(s);
const post = (body) => fetch('/api/cmd', { method: 'POST', body: JSON.stringify(body) });
const PRIO = { P1: 0xffc93c, P2: 0xff9f2e, 'P?': 0xffc93c, EVAC: 0x43d675 };
const prioClass = (p) => (p === 'P?' ? 'Pq' : p);
const AWAKE_POSES = ['standing', 'kneel', 'walk0', 'walk1', 'walk2', 'walk3'];

// ------------------------------------------------------------------ renderer
const canvas = $('#view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080b);
scene.fog = new THREE.FogExp2(0x15161a, 0.018);
const camera = new THREE.PerspectiveCamera(42, 1, 0.03, 200);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
const labels = new CSS2DRenderer({ element: $('#labels') });
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.55, 0.45, 0.9);
composer.addPass(bloom);
composer.addPass(new OutputPass());

scene.add(new THREE.HemisphereLight(0xb4bdd0, 0x2a2420, 1.1));
const ceiling = new THREE.DirectionalLight(0xdfe6ff, 1.1);
ceiling.position.set(4, 2, 12);
ceiling.castShadow = true;
ceiling.shadow.mapSize.set(2048, 2048);
Object.assign(ceiling.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12, near: 1, far: 30 });
scene.add(ceiling, ceiling.target);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  composer.setSize(r.width, r.height);
  labels.setSize(r.width, r.height);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// ------------------------------------------------------------------ helpers
const objCache = {};
const texCache = {};
const loadOBJ = (url) => (objCache[url] ??= new Promise((ok) => new OBJLoader().load(url, ok)));
const tex = (url, srgb = true) => texCache[url] ??= (() => {
  const t = new THREE.TextureLoader().load(url);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
})();
function label(text, cls) {
  const d = document.createElement('div');
  d.className = 'lbl ' + (cls || '');
  d.textContent = text;
  return new CSS2DObject(d);
}
function softSprite(inner, outer) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, inner); gr.addColorStop(1, outer);
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const SMOKE_TEX = softSprite('rgba(255,255,255,0.9)', 'rgba(255,255,255,0)');
function vinylTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    const v = 112 + ((i + j) % 2) * 10 + Math.random() * 6;
    g.fillStyle = `rgb(${v},${v + 2},${v - 2})`;
    g.fillRect(i * 64, j * 64, 64, 64);
  }
  g.strokeStyle = 'rgba(0,0,0,0.25)';
  for (let k = 0; k <= 4; k++) { g.beginPath(); g.moveTo(k * 64, 0); g.lineTo(k * 64, 256); g.moveTo(0, k * 64); g.lineTo(256, k * 64); g.stroke(); }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ------------------------------------------------------------------ fire + smoke particles
const FIRE_VS = `attribute float aSize; attribute float aLife; varying float vLife;
void main(){ vLife=aLife; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_PointSize=aSize*(420.0/-mv.z); gl_Position=projectionMatrix*mv; }`;
const FIRE_FS = `varying float vLife;
void main(){ vec2 p=gl_PointCoord-0.5; p.y+=0.12;                     // tall teardrop flame, wide at the base
 float w=0.5-0.32*clamp(-p.y+0.5,0.0,1.0)*0.9; float d=length(vec2(p.x/max(w,0.08)*0.5,p.y*0.95));
 float a=smoothstep(0.5,0.12,d);
 vec3 c=mix(vec3(1.0,0.86,0.45),vec3(1.0,0.46,0.07),smoothstep(0.0,0.28,vLife));
 c=mix(c,vec3(0.62,0.1,0.02),smoothstep(0.28,0.8,vLife));
 float al=a*(1.0-vLife)*0.75; gl_FragColor=vec4(c*al*0.95,al); }`;
const SMOKE_VS = FIRE_VS;
const SMOKE_FS = `varying float vLife; uniform sampler2D map;
void main(){ float d=length(gl_PointCoord-0.5); float a=smoothstep(0.5,0.0,d);
 float al=a*0.26*smoothstep(0.0,0.12,vLife)*(1.0-vLife); gl_FragColor=vec4(vec3(0.09,0.09,0.1),al); }`;

class Fire {
  constructor(x, y, z0) {
    this.x = x; this.y = y; this.z0 = z0; this.growth = 0; this.on = true;
    this.group = new THREE.Group();
    this.flame = this._points(700, FIRE_VS, FIRE_FS, THREE.AdditiveBlending);
    this.smoke = this._points(260, SMOKE_VS, SMOKE_FS, THREE.NormalBlending);
    this.light = new THREE.PointLight(0xff7a2a, 3, 6, 2);
    this.light.position.set(x, y, z0 + 0.8);
    this.light.castShadow = true;
    this.light.shadow.mapSize.set(512, 512);
    this.group.add(this.flame.pts, this.smoke.pts, this.light);
    for (const s of [this.flame, this.smoke]) for (let k = 0; k < s.n; k++) this._spawn(s, k, Math.random());
  }
  _points(n, vs, fs, blending) {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3), size = new Float32Array(n), life = new Float32Array(n);
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    geo.setAttribute('aLife', new THREE.BufferAttribute(life, 1));
    const mat = new THREE.ShaderMaterial({ vertexShader: vs, fragmentShader: fs, transparent: true, depthWrite: false,
      blending, uniforms: { map: { value: SMOKE_TEX } } });
    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    return { n, pts, pos, size, life, vel: new Float32Array(n * 3), age: new Float32Array(n), max: new Float32Array(n),
      base: new Float32Array(n), smoke: blending === THREE.NormalBlending };
  }
  _spawn(s, k, pre = 0) {
    const r = (0.22 + 0.9 * this.growth) * Math.sqrt(Math.random()), a = Math.random() * 6.283;
    s.pos[k * 3] = this.x + r * Math.cos(a); s.pos[k * 3 + 1] = this.y + r * Math.sin(a);
    s.pos[k * 3 + 2] = this.z0 + (s.smoke ? 0.9 + 0.6 * this.growth : 0.04);
    const up = s.smoke ? 0.35 + Math.random() * 0.3 : (0.7 + Math.random() * 0.9) * (1 + 0.7 * this.growth) * (1 - r);
    s.vel[k * 3] = (Math.random() - 0.5) * 0.15; s.vel[k * 3 + 1] = (Math.random() - 0.5) * 0.15; s.vel[k * 3 + 2] = up;
    s.max[k] = s.smoke ? 5 + Math.random() * 5 : 0.45 + Math.random() * 0.6;
    s.age[k] = pre * s.max[k];
    s.base[k] = s.smoke ? 1.2 + Math.random() : (0.14 + Math.random() * 0.18) * (1 + 0.8 * this.growth);
  }
  update(dt, t, showSmoke) {
    this.group.visible = this.on;
    this.smoke.pts.visible = showSmoke;
    if (!this.on) return;
    for (const s of [this.flame, this.smoke]) {
      for (let k = 0; k < s.n; k++) {
        s.age[k] += dt;
        if (s.age[k] >= s.max[k]) this._spawn(s, k);
        const i = k * 3, l = s.age[k] / s.max[k];
        if (s.smoke) {
          if (s.pos[i + 2] > 2.25) { s.vel[i + 2] = 0; s.vel[i] += (s.pos[i] - this.x) * 0.04 * dt; s.vel[i + 1] += (s.pos[i + 1] - this.y) * 0.04 * dt; }
          s.size[k] = s.base[k] * (1 + 2.5 * l);
        } else {
          s.vel[i] += (Math.sin(t * 7 + k) * 0.6 + (this.x - s.pos[i]) * 1.5) * dt;
          s.vel[i + 1] += (Math.cos(t * 6 + k * 1.3) * 0.6 + (this.y - s.pos[i + 1]) * 1.5) * dt;
          s.size[k] = s.base[k] * (1 - 0.6 * l);
        }
        s.pos[i] += s.vel[i] * dt; s.pos[i + 1] += s.vel[i + 1] * dt; s.pos[i + 2] += s.vel[i + 2] * dt;
        s.life[k] = l;
      }
      const g = s.pts.geometry;
      g.attributes.position.needsUpdate = g.attributes.aSize.needsUpdate = g.attributes.aLife.needsUpdate = true;
    }
    this.light.intensity = (2.5 + 5 * this.growth) * (0.8 + 0.2 * Math.sin(t * 17) * Math.random() + 0.1 * Math.sin(t * 5.3));
  }
}

// ------------------------------------------------------------------ robot model (shared with the showcase)
export function buildRobot(withLabels = false) {
  // Portable tracked scout in the style of fielded fire / EOD crawlers: low armoured hull on rubber tracks with
  // front flippers, aluminium heat shield, carry handle, pan-tilt sensor head on a short mast. Dimensions in metres
  // (hull 0.46 x 0.26 x 0.12, ~9 kg class).
  const g = new THREE.Group();
  const M = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.25, ...o });
  const add = (geo, mat, x, y, z, rx = 0, ry = 0, rz = 0) => {
    const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.rotation.set(rx, ry, rz);
    m.castShadow = m.receiveShadow = true; g.add(m); return m;
  };
  const box = (sx, sy, sz, mat, x, y, z, rx, ry, rz) => add(new THREE.BoxGeometry(sx, sy, sz), mat, x, y, z, rx, ry, rz);
  const cylY = (r, h, mat, x, y, z, seg = 24) => add(new THREE.CylinderGeometry(r, r, h, seg), mat, x, y, z);          // axis = y
  const cylX = (r, h, mat, x, y, z, seg = 24) => add(new THREE.CylinderGeometry(r, r, h, seg), mat, x, y, z, 0, 0, Math.PI / 2);
  const cylZ = (r, h, mat, x, y, z, seg = 24) => add(new THREE.CylinderGeometry(r, r, h, seg), mat, x, y, z, Math.PI / 2, 0, 0);

  const hull = M(0x464a50, { roughness: 0.55, metalness: 0.35 });            // powder-coated graphite
  const alu = M(0xb9bcc0, { roughness: 0.32, metalness: 0.9 });             // heat-shield plate
  const rubber = M(0x111112, { roughness: 0.95, metalness: 0 });
  const steel = M(0x5b5f64, { roughness: 0.4, metalness: 0.8 });
  const hiVis = M(0xff7a1a, { roughness: 0.5, metalness: 0.1 });            // fire-service orange
  const reflect = M(0xd9ff3a, { roughness: 0.3, emissive: 0x1a2200 });      // retro-reflective band
  const glass = M(0x0a0a0c, { roughness: 0.05, metalness: 0.9 });
  const gold = M(0xc8a24a, { roughness: 0.3, metalness: 1 });

  // tracks: rubber belt (stadium shape) with cleats, drive sprocket, idler and road wheels
  for (const s of [-1, 1]) {
    const y = 0.155 * s;
    box(0.4, 0.07, 0.1, rubber, 0, y, 0.06);
    cylY(0.05, 0.07, rubber, 0.2, y, 0.06); cylY(0.05, 0.07, rubber, -0.2, y, 0.06);
    for (let k = 0; k < 16; k++) box(0.012, 0.072, 0.008, M(0x1d1d1f), -0.19 + k * 0.025, y, 0.112);
    for (const x of [-0.2, 0.2]) cylY(0.032, 0.074, steel, x, y, 0.06, 12);
    for (const x of [-0.1, 0, 0.1]) cylY(0.026, 0.074, steel, x, y, 0.045, 12);
    // front flippers (tilted up) - climb door sills and debris
    const fl = new THREE.Group(); fl.position.set(0.2, y + 0.06 * s, 0.06); fl.rotation.y = -0.45;
    const b1 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.035, 0.06), rubber); b1.position.x = 0.1; b1.castShadow = true;
    const w1 = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.04, 12), hiVis); w1.position.x = 0.2;
    fl.add(b1, w1); g.add(fl);
  }
  // hull: lower body + bevelled upper deck
  box(0.42, 0.24, 0.08, hull, 0, 0, 0.1);
  box(0.36, 0.22, 0.035, hull, -0.01, 0, 0.155, 0, 0, 0);
  box(0.06, 0.22, 0.05, hull, 0.205, 0, 0.13, 0, 0.6, 0);                  // sloped glacis
  box(0.3, 0.2, 0.006, alu, -0.03, 0, 0.176);                                // heat-shield top plate
  for (const s of [-1, 1]) box(0.34, 0.004, 0.018, reflect, 0, 0.121 * s, 0.11);  // reflective side bands
  box(0.004, 0.16, 0.03, hiVis, -0.211, 0, 0.115);                           // rear orange panel
  // carry handle (crew places it at the doorway)
  for (const x of [-0.12, 0.04]) cylZ(0.008, 0.06, steel, x, 0, 0.21, 10);
  cylX(0.009, 0.17, steel, -0.04, 0, 0.24, 10);
  // mast + pan-tilt sensor head
  cylZ(0.022, 0.16, steel, 0.11, 0, 0.26, 16);
  const head = new THREE.Group(); head.position.set(0.12, 0, 0.36); g.add(head);
  const hb = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.17, 0.075), M(0x26282b, { roughness: 0.45, metalness: 0.4 })); hb.castShadow = true; head.add(hb);
  const visor = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.15, 0.055), glass); visor.position.x = 0.047; head.add(visor);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.012, 24), M(0x1a1410, { roughness: 0.1, metalness: 0.95 }));
  lens.rotation.z = Math.PI / 2; lens.position.set(0.052, 0.04, 0.005); head.add(lens);                       // germanium LWIR window
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.024, 0.003, 8, 24), gold); rim.rotation.y = Math.PI / 2; rim.position.set(0.055, 0.04, 0.005); head.add(rim);
  const radar = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.05, 0.045), M(0xeeeeea, { roughness: 0.7 })); radar.position.set(0.05, -0.045, 0); head.add(radar);
  const depth = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.07, 0.014), glass); depth.position.set(0.05, 0, 0.03); head.add(depth);
  for (let k = 0; k < 6; k++) {                                                                            // 850 nm IR ring
    const a = k / 6 * 6.283;
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.004), M(0x330000, { emissive: 0xff2010, emissiveIntensity: 1.4 }));
    led.position.set(0.054, 0.04 + 0.032 * Math.cos(a), 0.005 + 0.032 * Math.sin(a)); head.add(led);
  }
  for (const y of [-0.085, 0.085]) {                                                                       // white work lights
    const l = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.025, 0.02), M(0xffffff, { emissive: 0xfff4dc, emissiveIntensity: 2 }));
    l.position.set(0.212, y, 0.12); g.add(l);
  }
  // 4 microphones (hull corners), speaker grille, gas cell inlet, mesh-radio antenna, amber beacon
  for (const [x, y] of [[0.17, 0.1], [0.17, -0.1], [-0.17, 0.1], [-0.17, -0.1]]) cylZ(0.009, 0.012, rubber, x, y, 0.18, 10);
  const spk = cylZ(0.032, 0.006, M(0x1a1a1c, { roughness: 0.9 }), -0.12, 0.06, 0.181);
  cylX(0.012, 0.02, M(0x6c5ce7, { roughness: 0.6 }), 0.0, -0.128, 0.1, 12);
  cylZ(0.004, 0.2, rubber, -0.18, -0.09, 0.28, 8);
  const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.02, 0.03, 16), M(0xffa020, { emissive: 0xff8a00, emissiveIntensity: 2, transparent: true, opacity: 0.9 }));
  beacon.rotation.x = Math.PI / 2; beacon.position.set(-0.17, 0.08, 0.195); g.add(beacon);
  g.userData = { beacon, lens, head, speaker: spk };
  if (withLabels) {
    const L = [['Thermal camera (LWIR)', 's1', 0.2, 0.06, 0.42], ['60 GHz radar', 's2', 0.2, -0.08, 0.34],
      ['Depth + IR night camera', 's3', 0.18, 0, 0.46], ['4 microphones', 's4', 0.2, 0.13, 0.2],
      ['Speaker', 's5', -0.13, 0.08, 0.22], ['CO + O2 gas cells', 's6', 0, -0.17, 0.1],
      ['Self-temperature + battery', 's7', -0.06, 0.16, 0.12], ['Mesh radio', 's8', -0.18, -0.09, 0.4],
      ['Flippers: climb sills', '', 0.33, 0.2, 0.1], ['Carry handle', '', -0.04, 0, 0.27]];
    for (const [t, c, x, y, z] of L) { const l = label(t, 'spec ' + c); l.position.set(x, y, z); g.add(l); }
  }
  return g;
}

// ------------------------------------------------------------------ world (rebuilt per scenario)
let world = null, SC = null, robot = null, fires = {}, thingMeshes = {}, smokeLayer = null, mapTexPlane = null;
const dyn = new THREE.Group(); scene.add(dyn);   // routes, markers: rebuilt from state
let showSmoke = true, view = 'over';
const CUT_H = 1.1;  // walls drawn cut away at 1.1 m so people and the robot stay visible (sim walls are 2.4 m)

async function buildWorld(sc) {
  SC = sc;
  if (world) scene.remove(world);
  world = new THREE.Group(); scene.add(world);
  fires = {}; thingMeshes = {};
  const W = sc.width, H = sc.height;
  const floorTex = vinylTexture(); floorTex.repeat.set(W / 1.2, H / 1.2);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(W + 0.4, H + 0.4),
    new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.75, metalness: 0.05 }));
  floor.position.set(W / 2, H / 2, 0); floor.receiveShadow = true; world.add(floor);
  const plaster = new THREE.MeshStandardMaterial({ color: 0xd8d2c6, roughness: 0.92 });
  const skirt = new THREE.MeshStandardMaterial({ color: 0x5a554e, roughness: 0.7 });
  const capMat = new THREE.MeshBasicMaterial({ color: 0x1b1b1e });  // cut face of the wall
  for (const [x0, y0, x1, y1] of sc.walls) {
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len < 0.01) continue;
    const w = new THREE.Mesh(new THREE.BoxGeometry(len + sc.wall_t, sc.wall_t, CUT_H), plaster);  // cutaway
    w.position.set((x0 + x1) / 2, (y0 + y1) / 2, CUT_H / 2);
    const cap = new THREE.Mesh(new THREE.BoxGeometry(len + sc.wall_t, sc.wall_t + 0.004, 0.01), capMat);
    cap.position.set(w.position.x, w.position.y, CUT_H + 0.005); cap.rotation.z = Math.atan2(y1 - y0, x1 - x0); world.add(cap);
    w.rotation.z = Math.atan2(y1 - y0, x1 - x0);
    w.castShadow = w.receiveShadow = true; world.add(w);
    const s = new THREE.Mesh(new THREE.BoxGeometry(len + sc.wall_t + 0.01, sc.wall_t + 0.012, 0.1), skirt);
    s.position.set(w.position.x, w.position.y, 0.05); s.rotation.z = w.rotation.z; world.add(s);
  }
  // robot-built map painted on the floor (toggle "Robot map")
  mapTexPlane = new THREE.Mesh(new THREE.PlaneGeometry(W, H),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55, depthWrite: false }));
  mapTexPlane.position.set(W / 2, H / 2, 0.015); mapTexPlane.visible = showFloorMap; world.add(mapTexPlane);
  for (const p of sc.props) {
    loadOBJ(`/assets/${p.key}.obj`).then((o) => {
      const m = o.clone();
      const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...p.rgb), roughness: 0.6, metalness: 0.1 });
      m.traverse((c) => { if (c.isMesh) { c.material = mat; c.castShadow = c.receiveShadow = true; } });
      m.scale.set(...p.scale); m.position.set(p.xy[0], p.xy[1], p.z); m.rotation.z = p.yaw;
      world.add(m);
    });
  }
  for (const t of sc.things) await addThing(t);
  // smoke layer under the ceiling (thickens and banks down as the fire grows)
  smokeLayer = new THREE.Group();
  const smat = new THREE.SpriteMaterial({ map: SMOKE_TEX, color: 0x2a2a2d, transparent: true, opacity: 0.1, depthWrite: false });
  for (let k = 0; k < 70; k++) {
    const s = new THREE.Sprite(smat);
    s.position.set(Math.random() * W, Math.random() * H, 2.2 + Math.random() * 0.2);
    const sc2 = 2.5 + Math.random() * 2.5; s.scale.set(sc2, sc2, 1);
    s.userData = { x: s.position.x, y: s.position.y, z: s.position.z, p: Math.random() * 6 };
    smokeLayer.add(s);
  }
  world.add(smokeLayer);
  robot = buildRobot();
  robot.scale.setScalar(1.6);  // drawn 1.6x so it reads at building scale
  const spot = new THREE.SpotLight(0xfff4dc, 6, 9, 0.5, 0.6, 1.4);
  spot.position.set(0.14, 0, 0.3); spot.target.position.set(2, 0, 0); spot.castShadow = true;
  robot.add(spot, spot.target);
  const fov = new THREE.Mesh(new THREE.CircleGeometry(4 / 1.6, 32, -0.497, 0.995),   // 4 m thermal range (robot drawn 1.6x)
    new THREE.MeshBasicMaterial({ color: 0x5fd3ff, transparent: true, opacity: 0.045, depthWrite: false }));
  fov.position.z = 0.02; robot.add(fov);
  const halo = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.36, 48), new THREE.MeshBasicMaterial({ color: 0x2ee6ff, transparent: true, opacity: 0.8 }));
  halo.position.z = 0.012; robot.add(halo);
  robot.userData.halo = halo;
  const rl = label('FireScout', 'robot'); rl.position.set(0, 0, 0.5); robot.add(rl);
  world.add(robot);
  // room numbers
  sc.rooms.forEach((r, k) => {
    const l = label(`R${k + 1} ${r.kind}`, 'room');
    l.position.set(r.box[0] + 0.7, r.box[3] - 0.35, 2.45); world.add(l);
  });
  const ent = label('Entry · BA control', 'entry'); ent.position.set(sc.start[0], sc.start[1], 0.3); world.add(ent);
  setView(view);
  buildRoomPicker();
  $('#h-scenario').textContent = `${sc.name} · ${sc.width} x ${sc.height} m · layout ${sc.seed}`;
}

async function addThing(t) {
  const g = new THREE.Group();
  g.position.set(t.xy[0], t.xy[1], 0); g.rotation.z = t.yaw;
  if (t.kind === 'casualty' && t.style && t.awake) {   // conscious occupant: one mesh per pose, swapped live
    const [v, outfit, hair] = t.style.split('|');
    const sex = v[0];
    const bm = new THREE.MeshStandardMaterial({ map: tex(`/assets/tex_human_${outfit}.png`), roughness: 0.85 });
    const hmat = new THREE.MeshStandardMaterial({ map: tex('/assets/tex_hair.png'), color: new THREE.Color(...(SC.hair[hair] || [0.3, 0.25, 0.2])), roughness: 0.7 });
    g.userData.poses = {};
    for (const pose of AWAKE_POSES) {
      const [body, hm] = await Promise.all([loadOBJ(`/assets/human_${sex}_${pose}.obj`), loadOBJ(`/assets/human_${sex}_${pose}_hair.obj`)]);
      const pg = new THREE.Group();
      const b = body.clone(), h = hm.clone();
      b.traverse((c) => { if (c.isMesh) { c.material = bm; c.castShadow = c.receiveShadow = true; } });
      h.traverse((c) => { if (c.isMesh) { c.material = hmat; c.castShadow = true; } });
      pg.add(b, h); pg.visible = pose === 'standing';
      g.add(pg); g.userData.poses[pose] = pg;
    }
    g.userData.awake = true;
  } else if (t.kind === 'casualty' && t.style) {
    const [v, outfit, hair] = t.style.split('|');
    const [body, hm] = await Promise.all([loadOBJ(`/assets/human_${v}.obj`), loadOBJ(`/assets/human_${v}_hair.obj`)]);
    const b = body.clone(), h = hm.clone();
    const bm = new THREE.MeshStandardMaterial({ map: tex(`/assets/tex_human_${outfit}.png`), roughness: 0.85 });
    const hc = SC.hair[hair] || [0.3, 0.25, 0.2];
    const hmat = new THREE.MeshStandardMaterial({ map: tex('/assets/tex_hair.png'), color: new THREE.Color(...hc), roughness: 0.7 });
    b.traverse((c) => { if (c.isMesh) { c.material = bm; c.castShadow = c.receiveShadow = true; } });
    h.traverse((c) => { if (c.isMesh) { c.material = hmat; c.castShadow = true; } });
    g.add(b, h);
  } else if (t.kind === 'fire' || t.kind === 'spare') {
    const char = new THREE.MeshStandardMaterial({ color: 0x15100c, roughness: 1 });
    for (let k = 0; k < 6; k++) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.5 * Math.random() + 0.2, 0.08, 0.06), char);
      m.position.set((Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5, 0.04 + k * 0.02);
      m.rotation.z = Math.random() * 3; m.rotation.y = (Math.random() - 0.5) * 0.4; m.castShadow = true; g.add(m);
    }
    const ember = new THREE.Mesh(new THREE.CircleGeometry(0.4, 24), new THREE.MeshBasicMaterial({ color: 0xff4a0a }));
    ember.position.z = 0.012; g.add(ember); g.userData.ember = ember;
    fires[t.i] = new Fire(t.xy[0], t.xy[1], 0); world.add(fires[t.i].group);
    if (t.kind === 'spare') { fires[t.i].on = false; g.visible = false; g.userData.spare = true; }
  } else if (t.kind === 'shelf') {
    const o = (await loadOBJ('/assets/cabinet.obj')).clone();
    o.traverse((c) => { if (c.isMesh) { c.material = new THREE.MeshStandardMaterial({ color: 0x8a7a66, roughness: 0.7 }); c.castShadow = true; } });
    o.scale.set(1, 1, 0.55); g.add(o);
    fires[t.i] = new Fire(t.xy[0], t.xy[1], 1.36); fires[t.i].on = false; world.add(fires[t.i].group);
  } else if (t.style === 'radiator') {
    const o = (await loadOBJ('/assets/radiator.obj')).clone();
    o.traverse((c) => { if (c.isMesh) c.material = new THREE.MeshStandardMaterial({ color: 0xeeeeea, roughness: 0.4, metalness: 0.4 }); });
    g.add(o);
  } else if (t.style === 'pipe') {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, t.size[0], 20), new THREE.MeshStandardMaterial({ color: 0xb87333, metalness: 0.8, roughness: 0.3 }));
    p.rotation.z = Math.PI / 2; p.position.z = 0.1; g.add(p);
  } else if (t.style === 'pet') {
    const fur = new THREE.MeshStandardMaterial({ color: 0x8a6438, roughness: 1 });
    const b = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 12), fur); b.scale.set(2, 1, 1); b.position.z = 0.11; g.add(b);
    const hd = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), fur); hd.position.set(0.22, 0, 0.16); g.add(hd);
  }
  thingMeshes[t.i] = g;
  world.add(g);
}

// ------------------------------------------------------------------ routes + markers from state
const COL = { robot: 0x5fd3ff, crew: 0x43d675, exit: 0xecebe8, fire: 0xff4d3d, voice: 0xc08cff, evac: 0x43d675 };
function tube(pts, color, r, z, opacity = 1) {
  if (!pts || pts.length < 2) return null;
  const v = pts.map((p) => new THREE.Vector3(p[0], p[1], z));
  const curve = new THREE.CatmullRomCurve3(v, false, 'centripetal');
  const geo = new THREE.TubeGeometry(curve, Math.max(8, v.length * 3), r, 8, false);
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, toneMapped: false, transparent: opacity < 1, opacity }));
  m.userData = { curve, grow: 0, total: geo.index.count };   // draws on when the plan changes
  geo.setDrawRange(0, 0);
  return m;
}
let lastRoutes = '', crewArrows = [], pings = [];
const seenTracks = new Set();
function ping(x, y, color) {   // expanding ring when something new is found
  const m = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.28, 48), new THREE.MeshBasicMaterial({ color, transparent: true, toneMapped: false }));
  m.position.set(x, y, 0.05); m.userData.age = 0; scene.add(m); pings.push(m);
}
function updateDynamic(st, t, dt) {
  const evac = (st.things || []).filter((x) => x.route && x.route.length > 1).map((x) => x.route);
  const key = JSON.stringify([st.path, st.crew, st.exit, st.fires, st.tracks, st.voice, evac.map((r) => r.length)]);
  if (key !== lastRoutes) {
    lastRoutes = key;
    for (const c of [...dyn.children]) { dyn.remove(c); c.geometry?.dispose?.(); }
    crewArrows = [];
    const p = tube(st.path, COL.robot, 0.022, 0.06); if (p) dyn.add(p);
    const e = tube(st.exit, COL.exit, 0.014, 0.035, 0.45); if (e) dyn.add(e);
    const c = tube(st.crew, COL.crew, 0.05, 0.09);
    if (c) {
      dyn.add(c);
      for (let k = 0; k < 10; k++) {
        const a = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.22, 12), new THREE.MeshBasicMaterial({ color: 0xc9ffd9, toneMapped: false }));
        a.userData = { curve: c.userData.curve, off: k / 10 }; dyn.add(a); crewArrows.push(a);
      }
    }
    const tagged = [];   // one tag per fire cluster (hottest seat), every seat still gets its no-go zone
    for (const f of (st.fires || []).slice().sort((p, q) => q[2] - p[2])) {
      f.tag = !tagged.some((q) => Math.hypot(q[0] - f[0], q[1] - f[1]) < 2.2); if (f.tag) tagged.push(f);
    }
    for (const f of st.fires || []) {
      const zone = new THREE.Mesh(new THREE.CircleGeometry(1.0, 40), new THREE.MeshBasicMaterial({ color: COL.fire, transparent: true, opacity: 0.16, depthWrite: false }));
      zone.position.set(f[0], f[1], 0.02); dyn.add(zone);
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1.05, 48), new THREE.MeshBasicMaterial({ color: COL.fire, toneMapped: false }));
      ring.position.set(f[0], f[1], 0.025); dyn.add(ring);
      if (f.tag) { const l = label(`FIRE ${f[2]} C · ${f[3]}`, 'pin fire'); l.position.set(f[0], f[1], 1.9); dyn.add(l); }
    }
    for (const tr of st.tracks || []) {
      const col = PRIO[tr.prio] ?? PRIO['P?'];
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1.2, 8), new THREE.MeshBasicMaterial({ color: col, toneMapped: false }));
      beam.rotation.x = Math.PI / 2; beam.position.set(tr.xy[0], tr.xy[1], 0.6); dyn.add(beam);
      const ball = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), new THREE.MeshBasicMaterial({ color: col, toneMapped: false }));
      ball.position.set(tr.xy[0], tr.xy[1], 1.25); ball.userData.pulse = true; dyn.add(ball);
      const txt = tr.prio === 'EVAC' ? 'Walking out' : tr.state === 'pinned'
        ? `${tr.rank ? tr.rank + '. ' : ''}${tr.prio} person · ${tr.room}` : `Possible person · ${tr.room}`;
      const l = label(txt, 'pin person ' + prioClass(tr.prio)); l.position.set(tr.xy[0], tr.xy[1], 1.5); dyn.add(l);
      const gr = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.72, 48), new THREE.MeshBasicMaterial({ color: col, toneMapped: false, transparent: true, opacity: 0.9 }));
      gr.position.set(tr.xy[0], tr.xy[1], 0.03); dyn.add(gr);
      if (!seenTracks.has(tr.id + tr.state)) { seenTracks.add(tr.id + tr.state); ping(tr.xy[0], tr.xy[1], col); }
    }
    for (const r of evac) { const e2 = tube(r, COL.evac, 0.03, 0.08); if (e2) dyn.add(e2); }
    if (st.voice) {
      const [x, y, b] = st.voice;
      const v = tube([[x, y], [x + 4 * Math.cos(b), y + 4 * Math.sin(b)]], COL.voice, 0.02, 0.5); if (v) dyn.add(v);
    }
    updateLegend(st, evac.length > 0);
  }
  for (const c of dyn.children) {
    const u = c.userData;
    if (u.total && u.grow < 1) { u.grow = Math.min(1, u.grow + dt / 0.6); c.geometry.setDrawRange(0, Math.floor(u.total * (1 - (1 - u.grow) ** 3) / 3) * 3); }
    if (u.pulse) c.scale.setScalar(1 + 0.35 * Math.sin(t * 6));
  }
  for (const a of crewArrows) {
    const uu = (a.userData.off + t * 0.08) % 1, c = a.userData.curve;
    a.position.copy(c.getPointAt(uu));
    a.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), c.getTangentAt(uu));
  }
  pings = pings.filter((p) => {
    p.userData.age += dt; const a = p.userData.age / 1.4;
    p.scale.setScalar(1 + 9 * a); p.material.opacity = Math.max(0, 1 - a);
    if (a >= 1) { scene.remove(p); p.geometry.dispose(); return false; }
    return true;
  });
}
function updateLegend(st, evac) {
  const items = [['robot', 'var(--robot)', "Robot's next move"]];
  if (st.crew) items.push(['crew', 'var(--crew)', 'Crew route, rescue order']);
  if (evac) items.push(['crew', 'var(--crew)', 'Person walking out']);
  if (st.exit?.length) items.push(['exit', 'var(--text)', 'Way back to entry']);
  if (st.fires?.length) items.push(['fire', 'var(--fire)', 'Fire no-go zone']);
  if (st.tracks?.length) items.push(['p', 'var(--person)', 'People']);
  if (st.voice) items.push(['v', 'var(--voice)', 'Voice heard']);
  $('#legend').innerHTML = items.map(([, c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join('');
}

// ------------------------------------------------------------------ camera views
function fitOverview() {
  if (!SC) return;
  const fov = THREE.MathUtils.degToRad(camera.fov), a = camera.aspect;
  const need = Math.max((SC.height * 1.15) / 2 / Math.tan(fov / 2), (SC.width * 1.18) / 2 / (Math.tan(fov / 2) * a));
  const dir = new THREE.Vector3(0, -0.45, 1).normalize();
  controls.target.set(SC.width / 2, SC.height / 2 + 0.3, 0);
  camera.position.copy(controls.target).addScaledVector(dir, need);
}
function setView(v) {
  view = v;
  document.querySelectorAll('.viewbar [data-v]').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
  controls.enabled = v === 'over';
  renderer.toneMappingExposure = v === 'eye' ? 0.8 : 1.05;
  if (robot) robot.visible = v !== 'eye';
  camera.fov = { over: 40, plan: 40, follow: 55, eye: 52 }[v]; camera.near = 0.05;
  camera.updateProjectionMatrix();
  controls.enabled = v === 'over' || v === 'plan';
  if (v === 'over') fitOverview();
  if (v === 'plan') {   // straight down, like a fire-service floor plan
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const need = Math.max((SC.height * 1.12) / 2 / Math.tan(fov / 2), (SC.width * 1.12) / 2 / (Math.tan(fov / 2) * camera.aspect));
    controls.target.set(SC.width / 2, SC.height / 2, 0);
    camera.position.set(SC.width / 2, SC.height / 2 - 0.001, need);
  }
}
document.querySelectorAll('.viewbar [data-v]').forEach((b) => b.onclick = () => setView(b.dataset.v));
addEventListener('resize', () => { if (view === 'over' || view === 'plan') setView(view); });

// ------------------------------------------------------------------ panels
const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const MODE_TXT = { EXPLORE: ['Searching', 'ok'], VERIFY: ['Checking a person', 'ok'], RETREAT: ['Retreating from heat', 'warn'],
  IC_STOP: ['Stopped by IC', 'warn'], RECALL: ['Returning to entry', 'warn'], HOLD: ['Holding', 'warn'],
  RETURN_COMMS: ['Returning to radio range', 'warn'] };
const KIND_TXT = { PIN: ['Confirmed a person', 'person'], VERIFY: ['Second look', 'person'], POSSIBLE: ['Possible person', 'person'],
  PRIORITY: ['Rescue priority', 'person'], REJECT: ['Not a person', ''], REPLAN: ['New route', 'fire'], RETREAT: ['Retreat', 'fire'],
  HEAR: ['Heard a voice', 'voice'], SAY: ['Spoke to a person', 'voice'], IC: ['IC order', 'ic'], COMMS: ['Radio', 'fire'],
  HOLD: ['Holding', ''], LEARN: ['Learned from IC', 'ic'] };
let lastCards = '', lastMsg = '', ackTimer = 0, lastPeople = '';

function updatePanels(st) {
  $('#h-clock').textContent = fmt(st.t);
  if (!st.obs) return;
  const cin = st.metrics.crew_in;
  $('#h-crew').textContent = cin > 0 ? `in ${cin}s` : 'committed';
  const [mt, mc] = MODE_TXT[st.mode] || [st.mode, 'ok'];
  const mode = $('#h-mode'); mode.textContent = mt + (st.ic ? ' · IC order' : ''); mode.className = 'pill ' + mc;
  const radio = $('#h-radio');
  radio.textContent = st.obs.comms ? 'Radio linked' : `Radio LOST · ${st.obs.queued} queued`;
  radio.className = 'pill ' + (st.obs.comms ? 'ok' : 'bad');
  $('#stage').classList.toggle('lost', !st.obs.comms);
  $('#lost').hidden = st.obs.comms;
  if (!st.obs.comms) $('#lost-t').textContent = `Robot keeps searching on its own. ${st.obs.queued} findings queued; screen shows its last upload.`;

  // people (left)
  const cas = (st.tracks || []).slice().sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99) || (a.prio === 'P?') - (b.prio === 'P?'));
  $('#p-count').textContent = cas.length;
  const peopleHtml = cas.length ? cas.map((c) => {
    const badge = c.prio === 'EVAC' ? 'Walking out' : c.state === 'pinned' ? c.prio : 'Possible';
    return `<li class="p ${prioClass(c.prio)}"><div class="t"><span class="badge">${badge}</span>
      <span class="where">${esc(c.room)}${c.rank ? ` · rescue ${c.rank}` : ''}</span></div>
      <div class="why">${esc(c.why)}${c.state !== 'pinned' && c.prio !== 'EVAC' ? ' · crew must check' : ''}</div>
      ${c.say ? `<div class="said">Robot told them: "${esc(c.say)}"</div>` : ''}</li>`;
  }).join('') : '<li class="empty">Searching. People appear here the moment the robot sees them.</li>';
  if (peopleHtml !== lastPeople) { lastPeople = peopleHtml; $('#people').innerHTML = peopleHtml; }  // re-render only on change

  // strip
  const top = cas.find((c) => c.rank === 1);
  $('#s-first').classList.toggle('idle', !top);
  $('#s-first-t').textContent = top ? `${top.prio} in ${top.room}: ${top.why}` :
    cas.length ? `${cas.length} possible ${cas.length > 1 ? 'people' : 'person'} - robot is checking for breathing` : 'Searching - no person found yet';
  const fires = (st.fires || []).slice().sort((a, b) => b[2] - a[2]);
  $('#s-fire').classList.toggle('idle', !fires.length);
  $('#s-fire-t').textContent = fires.length ? `Fire ${fires[0][2]} C in ${fires[0][3]}${fires.length > 1 ? ` (+${fires.length - 1} more)` : ''} - routes avoid it` : 'No fire seen yet';
  const say = (st.say || [])[0];
  $('#s-say').hidden = !say; if (say) $('#s-say-t').textContent = `Robot speaker: "${say}"`;

  // sense
  const o = st.obs;
  const air = $('#r-air'); air.textContent = `CO ${o.co} ppm · O2 ${o.o2}%`; air.className = o.co > 400 || o.o2 < 17 ? 'bad' : '';
  const body = $('#r-body'); body.textContent = `${o.self_temp} C · battery ${Math.round(o.battery * 100)}%`; body.className = o.self_temp > 50 ? 'bad' : '';
  const mic = $('#r-mic'); mic.textContent = st.voice ? 'Voice heard' : 'Listening'; mic.className = st.voice ? 'voice' : '';
  const mp = st.metrics.map;
  $('#r-map').textContent = mp.precision == null ? '-' : `${mp.walls_found}% of walls`;

  // decide
  const [now, why] = st.now || ['', ''];
  $('#d-now').textContent = now; $('#d-why').textContent = why ? `because ${why}` : '';
  const cards = st.cards || [];
  if (cards.join('|') !== lastCards) {
    const old = new Set(lastCards.split('|'));
    lastCards = cards.join('|');
    $('#d-list').innerHTML = cards.slice(-7).reverse().map((c) => {
      const m = c.match(/^\[(\d\d:\d\d)\] (\w+) trigger=(\S+) (.*?) \| (.*?) \| rule=(.*)$/);
      if (!m) return '';
      const [kt, kc] = KIND_TXT[m[2]] || [m[2], ''];
      return `<li class="${old.has(c) ? '' : 'new'}"><span class="t">${m[1]}</span><div><span class="k ${kc}">${kt}:</span>
        ${esc(m[4])} <span class="why">→ ${esc(m[5])}</span>
        <details><summary>sensor + rule</summary>sensor: ${esc(m[3])} · rule: ${esc(m[6])}</details></div></li>`;
    }).join('');
  }

  // act
  $('#a-drive').textContent = `${st.speed} m/s · ${st.speed_why.replace(/^\w+: /, '')}`;
  $('#a-say').textContent = say ? 'speaking now' : (cas.find((c) => c.say) ? 'last: ' + cas.find((c) => c.say).say : 'silent');
  $('#a-cov').textContent = `${st.metrics.coverage}% of floor`;
  $('#a-first').textContent = st.metrics.first == null ? 'not yet' : `T+${fmt(st.metrics.first)}, ${Math.round(150 - st.metrics.first)}s before crew`;

  // drawer evaluation + METHANE + IC acknowledgement
  const tr = st.metrics.truth;
  $('#eval').innerHTML = [['People in building', tr.n], ['Confirmed correctly', tr.ok], ['False confirmations', tr.fp],
    ['Walked out', st.evacuated || 0], ['Map walls found', mp.walls_found != null ? mp.walls_found + '%' : '-'],
    ['Map walls correct', mp.precision != null ? mp.precision + '%' : '-'],
    ['Brain loop p50 / p95', `${st.metrics.lat50} / ${st.metrics.lat95} ms`], ['Robot is in', st.robot_room || '-']]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join('');
  $('#methane').textContent = (st.methane || []).join('\n');
  if (st.msg && st.msg !== lastMsg) { lastMsg = st.msg; ack(st.msg); }
}
function ack(text) {
  const a = $('#ack'); a.textContent = text; clearTimeout(ackTimer); ackTimer = setTimeout(() => { a.textContent = ''; }, 6000);
}

// ------------------------------------------------------------------ sense: camera feeds + map
let feedName = 'thermal', feedT = 0;
const CAP = { thermal: 'Thermal camera: body heat and fire, through smoke', night: 'Night IR camera: 850 nm light, sees through thin smoke',
  colour: 'Colour camera: what a person would see (blinded by smoke)', map: 'Robot-built map (white walls, brown furniture) over the true plan (grey, hidden from the robot)' };
const feedImg = $('#feed'), mapCanvas = $('#mapc');
function selectFeed(name) {
  feedName = name;
  document.querySelectorAll('#feedtabs button').forEach((x) => x.classList.toggle('on', x.dataset.f === name));
  $('#feedcap').textContent = CAP[name];
  feedImg.hidden = name === 'map'; mapCanvas.hidden = name !== 'map';
  if (name !== 'map') feedImg.src = `/feed/${name}.jpg?${Date.now()}`;
}
document.querySelectorAll('#feedtabs button').forEach((b) => b.onclick = () => selectFeed(b.dataset.f));
feedImg.onload = () => { feedT = performance.now(); setTimeout(() => { if (feedName !== 'map') feedImg.src = `/feed/${feedName}.jpg?${Date.now()}`; }, 200); };
feedImg.onerror = () => setTimeout(() => { feedImg.src = `/feed/${feedName}.jpg?${Date.now()}`; }, 800);
feedImg.src = `/feed/thermal.jpg?${Date.now()}`;   // starts the refresh loop
setInterval(() => {
  const age = (performance.now() - feedT) / 1000, el = $('#feedage');
  el.textContent = feedName === 'map' ? 'live' : age < 1.5 ? 'live' : `${age.toFixed(0)}s old`;
  el.classList.toggle('stale', feedName !== 'map' && age >= 1.5);
}, 500);
const mapImg = new Image();
let mapTex = null;
mapImg.onload = mapImg.onerror = () => {
  if (mapImg.naturalWidth && SC) {
    const c = mapCanvas, g = c.getContext('2d');
    c.width = mapImg.naturalWidth; c.height = mapImg.naturalHeight;
    g.drawImage(mapImg, 0, 0);
    const sx = c.width / SC.width, sy = c.height / SC.height;
    g.strokeStyle = 'rgba(200,200,200,0.75)'; g.lineWidth = 1.5; g.setLineDash([5, 4]);
    g.beginPath();
    for (const [x0, y0, x1, y1] of SC.walls) { g.moveTo(x0 * sx, c.height - y0 * sy); g.lineTo(x1 * sx, c.height - y1 * sy); }
    g.stroke();
    if (mapTexPlane && showFloorMap) {
      mapTex?.dispose(); mapTex = new THREE.CanvasTexture(c); mapTex.colorSpace = THREE.SRGBColorSpace;
      mapTexPlane.material.map = mapTex; mapTexPlane.material.needsUpdate = true;
    }
  }
  setTimeout(() => { mapImg.src = `/feed/map.jpg?${Date.now()}`; }, 400);
};
mapImg.src = '/feed/map.jpg';

// ------------------------------------------------------------------ command + test controls
document.querySelectorAll('[data-a]').forEach((b) => b.onclick = () => post({ action: b.dataset.a }));
$('#room-pick').onchange = (e) => { if (e.target.value !== '') post({ action: 'room', room: +e.target.value }); };
function buildRoomPicker() {
  $('#room-pick').innerHTML = '<option value="">choose a room...</option>' +
    SC.rooms.map((r, k) => `<option value="${k}">R${k + 1} ${r.kind}</option>`).join('');
}
const drawer = $('#drawer');
$('#b-test').onclick = () => { drawer.hidden = !drawer.hidden; $('#b-test').classList.toggle('on', !drawer.hidden); };
$('#drawer-x').onclick = () => { drawer.hidden = true; $('#b-test').classList.remove('on'); };
let showFloorMap = false, armFire = false;
$('#t-smoke').onclick = (e) => { showSmoke = !showSmoke; e.currentTarget.classList.toggle('on', showSmoke); };
$('#t-floor').onclick = (e) => { showFloorMap = !showFloorMap; e.currentTarget.classList.toggle('on', showFloorMap); if (mapTexPlane) mapTexPlane.visible = showFloorMap; };
$('#t-fire-click').onclick = () => { armFire = true; $('#clickhint').hidden = false; setView('over'); };
const ray = new THREE.Raycaster(), floorPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
canvas.addEventListener('click', (e) => {
  if (!armFire) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = new THREE.Vector3();
  if (ray.ray.intersectPlane(floorPlane, hit)) { post({ action: 'fire_at', x: hit.x, y: hit.y }); ping(hit.x, hit.y, COL.fire); }
  armFire = false; $('#clickhint').hidden = true;
});
addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea') || document.querySelector('dialog[open]')) return;
  const k = e.key.toLowerCase();
  if (k === 'escape') { armFire = false; $('#clickhint').hidden = true; drawer.hidden = true; }
  else if (k === 's') post({ action: 'stop' });
  else if (k === 'r') post({ action: 'recall' });
  else if (k === 'v') setView({ over: 'plan', plan: 'follow', follow: 'eye', eye: 'over' }[view]);
  else if ('1234'.includes(k)) selectFeed(['thermal', 'night', 'colour', 'map'][+k - 1]);
});

// scenario dialog
$('#b-new').onclick = () => {
  $('#n-preset').innerHTML = Object.entries(SC.presets).map(([k, v]) => `<option value="${k}" ${k === SC.preset ? 'selected' : ''}>${v}</option>`).join('');
  $('#d-new').showModal();
};
$('#n-cancel').onclick = () => $('#d-new').close();
const startMission = (body) => { post({ action: 'new', ...body }); lastCards = ''; seenTracks.clear(); $('#d-new').close(); };
document.querySelectorAll('.scen button').forEach((b) => b.onclick = () => {
  const [preset, cas, fires, seed] = b.dataset.s.split(',');
  startMission({ preset, casualties: +cas, fires: +fires, seed: +seed });
});
$('#n-go').onclick = () => startMission({ preset: $('#n-preset').value, casualties: +$('#n-cas').value, fires: +$('#n-fires').value, seed: +$('#n-seed').value });
// METHANE
$('#b-methane').onclick = () => $('#d-methane').showModal();
$('#m-close').onclick = () => $('#d-methane').close();
$('#m-copy').onclick = () => navigator.clipboard?.writeText($('#methane').textContent);
// sound (browser needs a click before audio may play)
let sound = false;
$('#b-sound').onclick = (e) => {
  sound = !sound; const b = e.currentTarget;
  b.lastChild.textContent = sound ? 'Sound on' : 'Sound off'; b.classList.toggle('on', sound); b.setAttribute('aria-pressed', sound);
};
let audioBusy = 0;
function play(ev) {
  if (!sound || performance.now() < audioBusy) return;
  const lvl = ev.db >= 55 ? 'near' : ev.db >= 42 ? 'mid' : 'far';
  new Audio(`/assets/audio/${ev.sound}${ev.sound.startsWith('robot_') ? '' : '_' + lvl}.wav`).play().catch(() => {});
  audioBusy = performance.now() + (ev.sound.startsWith('robot_') ? 4500 : 2500);
}

// robot showcase
let showcase = null;
$('#b-robot').onclick = () => {
  $('#d-robot').showModal();
  if (showcase) return;
  const c = $('#robotc');
  const r = new THREE.WebGLRenderer({ canvas: c, antialias: true, alpha: true });
  const lab = new CSS2DRenderer();
  Object.assign(lab.domElement.style, { position: 'absolute', pointerEvents: 'none', top: '0', left: '0' });
  c.parentElement.append(lab.domElement);
  const s = new THREE.Scene();
  s.add(new THREE.HemisphereLight(0xdfe6ff, 0x2a2622, 1.6));
  const dl = new THREE.DirectionalLight(0xffffff, 3.2); dl.position.set(2, -2, 3); s.add(dl);
  const rim = new THREE.DirectionalLight(0xffb070, 2.0); rim.position.set(-2, 2, 1.5); s.add(rim);
  const floorDisc = new THREE.Mesh(new THREE.CircleGeometry(0.6, 48), new THREE.MeshStandardMaterial({ color: 0x2a2a2e, roughness: 0.9 }));
  s.add(floorDisc, buildRobot(true));
  r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.3;
  const cam = new THREE.PerspectiveCamera(32, 1, 0.01, 20); cam.up.set(0, 0, 1); cam.position.set(1.35, -1.25, 0.95);
  const oc = new OrbitControls(cam, c); oc.target.set(0, 0, 0.2); oc.autoRotate = true; oc.autoRotateSpeed = 1.2;
  showcase = () => {
    const w = c.clientWidth; r.setSize(w, w, false); lab.setSize(w, w); cam.aspect = 1; cam.updateProjectionMatrix();
    oc.update(); r.render(s, cam); lab.render(s, cam);
  };
};
$('#r-close').onclick = () => $('#d-robot').close();

// ------------------------------------------------------------------ live state
let ST = null, version = -1, robotPose = null, lastSeq = -1;
const es = new EventSource('/api/state');
es.onmessage = async (e) => {
  const st = JSON.parse(e.data);
  if (st.seq === lastSeq) return;  // stream repeats the latest state until a new one is ready
  lastSeq = st.seq;
  if (st.version !== version) {
    version = st.version;
    const sc = await (await fetch('/api/scene')).json();
    await buildWorld(sc);
  }
  ST = st;
  updatePanels(st);
  for (const ev of st.events || []) play(ev);
};

// ------------------------------------------------------------------ frame loop
const clock = new THREE.Clock();
let smooth = { x: 0, y: 0, yaw: 0 };
function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  if (ST && world && robot) {
    const r = ST.robot;
    const k = 1 - Math.exp(-dt * 12);
    let dy = r.yaw - smooth.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    smooth = { x: smooth.x + (r.x - smooth.x) * k, y: smooth.y + (r.y - smooth.y) * k, yaw: smooth.yaw + dy * k };
    if (!robotPose) { smooth = { ...r }; robotPose = true; }
    robot.position.set(smooth.x, smooth.y, 0); robot.rotation.z = smooth.yaw;
    robot.userData.beacon.material.emissiveIntensity = 1 + 2 * (Math.sin(t * 8) > 0);
    robot.userData.head.rotation.z = 0.35 * Math.sin(t * 0.9);  // pan-tilt head scanning
    robot.userData.halo.scale.setScalar(1 + 0.15 * Math.sin(t * 3));
    let maxG = 0;
    for (const th of ST.things) {
      const f = fires[th.i];
      const m = thingMeshes[th.i];
      if (f && m?.userData.spare && th.burning && (f.x !== th.xy[0] || f.y !== th.xy[1])) {  // unscripted fire moved in
        f.x = th.xy[0]; f.y = th.xy[1]; f.light.position.set(f.x, f.y, 0.8); m.position.set(f.x, f.y, 0); m.visible = true;
      }
      if (f) { f.on = th.burning; f.growth = th.growth; if (th.burning) maxG = Math.max(maxG, th.growth); }
      if (m && m.userData.ember) m.userData.ember.scale.setScalar(1 + 2.2 * th.growth);
      if (m && m.userData.awake && th.mode) {
        m.visible = th.mode !== 'out';
        m.position.x += (th.xy[0] - m.position.x) * 0.2; m.position.y += (th.xy[1] - m.position.y) * 0.2;
        m.rotation.z = th.yaw;
        const want = th.mode === 'walk' ? `walk${Math.floor(t * 4) % 4}` : th.mode === 'kneel' ? 'kneel' : 'standing';
        for (const [k, pg] of Object.entries(m.userData.poses)) pg.visible = k === want;
      }
    }
    for (const f of Object.values(fires)) f.update(dt, t, showSmoke);
    const fill = Math.min(1, ST.t / 240) * (Object.values(fires).some((f) => f.on) ? 1 : 0);
    if (smokeLayer) {
      smokeLayer.visible = showSmoke;
      for (const s of smokeLayer.children) {
        const u = s.userData;
        s.position.set(u.x + 0.3 * Math.sin(t * 0.1 + u.p), u.y + 0.25 * Math.cos(t * 0.08 + u.p), u.z - 1.0 * fill);
        s.material.opacity = 0.05 + 0.2 * fill;
      }
    }
    scene.fog.density = 0.012 + 0.03 * fill;
    updateDynamic(ST, t, dt);
    if (view === 'follow') {
      const d = new THREE.Vector3(Math.cos(smooth.yaw), Math.sin(smooth.yaw), 0);
      const want = new THREE.Vector3(smooth.x, smooth.y, 0).addScaledVector(d, -1.8).add(new THREE.Vector3(0, 0, 3.4));
      camera.position.lerp(want, 1 - Math.exp(-dt * 4));
      camera.lookAt(smooth.x + d.x * 0.9, smooth.y + d.y * 0.9, 0.1);
    } else if (view === 'eye') {
      const d = new THREE.Vector3(Math.cos(smooth.yaw), Math.sin(smooth.yaw), 0);
      camera.position.set(smooth.x + d.x * 0.25, smooth.y + d.y * 0.25, 0.4);   // thermal camera mount
      camera.lookAt(smooth.x + d.x * 3, smooth.y + d.y * 3, 0.3);
    } else controls.update();
  }
  composer.render();
  labels.render(scene, camera);
  if (showcase && $('#d-robot').open) showcase();
}
resize();
frame();
