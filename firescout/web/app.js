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
const PRIO = { P1: 0xff3b30, P2: 0xffa424, 'P?': 0xffe14a, EVAC: 0x38ff7a };
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
void main(){ float d=length(gl_PointCoord-0.5); float a=smoothstep(0.5,0.05,d);
 vec3 c=mix(vec3(1.0,0.93,0.65),vec3(1.0,0.5,0.1),smoothstep(0.0,0.3,vLife));
 c=mix(c,vec3(0.75,0.12,0.03),smoothstep(0.3,0.75,vLife));
 float al=a*(1.0-vLife)*0.7; gl_FragColor=vec4(c*al*1.15,al); }`;
const SMOKE_VS = FIRE_VS;
const SMOKE_FS = `varying float vLife; uniform sampler2D map;
void main(){ float d=length(gl_PointCoord-0.5); float a=smoothstep(0.5,0.0,d);
 float al=a*0.16*smoothstep(0.0,0.15,vLife)*(1.0-vLife); gl_FragColor=vec4(vec3(0.13,0.13,0.14),al); }`;

class Fire {
  constructor(x, y, z0) {
    this.x = x; this.y = y; this.z0 = z0; this.growth = 0; this.on = true;
    this.group = new THREE.Group();
    this.flame = this._points(520, FIRE_VS, FIRE_FS, THREE.AdditiveBlending);
    this.smoke = this._points(160, SMOKE_VS, SMOKE_FS, THREE.NormalBlending);
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
    s.base[k] = s.smoke ? 1.2 + Math.random() : (0.1 + Math.random() * 0.14) * (1 + 0.8 * this.growth);
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
  const g = new THREE.Group();
  const M = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.45, metalness: 0.35, ...o });
  const box = (sx, sy, sz, m, x, y, z) => { const b = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), m); b.position.set(x, y, z); b.castShadow = true; g.add(b); return b; };
  const cyl = (r, h, m, x, y, z, axis = 'z') => {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 24), m);
    if (axis === 'z') c.rotation.x = Math.PI / 2; else if (axis === 'x') c.rotation.z = Math.PI / 2;
    c.position.set(x, y, z); c.castShadow = true; g.add(c); return c;
  };
  const red = M(0xc4231a), dark = M(0x1d1f24, { roughness: 0.8 }), white = M(0xe9e9e4), black = M(0x050506, { roughness: 0.2 });
  box(0.42, 0.24, 0.11, red, 0, 0, 0.1);                       // chassis (heat-shielded shell)
  box(0.34, 0.2, 0.02, dark, -0.02, 0, 0.165);                  // deck
  for (const s of [-1, 1]) {                                     // tracks with treads
    box(0.46, 0.075, 0.1, dark, 0, 0.165 * s, 0.056);
    for (let k = 0; k < 12; k++) box(0.02, 0.08, 0.012, M(0x0c0c0e), -0.21 + k * 0.038, 0.165 * s, 0.108);
  }
  cyl(0.02, 0.15, dark, 0.06, 0, 0.25);                          // mast
  box(0.1, 0.16, 0.07, white, 0.08, 0, 0.33);                    // sensor head
  const lens = cyl(0.024, 0.02, black, 0.135, 0.035, 0.33, 'x');  // thermal camera
  cyl(0.03, 0.006, M(0xff8a3d, { emissive: 0x552200 }), 0.128, 0.035, 0.33, 'x');
  box(0.008, 0.06, 0.05, M(0xf4f4f4), 0.133, -0.035, 0.33);      // 60 GHz radar panel
  for (let k = 0; k < 6; k++) {                                  // 850 nm IR illuminator ring
    const a = k / 6 * 6.283;
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.005), M(0x330000, { emissive: 0xff2010, emissiveIntensity: 1.5 }));
    led.position.set(0.137, 0.035 + 0.034 * Math.cos(a), 0.33 + 0.034 * Math.sin(a)); g.add(led);
  }
  cyl(0.012, 0.02, black, 0.13, -0.035, 0.375);                  // depth camera
  for (const [x, y] of [[0.14, 0.1], [0.14, -0.1], [-0.14, 0.1], [-0.14, -0.1]]) cyl(0.008, 0.02, black, x, y, 0.18); // 4 mics
  cyl(0.035, 0.008, M(0x2a2d33), -0.09, 0, 0.18);               // speaker
  cyl(0.026, 0.009, M(0x111215), -0.09, 0, 0.181);
  cyl(0.012, 0.03, M(0x6c5ce7), 0.0, 0.125, 0.12, 'y');          // gas cells (CO + O2) vent
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.022, 16, 12), M(0xffa020, { emissive: 0xff8a00, emissiveIntensity: 2 }));
  beacon.position.set(-0.13, 0, 0.2); g.add(beacon);
  cyl(0.004, 0.18, dark, -0.15, 0.07, 0.27);                     // mesh-radio antenna
  g.userData = { beacon, lens };
  if (withLabels) {
    const L = [['Thermal camera', 's1', 0.14, 0.05, 0.36], ['60 GHz radar', 's2', 0.14, -0.06, 0.3],
      ['Depth + IR night camera', 's3', 0.13, -0.04, 0.4], ['Microphone array', 's4', 0.15, 0.12, 0.2],
      ['Speaker', 's5', -0.1, 0, 0.21], ['CO + O2 gas cells', 's6', 0, 0.15, 0.12],
      ['Self-temperature + battery', 's7', -0.05, -0.13, 0.12], ['Mesh radio', 's8', -0.15, 0.07, 0.37]];
    for (const [t, c, x, y, z] of L) { const l = label(t, ''); l.element.classList.add(c); l.position.set(x, y, z); g.add(l); }
  }
  return g;
}

// ------------------------------------------------------------------ world (rebuilt per scenario)
let world = null, SC = null, robot = null, fires = {}, thingMeshes = {}, smokeLayer = null, mapTexPlane = null;
const dyn = new THREE.Group(); scene.add(dyn);   // routes, markers: rebuilt from state
let showSmoke = true, showTruth = false, view = 'over';

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
  for (const [x0, y0, x1, y1] of sc.walls) {
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len < 0.01) continue;
    const w = new THREE.Mesh(new THREE.BoxGeometry(len + sc.wall_t, sc.wall_t, sc.wall_h), plaster);
    w.position.set((x0 + x1) / 2, (y0 + y1) / 2, sc.wall_h / 2);
    w.rotation.z = Math.atan2(y1 - y0, x1 - x0);
    w.castShadow = w.receiveShadow = true; world.add(w);
    const s = new THREE.Mesh(new THREE.BoxGeometry(len + sc.wall_t + 0.01, sc.wall_t + 0.012, 0.1), skirt);
    s.position.set(w.position.x, w.position.y, 0.05); s.rotation.z = w.rotation.z; world.add(s);
  }
  // robot-built map painted on the floor (toggle "Robot map")
  mapTexPlane = new THREE.Mesh(new THREE.PlaneGeometry(W, H),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55, depthWrite: false }));
  mapTexPlane.position.set(W / 2, H / 2, 0.015); mapTexPlane.visible = showTruth; world.add(mapTexPlane);
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
  const fov = new THREE.Mesh(new THREE.CircleGeometry(4, 32, -0.497, 0.995),
    new THREE.MeshBasicMaterial({ color: 0xff9a3d, transparent: true, opacity: 0.09, depthWrite: false }));
  fov.position.z = 0.02; robot.add(fov);
  const halo = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.36, 48), new THREE.MeshBasicMaterial({ color: 0x2ee6ff, transparent: true, opacity: 0.8 }));
  halo.position.z = 0.012; robot.add(halo);
  robot.userData.halo = halo;
  const rl = label('FireScout', 'robot'); rl.position.set(0, 0, 0.5); robot.add(rl);
  world.add(robot);
  // room numbers
  sc.rooms.forEach((r, k) => {
    const l = label(`R${k + 1} ${r.kind}`, ''); l.element.style.opacity = 0.7;
    l.position.set(r.box[0] + 0.7, r.box[3] - 0.35, 2.45); world.add(l);
  });
  const ent = label('ENTRY / BA control', ''); ent.position.set(sc.start[0], sc.start[1], 0.3); world.add(ent);
  setView(view, true);
  buildICButtons();
  $('#c-building').textContent = `${sc.name} · ${sc.width}x${sc.height} m · seed ${sc.seed}`;
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
  } else if (t.kind === 'fire') {
    const char = new THREE.MeshStandardMaterial({ color: 0x15100c, roughness: 1 });
    for (let k = 0; k < 6; k++) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.5 * Math.random() + 0.2, 0.08, 0.06), char);
      m.position.set((Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5, 0.04 + k * 0.02);
      m.rotation.z = Math.random() * 3; m.rotation.y = (Math.random() - 0.5) * 0.4; m.castShadow = true; g.add(m);
    }
    const ember = new THREE.Mesh(new THREE.CircleGeometry(0.4, 24), new THREE.MeshBasicMaterial({ color: 0xff4a0a }));
    ember.position.z = 0.012; g.add(ember); g.userData.ember = ember;
    fires[t.i] = new Fire(t.xy[0], t.xy[1], 0); world.add(fires[t.i].group);
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
function tube(pts, color, r, z) {
  if (!pts || pts.length < 2) return null;
  const v = pts.map((p) => new THREE.Vector3(p[0], p[1], z));
  const curve = new THREE.CatmullRomCurve3(v, false, 'centripetal');
  const m = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, v.length * 3), r, 8, false),
    new THREE.MeshBasicMaterial({ color, toneMapped: false }));
  m.userData.curve = curve;
  return m;
}
let lastRoutes = '', crewArrows = [];
function updateDynamic(st, t) {
  const evac = (st.things || []).filter((x) => x.route && x.route.length > 1).map((x) => x.route);
  const key = JSON.stringify([st.path, st.crew, st.exit, st.fires, st.tracks, st.voice, evac.map((r) => r.length)]);
  if (key !== lastRoutes) {
    lastRoutes = key;
    for (const c of [...dyn.children]) { dyn.remove(c); c.geometry?.dispose?.(); }
    crewArrows = [];
    const p = tube(st.path, 0x2ee6ff, 0.025, 0.06); if (p) dyn.add(p);
    const e = tube(st.exit, 0xffffff, 0.015, 0.035); if (e) { e.material.transparent = true; e.material.opacity = 0.55; dyn.add(e); }
    const c = tube(st.crew, 0x38ff7a, 0.05, 0.09);
    if (c) {
      dyn.add(c);
      for (let k = 0; k < 10; k++) {
        const a = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.22, 12), new THREE.MeshBasicMaterial({ color: 0xb8ffcf, toneMapped: false }));
        a.userData = { curve: c.userData.curve, off: k / 10 }; dyn.add(a); crewArrows.push(a);
      }
    }
    for (const f of st.fires || []) {
      const zone = new THREE.Mesh(new THREE.CircleGeometry(1.0, 40), new THREE.MeshBasicMaterial({ color: 0xff2a10, transparent: true, opacity: 0.18, depthWrite: false }));
      zone.position.set(f[0], f[1], 0.02); dyn.add(zone);
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1.05, 48), new THREE.MeshBasicMaterial({ color: 0xff4a3a, toneMapped: false }));
      ring.position.set(f[0], f[1], 0.025); dyn.add(ring);
      const l = label(`FIRE ${f[2]}C · NO-GO`, 'fire'); l.position.set(f[0], f[1], 1.9); dyn.add(l);
    }
    for (const tr of st.tracks || []) {
      const col = PRIO[tr.prio] ?? 0xffe14a;
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1.2, 8), new THREE.MeshBasicMaterial({ color: col, toneMapped: false }));
      beam.rotation.x = Math.PI / 2; beam.position.set(tr.xy[0], tr.xy[1], 0.6); dyn.add(beam);
      const ball = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), new THREE.MeshBasicMaterial({ color: col, toneMapped: false }));
      ball.position.set(tr.xy[0], tr.xy[1], 1.25); ball.userData.pulse = true; dyn.add(ball);
      const txt = tr.prio === 'EVAC' ? `#${tr.id} told to walk out` :
        tr.state === 'pinned' ? `${tr.prio} #${tr.id}${tr.rank ? ' · rescue ' + tr.rank : ''}${tr.say ? ' · told: ' + tr.say : tr.responsive ? ' · calling' : ''}`
        : `possible #${tr.id} · checking`;
      const l = label(txt, prioClass(tr.prio)); l.position.set(tr.xy[0], tr.xy[1], 1.5); dyn.add(l);
    }
    for (const r of evac) { const e2 = tube(r, 0xffe14a, 0.035, 0.08); if (e2) dyn.add(e2); }
    if (st.voice) {
      const [x, y, b] = st.voice;
      const v = tube([[x, y], [x + 4 * Math.cos(b), y + 4 * Math.sin(b)]], 0xff7cff, 0.02, 0.5);
      if (v) dyn.add(v);
    }
  }
  for (const a of crewArrows) {
    const u = (a.userData.off + t * 0.08) % 1, c = a.userData.curve;
    a.position.copy(c.getPointAt(u));
    const tan = c.getTangentAt(u);
    a.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tan);
  }
  for (const c of dyn.children) if (c.userData.pulse) c.scale.setScalar(1 + 0.35 * Math.sin(t * 6));
}

// ------------------------------------------------------------------ camera views
function setView(v, snap = false) {
  view = v;
  document.querySelectorAll('.viewbar [data-v]').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
  controls.enabled = v === 'over';
  if (v === 'over' && SC) {
    controls.target.set(SC.width / 2, SC.height / 2, 0);
    camera.position.set(SC.width / 2, -SC.height * 0.35, Math.max(SC.width * 0.95, SC.height * 1.5));
    camera.fov = 42; camera.near = 0.05; camera.updateProjectionMatrix();
  }
  if (v === 'eye') { camera.fov = 45; camera.near = 0.02; camera.updateProjectionMatrix(); }
  if (v === 'follow') { camera.fov = 55; camera.near = 0.03; camera.updateProjectionMatrix(); }
}
document.querySelectorAll('.viewbar [data-v]').forEach((b) => b.onclick = () => setView(b.dataset.v));
$('#t-smoke').onclick = (e) => { showSmoke = !showSmoke; e.target.classList.toggle('on', showSmoke); };
$('#t-truth').textContent = 'Robot map on floor';
$('#t-truth').onclick = (e) => { showTruth = !showTruth; e.target.classList.toggle('on', showTruth); if (mapTexPlane) mapTexPlane.visible = showTruth; };

// ------------------------------------------------------------------ panels
const LOGCOL = { RETREAT: '#ff6a5a', PIN: '#ff5a50', PRIORITY: '#ff8a6a', REPLAN: '#ffb02e', HEAR: '#ff7cff', IC: '#2ee6ff',
  COMMS: '#ffb02e', HOLD: '#ffb02e', VERIFY: '#ffe14a', POSSIBLE: '#ffe14a', REJECT: '#8b93a3', LEARN: '#9cff57', SAY: '#ff7cff' };
function chip(id, text, cls) { const c = $(id); c.textContent = text; c.className = 'chip ' + (cls || ''); }
const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

function updatePanels(st) {
  chip('#c-clock', 'T+' + fmt(st.t), 'big');
  if (!st.obs) return;
  chip('#c-crew', st.metrics.crew_in > 0 ? `crew enters in ${st.metrics.crew_in}s` : 'crew committed', st.metrics.crew_in > 0 ? '' : 'warn');
  chip('#c-mode', st.mode + (st.ic ? ' · IC override' : ''), ['RETREAT', 'IC_STOP', 'RECALL', 'HOLD'].includes(st.mode) ? 'warn' : 'ok');
  chip('#c-radio', st.obs.comms ? 'radio linked' : `radio LOST · ${st.obs.queued} queued`, st.obs.comms ? 'ok' : 'bad');
  const cas = (st.tracks || []).slice().sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
  $('#casualties').innerHTML = cas.length ? cas.map((c) => `<div class="cas ${prioClass(c.prio)}">
      <div class="t">${c.state === 'pinned' ? `${c.prio} · casualty #${c.id}` : `Possible person #${c.id}`}${c.rank ? ` · rescue ${c.rank}` : ''}</div>
      <div class="w">${c.why}${c.state !== 'pinned' ? ' · crew must check' : ''} · (${c.xy[0]}, ${c.xy[1]})</div>
      ${c.say ? `<div class="say">robot said: "${c.say}"</div>` : ''}</div>`).join('') : '<div class="muted">searching - none found yet</div>';
  $('#h-fires').textContent = st.fires.length ? st.fires.map((f) => `${f[2]}C`).join(', ') : 'none seen yet';
  $('#h-air').textContent = `CO ${st.obs.co} ppm · O2 ${st.obs.o2}%`;
  $('#h-air').style.color = st.obs.co > 400 || st.obs.o2 < 17 ? '#ff6a5a' : '';
  $('#h-health').textContent = `${st.obs.self_temp}C / 60 · battery ${Math.round(st.obs.battery * 100)}%`;
  $('#h-mic').textContent = st.voice ? 'VOICE HEARD - heading to it' : 'listening';
  $('#h-mic').style.color = st.voice ? '#ff7cff' : '';
  $('#a-speed').textContent = `${st.speed} m/s · ${st.speed_why}`;
  $('#a-cov').textContent = `${st.metrics.coverage}% (never declared empty)`;
  $('#a-first').textContent = st.metrics.first == null ? 'searching' : `T+${fmt(st.metrics.first)} (${Math.round(150 - st.metrics.first)}s before crew)`;
  $('#a-lat').textContent = `p50 ${st.metrics.lat50} ms · p95 ${st.metrics.lat95} ms`;
  const mp = st.metrics.map;
  $('#a-map').textContent = mp.precision == null ? '-' : `${mp.walls_found}% walls found · ${mp.precision}% correct`;
  const tr = st.metrics.truth;
  $('#a-truth').textContent = `${tr.n} people · ${tr.ok} confirmed · ${tr.fp} false · ${st.evacuated || 0} walked out`;
  const log = $('#log');
  if ((st.cards || []).join('|') === lastCards) return finishPanels(st, cas);
  lastCards = (st.cards || []).join('|');
  log.innerHTML = '';
  for (const c of st.cards || []) {
    const m = c.match(/^\[(\d\d:\d\d)\] (\w+) (.*)$/);
    const d = document.createElement('div');
    d.innerHTML = m ? `<span class="muted">${m[1]}</span> <span class="k" style="color:${LOGCOL[m[2]] || '#e8eaf0'}">${m[2]}</span> ${m[3].replace(/</g, '&lt;')}` : c;
    log.prepend(d);
  }
  finishPanels(st, cas);
}
let lastCards = '';
function finishPanels(st, cas) {
  $('#methane').textContent = (st.methane || []).join('\n');
  $('#msg').textContent = st.msg || '';
  // banner: what the crew must handle first
  const top = cas.find((c) => c.rank === 1);
  const b = [];
  if (top) b.push(`<div class="b">HANDLE FIRST: ${top.prio} casualty #${top.id} - ${top.why}</div>`);
  if (st.fires.length) { const f = st.fires.slice().sort((a, c) => c[2] - a[2])[0]; b.push(`<div class="b">FIRE ${f[2]}C at (${f[0]}, ${f[1]}) - no-go zone, route avoids it</div>`); }
  for (const s of st.say || []) b.push(`<div class="b say">ROBOT SPEAKER: "${s}"</div>`);
  $('#banner').innerHTML = b.join('');
}

// camera feeds + map
let feedName = 'thermal';
const CAP = { thermal: 'LWIR thermal 160x120 - sees bodies and fire through smoke', night: 'Near-IR night vision, 850 nm illuminator',
  colour: 'Colour camera - blinded by smoke' };
document.querySelectorAll('#feedtabs button').forEach((b) => b.onclick = () => {
  feedName = b.dataset.f; document.querySelectorAll('#feedtabs button').forEach((x) => x.classList.toggle('on', x === b));
  $('#feedcap').textContent = CAP[feedName];
});
const feedImg = $('#feed');
feedImg.onload = feedImg.onerror = () => setTimeout(() => { feedImg.src = `/feed/${feedName}.jpg?${Date.now()}`; }, 180);
const mapImg = new Image();
const mapCanvas = $('#mapc');
let mapTex = null;
mapImg.onload = mapImg.onerror = () => {
  if (mapImg.naturalWidth && SC) {
    const c = mapCanvas, g = c.getContext('2d');
    c.width = mapImg.naturalWidth; c.height = mapImg.naturalHeight;
    g.drawImage(mapImg, 0, 0);
    const sx = c.width / SC.width, sy = c.height / SC.height;
    g.strokeStyle = 'rgba(255,90,255,0.85)'; g.lineWidth = 1.5;
    g.beginPath();
    for (const [x0, y0, x1, y1] of SC.walls) { g.moveTo(x0 * sx, c.height - y0 * sy); g.lineTo(x1 * sx, c.height - y1 * sy); }
    g.stroke();
    if (mapTexPlane && showTruth) {
      mapTex?.dispose(); mapTex = new THREE.CanvasTexture(c); mapTex.colorSpace = THREE.SRGBColorSpace;
      mapTexPlane.material.map = mapTex; mapTexPlane.material.needsUpdate = true;
    }
  }
  setTimeout(() => { mapImg.src = `/feed/map.jpg?${Date.now()}`; }, 400);
};
mapImg.src = '/feed/map.jpg';

// IC buttons
document.querySelectorAll('[data-a]').forEach((b) => b.onclick = () => post({ action: b.dataset.a }));
function buildICButtons() {
  $('#rooms').innerHTML = '';
  SC.rooms.forEach((r, k) => {
    const b = document.createElement('button'); b.textContent = `R${k + 1} ${r.kind}`;
    b.onclick = () => post({ action: 'room', room: k }); $('#rooms').append(b);
  });
}
// scenario dialog
$('#b-new').onclick = () => {
  const s = $('#n-preset'); s.innerHTML = Object.entries(SC.presets).map(([k, v]) => `<option value="${k}" ${k === SC.preset ? 'selected' : ''}>${v}</option>`).join('');
  $('#d-new').showModal();
};
$('#n-cancel').onclick = () => $('#d-new').close();
document.querySelectorAll('.scen button').forEach((b) => b.onclick = () => {
  const [preset, cas, fires, seed] = b.dataset.s.split(',');
  post({ action: 'new', preset, casualties: +cas, fires: +fires, seed: +seed });
  $('#log').innerHTML = ''; $('#d-new').close();
});
$('#n-go').onclick = () => {
  post({ action: 'new', preset: $('#n-preset').value, casualties: +$('#n-cas').value, fires: +$('#n-fires').value, seed: +$('#n-seed').value });
  $('#log').innerHTML = ''; $('#d-new').close();
};
// sound (browser needs a click before audio may play)
let sound = false;
$('#b-sound').onclick = (e) => { sound = !sound; e.target.textContent = 'Sound: ' + (sound ? 'on' : 'off'); e.target.classList.toggle('on', sound); };
let audioBusy = 0;
function play(ev) {
  if (!sound || performance.now() < audioBusy) return;
  const lvl = ev.db >= 55 ? 'near' : ev.db >= 42 ? 'mid' : 'far';
  const a = new Audio(`/assets/audio/${ev.sound}${ev.sound.startsWith('robot_') ? '' : '_' + lvl}.wav`);
  a.play().catch(() => {});
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
  lab.domElement.style.position = 'absolute'; lab.domElement.style.pointerEvents = 'none';
  c.parentElement.style.position = 'relative'; c.parentElement.append(lab.domElement);
  lab.domElement.style.top = '0'; lab.domElement.style.left = '0';
  const s = new THREE.Scene();
  s.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.2));
  const dl = new THREE.DirectionalLight(0xffffff, 2); dl.position.set(2, -2, 3); s.add(dl);
  const rb = buildRobot(true); s.add(rb);
  const cam = new THREE.PerspectiveCamera(35, 1, 0.01, 20); cam.up.set(0, 0, 1); cam.position.set(1.0, -0.9, 0.7);
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
    robot.userData.halo.scale.setScalar(1 + 0.15 * Math.sin(t * 3));
    let maxG = 0;
    for (const th of ST.things) {
      const f = fires[th.i];
      if (f) { f.on = th.burning; f.growth = th.growth; if (th.burning) maxG = Math.max(maxG, th.growth); }
      const m = thingMeshes[th.i];
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
    updateDynamic(ST, t);
    if (view === 'follow') {
      const d = new THREE.Vector3(Math.cos(smooth.yaw), Math.sin(smooth.yaw), 0);
      const want = new THREE.Vector3(smooth.x, smooth.y, 0).addScaledVector(d, -2.0).add(new THREE.Vector3(0, 0, 1.6));
      camera.position.lerp(want, 1 - Math.exp(-dt * 4));
      camera.lookAt(smooth.x + d.x * 1.2, smooth.y + d.y * 1.2, 0.2);
    } else if (view === 'eye') {
      const d = new THREE.Vector3(Math.cos(smooth.yaw), Math.sin(smooth.yaw), 0);
      camera.position.set(smooth.x + d.x * 0.17, smooth.y + d.y * 0.17, 0.36);
      camera.lookAt(smooth.x + d.x * 3, smooth.y + d.y * 3, 0.25);
    } else controls.update();
  }
  composer.render();
  labels.render(scene, camera);
  if (showcase && $('#d-robot').open) showcase();
}
resize();
frame();
