
async function loadThree() {
  try {

    return await import('/node_modules/three/build/three.module.js');
  } catch (e) {
    console.log('Failed to load Three.js from node_modules, trying CDN...');

    return await import('https://cdn.jsdelivr.net/npm/three@0.154.0/build/three.module.js');
  }
}

// Declare shader variables
let vertexShader, fragmentShader;
let THREE;

// ---- Lorentz/minkowski helpers ----
function ldot(a, b) { return -a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }
function lnorm(a) { return Math.sqrt(Math.max(ldot(a, a), 1e-18)); }
const L2   = v => Math.sqrt(Math.max(ldot(v,v),1e-18));

const lorentzLen = v => Math.sqrt(Math.max(ldot(v,v), 1e-18));
function scale(v,k){ return new Float32Array([k*v[0],k*v[1],k*v[2],k*v[3]]); }
function add(a,b){ return new Float32Array([a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]]); }
function sub(a,b){ return new Float32Array([a[0]-b[0],a[1]-b[1],a[2]-b[2],a[3]-b[3]]); }

// Hyperbolic cosh/sinh
const cosh = Math.cosh, sinh = Math.sinh, acosh = Math.acosh;

// Normalize a 3D vector
function norm3(v) {
  const L = Math.hypot(v.x, v.y, v.z) || 1.0;
  return new THREE.Vector3(v.x / L, v.y / L, v.z / L);
}

function ensureOnSheet(X){
  const m = -ldot(X,X);
  const sc = 1 / Math.sqrt(Math.max(m, 1e-18));
  X[0]*=sc; X[1]*=sc; X[2]*=sc; X[3]*=sc;
  return X;
}

// Make V tangent & unit at X (kept for non-rotation utilities)
function tangentUnitAt(X, V){
  const k = ldot(X, V);
  const T = sub(V, scale(X, k));
  let L2v = ldot(T,T);
  if (L2v < 1e-12) {
    const seed = new Float32Array([0,0,0,1]); // (0,0,0,1)
    const F = sub(seed, scale(X, ldot(X,seed)));
    L2v = ldot(F,F);
    const inv = 1 / Math.sqrt(Math.max(L2v, 1e-18));
    return scale(F, inv);
  }
  return scale(T, 1 / Math.sqrt(L2v));
}

// Build an orthonormal tangent frame at X for right/up/fwd
// n is the spatial unit direction associated with X (i.e., s/|s|).
function tangentFrameAtX(X, n) {
  const rho = acosh(Math.max(X[0], 1.0));
  const ch = cosh(rho), sh = sinh(rho);

  // Pick a world-up not parallel to n
  const worldUp = Math.abs(n.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const e1 = worldUp.clone().sub(n.clone().multiplyScalar(worldUp.dot(n))).normalize(); // right (3D)
  const e2 = new THREE.Vector3().crossVectors(n, e1).normalize();                      // up (3D)

  // Lift to 4D tangents: (0,e1), (0,e2), and forward = (sinh, cosh*n)
  const Ur = new Float32Array([0, e1.x, e1.y, e1.z]);
  const Uu = new Float32Array([0, e2.x, e2.y, e2.z]);
  const Uf = new Float32Array([sh, ch * n.x, ch * n.y, ch * n.z]);

  return { Ur, Uu, Uf };
}

// -------- Translation pipeline  --------

function transportFrame(X_old, Umove_old, d, frameOld){
  const { Xn: X_new, Un: Umove_new } = stepGeodesic(X_old, Umove_old, d);

  function transportVec(V){
    const alpha = ldot(V, Umove_old);
    const Vp = new Float32Array([
      V[0]-alpha*Umove_old[0],
      V[1]-alpha*Umove_old[1],
      V[2]-alpha*Umove_old[2],
      V[3]-alpha*Umove_old[3],
    ]);
    const Vn = new Float32Array([
      alpha*Umove_new[0]+Vp[0],
      alpha*Umove_new[1]+Vp[1],
      alpha*Umove_new[2]+Vp[2],
      alpha*Umove_new[3]+Vp[3],
    ]);
    return Vn;
  }

  let Ur_new = transportVec(frameOld.Ur);
  let Uu_new = transportVec(frameOld.Uu);
  let Uf_new = transportVec(frameOld.Uf);

  function normL(V){
    const n = L2(V);
    V[0]/=n; V[1]/=n; V[2]/=n; V[3]/=n;
    return V;
  }
  Ur_new = normL(Ur_new);
  Uu_new = normL(Uu_new);
  Uf_new = normL(Uf_new);


  ({ Ur: Ur_new, Uu: Uu_new, Uf: Uf_new } = renormFrame({ Ur: Ur_new, Uu: Uu_new, Uf: Uf_new }));

  return { X_new, Ur_new, Uu_new, Uf_new };
}

// Combine a frame with joystick-style weights to a unit tangent
function unitTangentFromFrame(weights, Ur, Uu, Uf) {
  const { rx, ry, fwd } = weights; // right, up, forward scalars
  const W = new Float32Array([
    rx * Ur[0] + ry * Uu[0] + fwd * Uf[0],
    rx * Ur[1] + ry * Uu[1] + fwd * Uf[1],
    rx * Ur[2] + ry * Uu[2] + fwd * Uf[2],
    rx * Ur[3] + ry * Uu[3] + fwd * Uf[3],
  ]);
  const L = lnorm(W) || 1.0;
  W[0] /= L; W[1] /= L; W[2] /= L; W[3] /= L;   // now <W,W>_L = 1
  return W;
}

function stepGeodesic(X, U, d) {
  const c = Math.cosh(d), s = Math.sinh(d);
  const Xn = new Float32Array([c * X[0] + s * U[0], c * X[1] + s * U[1], c * X[2] + s * U[2], c * X[3] + s * U[3]]);
  const Un = new Float32Array([s * X[0] + c * U[0], s * X[1] + c * U[1], s * X[2] + c * U[2], s * X[3] + c * U[3]]);

  // Snap Xn back on the sheet 
  const m  = -ldot(Xn, Xn);
  const sc = 1 / Math.sqrt(Math.max(m, 1e-18));
  Xn[0]*=sc; Xn[1]*=sc; Xn[2]*=sc; Xn[3]*=sc;


  return { Xn, Un };
}

// -------- Stable frame-only rotations  --------

// Spacelike unit via Lorentz norm (frame-only)
function unitSpacelike(v){
  let L2v = ldot(v,v);
  if (!(L2v > 1e-24)) return new Float32Array([0,0,0,1]); // fallback
  return scale(v, 1/Math.sqrt(L2v));
}

// Gram–Schmidt within the frame only (keeps all three tangent, no X used)
function renormFrame(F){ // F = {Ur,Uu,Uf}
  // 1) normalize Uf
  let Uf = unitSpacelike(F.Uf);

  // 2) make Ur ⟂ Uf, normalize
  let Ur = sub(F.Ur, scale(Uf, ldot(F.Ur, Uf)));
  Ur = unitSpacelike(Ur);

  // 3) make Uu ⟂ Uf and Ur, normalize
  let Uu = sub(F.Uu, scale(Uf, ldot(F.Uu, Uf)));
  Uu = sub(Uu, scale(Ur, ldot(Uu, Ur)));
  Uu = unitSpacelike(Uu);

  return { Ur, Uu, Uf };
}

// 2×2 rotation of a pair 
function rotPair(A, B, ang){
  const c = Math.cos(ang), s = Math.sin(ang);
  const A2 = add(scale(A, c), scale(B, s));
  const B2 = add(scale(A, -s), scale(B, c));
  return [A2, B2];
}

// Pure frame rotation: yaw (Ur↔Uf), pitch (Uu↔Uf), roll (Ur↔Uu), Frame = the 3 vectors
function rotateFrame(F, yaw, pitch, roll){
  let Ur = F.Ur, Uu = F.Uu, Uf = F.Uf;

  if (yaw)   { [Ur, Uf] = rotPair(Ur, Uf, yaw); }       // about Uu
  if (pitch) { [Uu, Uf] = rotPair(Uu, Uf, -pitch); }    // about Ur (sign per right-handed)
  if (roll)  { [Ur, Uu] = rotPair(Ur, Uu, roll); }      // about Uf

  return renormFrame({ Ur, Uu, Uf });
}

// Function to initialize the scene once shaders are loaded
function init() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  camera.position.z = 2;

  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(Math.min(window.innerWidth, window.innerHeight), Math.min(window.innerWidth, window.innerHeight));
  document.body.appendChild(renderer.domElement);

  const initX0 = new THREE.Vector4(1, 0, 0, 0);
  const initComponents = tangentFrameAtX(initX0.toArray(), new THREE.Vector3(0, 0, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader: vertexShader,
    fragmentShader: fragmentShader,
    uniforms: {
      uTime: { value: 0.0 },
      uPos: { value: new THREE.Vector3(0, 0, -5) },
      // New hyperbolic camera uniforms:
      uX:  { value: initX0 },                                // (t,x,y,z)
      uUr: { value: new THREE.Vector4(...initComponents.Ur) }, // (0, e_right)
      uUu: { value: new THREE.Vector4(...initComponents.Uu) }, // (0, e_up)
      uUf: { value: new THREE.Vector4(...initComponents.Uf) }, // forward tangent
    }
  });

  const geometry = new THREE.PlaneGeometry(2, 2, 1, 1);
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // Movement & turn state
  const move = { forward: false, backward: false, left: false, right: false, up: false, down: false };
  const turn = { yawL:false, yawR:false, pitchU:false, pitchD:false, rollL:false, rollR:false };
  const yawRate   = Math.PI * 0.8;  // rad/s
  const pitchRate = Math.PI * 0.6;
  const rollRate  = Math.PI * 0.6;
  const speed = 0.5; // units per second

  window.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    switch (e.code) {
      case 'KeyW': move.forward = true; break;
      case 'KeyS': move.backward = true; break;
      case 'KeyA': move.left = true; break;
      case 'KeyD': move.right = true; break;
      case 'Space': move.up = true; break;
      case 'ShiftLeft':
      case 'ShiftRight': move.down = true; break;
    }
  });
  window.addEventListener('keyup', (e) => {
    switch (e.code) {
      case 'KeyW': move.forward = false; break;
      case 'KeyS': move.backward = false; break;
      case 'KeyA': move.left = false; break;
      case 'KeyD': move.right = false; break;
      case 'Space': move.up = false; break;
      case 'ShiftLeft':
      case 'ShiftRight': move.down = false; break;
    }
  });

  window.addEventListener('keydown', e => {
    if (e.repeat) return;
    if (e.code==='ArrowLeft')  turn.yawL = true;
    if (e.code==='ArrowRight') turn.yawR = true;
    if (e.code==='ArrowUp')    turn.pitchU = true;
    if (e.code==='ArrowDown')  turn.pitchD = true;
    if (e.code==='KeyQ')       turn.rollL = true;
    if (e.code==='KeyE')       turn.rollR = true;
  });
  window.addEventListener('keyup', e => {
    if (e.code==='ArrowLeft')  turn.yawL = false;
    if (e.code==='ArrowRight') turn.yawR = false;
    if (e.code==='ArrowUp')    turn.pitchU = false;
    if (e.code==='ArrowDown')  turn.pitchD = false;
    if (e.code==='KeyQ')       turn.rollL = false;
    if (e.code==='KeyE')       turn.rollR = false;
  });

  window.eStartTime = Date.now();
  let lastTime = window.eStartTime;

  function animate() {
    requestAnimationFrame(animate);

    const now = Date.now();
    const elapsedTime = (now - window.eStartTime) / 1000; // seconds
    const delta = (now - lastTime) / 1000;
    lastTime = now;

    // ---------- MOVEMENT ----------
    let rx = 0, ry = 0, fwd = 0;
    if (move.right)   rx += 1;
    if (move.left)    rx -= 1;
    if (move.up)      ry += 1;
    if (move.down)    ry -= 1;
    if (move.forward) fwd += 1;
    if (move.backward)fwd -= 1;

    const mag = Math.hypot(rx, ry, fwd);
    if (mag > 0) {
      rx /= mag; ry /= mag; fwd /= mag;

      const Umove = unitTangentFromFrame(
        { rx, ry, fwd },
        material.uniforms.uUr.value.toArray(),
        material.uniforms.uUu.value.toArray(),
        material.uniforms.uUf.value.toArray()
      );

      const moved = transportFrame(
        material.uniforms.uX.value.toArray(),
        Umove,
        speed * delta,
        {
          Ur: material.uniforms.uUr.value.toArray(),
          Uu: material.uniforms.uUu.value.toArray(),
          Uf: material.uniforms.uUf.value.toArray()
        }
      );

      // Write back pose after translation
      material.uniforms.uX.value .set(moved.X_new[0],  moved.X_new[1],  moved.X_new[2],  moved.X_new[3]);
      material.uniforms.uUr.value.set(moved.Ur_new[0], moved.Ur_new[1], moved.Ur_new[2], moved.Ur_new[3]);
      material.uniforms.uUu.value.set(moved.Uu_new[0], moved.Uu_new[1], moved.Uu_new[2], moved.Uu_new[3]);
      material.uniforms.uUf.value.set(moved.Uf_new[0], moved.Uf_new[1], moved.Uf_new[2], moved.Uf_new[3]);
    }

    // ---------- ROTATION ----------
    {
      let F = {
        Ur: material.uniforms.uUr.value.toArray(),
        Uu: material.uniforms.uUu.value.toArray(),
        Uf: material.uniforms.uUf.value.toArray()
      };

      const yawAng   = ((turn.yawL  ? 1 : 0) + (turn.yawR  ? -1 : 0)) * yawRate   * delta;
      const pitchAng = ((turn.pitchU? 1 : 0) + (turn.pitchD? -1 : 0)) * pitchRate * delta;
      const rollAng  = ((turn.rollL ? 1 : 0) + (turn.rollR ? -1 : 0)) * rollRate  * delta;

      if (yawAng || pitchAng || rollAng){
        F = rotateFrame(F, yawAng, pitchAng, rollAng);

        material.uniforms.uUr.value.set(F.Ur[0], F.Ur[1], F.Ur[2], F.Ur[3]);
        material.uniforms.uUu.value.set(F.Uu[0], F.Uu[1], F.Uu[2], F.Uu[3]);
        material.uniforms.uUf.value.set(F.Uf[0], F.Uf[1], F.Uf[2], F.Uf[3]);
      }
    }

    material.uniforms.uTime.value = elapsedTime;
    renderer.render(scene, camera);
  }

  animate();

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(Math.min(window.innerWidth, window.innerHeight), Math.min(window.innerWidth, window.innerHeight));
  });
}

// Load everything needed
Promise.all([
  loadThree(),
  fetch('shaders/vertex-shader.glsl').then(r => r.text()),
  fetch('shaders/fragment-shader.glsl').then(r => r.text())
]).then(([threeModule, vertex, fragment]) => {
  THREE = threeModule;
  vertexShader = vertex;
  fragmentShader = fragment;
  init();
});
