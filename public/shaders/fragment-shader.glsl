
uniform vec4 uX;        // (t,x,y,z)
uniform vec4 uUr;        // (0, e_right)
uniform vec4 uUu;        // (0, e_up)
uniform vec4 uUf;        // forward tangent
const float uFovY = 1.0472;       // vertical FOV in radians (e.g., 60 degrees)
const float uAspect = 1.0;     // width/height
uniform float uTime;
varying vec2 vUv;

struct Material { vec3 ka, kd, ks; float shin; };

// --- Lorentz helpers ---
float ldot(vec4 a, vec4 b) { return -a.x*b.x + dot(a.yzw, b.yzw); }

// Length in the tangent metric (positive-definite on tangents)
float lnorm(vec4 w) { return sqrt(max(ldot(w, w), 1e-12)); }





// Project a 4D tangent vector into the camera's 3D tangent basis (Ur,Uu,Uf)
vec3 toFrame(vec4 V, vec4 Ur, vec4 Uu, vec4 Uf){
  return vec3(ldot(V, Ur), ldot(V, Uu), ldot(V, Uf));
}


float acosh_safe(float x) {
  float xm1 = max(x - 1.0, 0.0);
  float xp1 = x + 1.0;
  return log(x + sqrt(xm1) * sqrt(max(xp1, 0.0)));
}

float asinh_stable(float x) {
  float ax = abs(x);
  float y  = log(ax + sqrt(ax*ax + 1.0));
  return (x >= 0.0) ? y : -y;  // preserves sign; avoids cancellation for x<0
}





// Orthonormal basis perpendicular to uForward:
void makeRightUp(in vec3 n, out vec3 e1, out vec3 e2) {
  vec3 a = (abs(n.y) < 0.99) ? vec3(0,1,0) : vec3(1,0,0);
  e1 = normalize(a - dot(a, n) * n);      // right
  e2 = normalize(cross(n, e1));           // up
}

vec4 pixelDirection(vec2 uv, vec4 Ur, vec4 Uu, vec4 Uf) {
  // uv in [0,1]^2  -> NDC in [-1,1]^2
  vec2 ndc = 2.0 * uv - 1.0;

  // Pinhole slopes from FOV and aspect
  float sy = tan(0.5 * uFovY);
  float sx = uAspect * sy;

  // Raw tangent direction in the camera frame
  vec4 W = ndc.x * sx * Ur + ndc.y * sy * Uu + Uf;

  // Unit-speed (so 1 unit of parameter = 1 unit of hyperbolic distance)
  return W / lnorm(W);
}

void stepGeodesic(inout vec4 X, inout vec4 U, float d) {
  float c = 0.5 * (exp(d) + exp(-d));   // cosh(d)
  float s = 0.5 * (exp(d) - exp(-d));   // sinh(d)
  vec4 Xn = c * X + s * U;
  vec4 Un = s * X + c * U;
  X = Xn; U = Un;
}



float sdf_hSphereC(vec4 X, vec4 C, float R){
  float coshDist = -ldot(X, C);
  return acosh_safe(coshDist) - R;
}

// Signed distance to a centered slab, orthogonal to N, with half-size 'a'
// C anchors the slab's mid-plane: delta = 0 at X = C
float sd_hSlab(vec4 X, vec4 N, vec4 C, float a)
{
  // Signed distance to N-midplane through C
  float delta = asinh_stable(ldot(X, N)) - asinh_stable(ldot(C, N));
  // Slab SDF (two parallel planes at +/- a)
  return abs(delta) - a;
}

// N1,N2,N3 must be unit spacelike and mutually orthogonal (i.e. basis for tangent space)
float sd_hBox(vec4 X, vec4 C, vec4 N1, vec4 N2, vec4 N3, vec3 halfA)
{
  float d1 = sd_hSlab(X, N1, C, halfA.x);
  float d2 = sd_hSlab(X, N2, C, halfA.y);
  float d3 = sd_hSlab(X, N3, C, halfA.z);
  return max(max(d1, d2), d3);  
}

float world_distance(vec4 X) {
  vec3 right, up, forward;
  forward = vec3(0.0, 0.0, 1.0);
  makeRightUp(forward, right, up);  // right = makeRightUp(forward)


  vec4 spherepos1 = vec4(cosh(2.0), sinh(2.0) * forward);
  vec4 spherepos2 = vec4(cosh(2.0), sinh(2.0) * right);
  vec4 boxCenter = vec4(cosh(2.0), sinh(2.0) * forward);
  vec4 boxN1 = vec4(0.0, 1.0, 0.0, 0.0);
  vec4 boxN2 = vec4(0.0, 0.0, 1.0, 0.0);
  vec4 boxN3 = vec4(0.0, 0.0, 0.0, 1.0);
  vec3 boxHalfSize = vec3(0.5, 0.5, 0.5);



  return min(sd_hBox(X, boxCenter, boxN1, boxN2, boxN3, boxHalfSize), 
      sdf_hSphereC(X, spherepos2, 0.5));

  // return min(
  //   sdf_hSphereC(X, spherepos1, 0.5),
  //   min(
  //     sdf_hSphereC(X, spherepos2, 0.5),
  //     sd_hBox(X, boxCenter, boxN1, boxN2, boxN3, boxHalfSize)
  //   )
  // );
}

// Sample SDF at a point displaced by s along unit tangent W
float sampleAlong(vec4 X, vec4 W, float s){
  vec4 Xt = X, Wt = W;
  stepGeodesic(Xt, Wt, s);
  return world_distance(Xt);            // your hyperbolic SDF
}

// Gradient (df/dr, df/du, df/dfwd) in the tangent frame at X
vec3 sdfGradFrame(vec4 X, vec4 Ur, vec4 Uu, vec4 Uf){
  const float eps = 1e-3;               // hyperbolic step for FD
  float dr = (sampleAlong(X, Ur, +eps) - sampleAlong(X, Ur, -eps)) / (2.0*eps);
  float du = (sampleAlong(X, Uu, +eps) - sampleAlong(X, Uu, -eps)) / (2.0*eps);
  float df = (sampleAlong(X, Uf, +eps) - sampleAlong(X, Uf, -eps)) / (2.0*eps);
  return vec3(dr, du, df);
}

// Turn that gradient into a unit surface normal in frame coordinates
vec3 frameNormal(vec3 grad, vec3 viewDir3){
  // For isosurfaces, gradient points outward; flip to face the viewer if needed
  vec3 n = normalize(grad);
  if(dot(n, viewDir3) < 0.0) n = -n;
  return n;
}

vec3 ray_march_hyperbolic(vec4 X0, vec4 U0) {
  vec4 X = X0;
  vec4 U = U0;

  const int   MAX_STEPS = 1024;
  const float EPS       = 1e-5;    // hit threshold in hyperbolic units
  const float MAX_DIST  = 30.0;    // travel budget in hyperbolic units
  const float SAFETY    = 1.0;     // 1.0 for exact SDFs; <1 if you use smooth blends

  float traveled = 0.0;

  for (int i = 0; i < MAX_STEPS; ++i) {
    float d = world_distance(X);       // hyperbolic signed distance
    if (d < EPS) {

      vec3 grad = sdfGradFrame(X, uUr, uUu, uUf);
      vec3 v3   = toFrame(-U, uUr, uUu, uUf);         // view toward camera
      vec3 n3   = normalize((dot(grad, v3) < 0.0) ? -grad : grad);
    return(abs(n3));



    }
    if (traveled > MAX_DIST) break;

    float stepLen = SAFETY * d;
    stepGeodesic(X, U, stepLen);
    traveled += stepLen;


  }

  return vec3(0.0);                 // miss = black
}




void main() {
  // vec4 X0, Ur, Uu, Uf;
  // cameraFrame(X0, Ur, Uu, Uf);

  vec4 U0 = pixelDirection(vUv, uUr, uUu, uUf);

  vec3 color = ray_march_hyperbolic(uX, U0); 
  gl_FragColor = vec4(color, 1.0);
}
