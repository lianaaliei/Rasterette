/*
 * raster.c — a software 3D renderer written from scratch in plain C99.
 *
 * No graphics API, no libraries (only libm). Everything between "triangle" and
 * "PNG file" is implemented here:
 *
 *   - 4x4 matrices, look-at camera, perspective + orthographic projection
 *   - homogeneous near-plane clipping (Sutherland–Hodgman)
 *   - edge-function triangle rasterizer with perspective-correct interpolation
 *   - z-buffer, back-face culling
 *   - procedural meshes (sphere, cube, torus knot, floor) and an OBJ loader
 *   - per-pixel Blinn-Phong, procedural textures (checker, marble, wood, palette)
 *   - sky-gradient reflections with Fresnel, distance fog
 *   - shadow mapping (orthographic light, 5x5 PCF)
 *   - supersampling anti-aliasing, ACES tone mapping, gamma, vignette
 *   - a PNG encoder (stored deflate blocks, CRC32, Adler-32)
 *
 * Build:   gcc -O3 -march=native -o raster raster.c -lm
 * Run:     ./raster                        -> render.png (1280x720)
 *          ./raster -w 1920 -h 1080 -s 3   -> bigger, smoother
 *          ./raster --frames 60 -o spin    -> spin_000.png ... spin_059.png (turntable)
 *          ./raster --obj model.obj        -> replaces the torus knot with your OBJ
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979f
#define NV 8              /* varyings per vertex: world pos(3), normal(3), uv(2) */
#define SM_RES 2048       /* shadow map resolution */
#define SM_RADIUS 8.0f    /* half-width of the orthographic shadow frustum */

/* ------------------------------------------------------------------ math */
typedef struct { float x, y, z; } V3;
typedef struct { float x, y, z, w; } V4;
typedef struct { float m[16]; } M4; /* row-major, column vectors: out = M * v */

static V3 v3(float x, float y, float z) { V3 r = {x, y, z}; return r; }
static V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 mul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static V3 vmul(V3 a, V3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static float len(V3 a) { return sqrtf(dot(a, a)); }
static V3 norm(V3 a) { float l = len(a); return l > 0 ? mul(a, 1.0f / l) : a; }
static float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static float sat(float x) { return clampf(x, 0.0f, 1.0f); }
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static V3 mix3(V3 a, V3 b, float t) { return add(mul(a, 1.0f - t), mul(b, t)); }
static float smooth01(float a, float b, float x) {
    float t = sat((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

static M4 m4_identity(void) {
    M4 r; memset(&r, 0, sizeof r);
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}
static M4 m4_mul(M4 a, M4 b) {
    M4 r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[i * 4 + k] * b.m[k * 4 + j];
            r.m[i * 4 + j] = s;
        }
    return r;
}
static M4 m4_translate(V3 t) {
    M4 r = m4_identity();
    r.m[3] = t.x; r.m[7] = t.y; r.m[11] = t.z;
    return r;
}
static M4 m4_scale(float s) {
    M4 r = m4_identity();
    r.m[0] = r.m[5] = r.m[10] = s;
    return r;
}
static M4 m4_rotx(float a) {
    M4 r = m4_identity(); float c = cosf(a), s = sinf(a);
    r.m[5] = c; r.m[6] = -s; r.m[9] = s; r.m[10] = c;
    return r;
}
static M4 m4_roty(float a) {
    M4 r = m4_identity(); float c = cosf(a), s = sinf(a);
    r.m[0] = c; r.m[2] = s; r.m[8] = -s; r.m[10] = c;
    return r;
}
static V4 m4_mulp(M4 m, V3 p) {
    V4 r;
    r.x = m.m[0] * p.x + m.m[1] * p.y + m.m[2] * p.z + m.m[3];
    r.y = m.m[4] * p.x + m.m[5] * p.y + m.m[6] * p.z + m.m[7];
    r.z = m.m[8] * p.x + m.m[9] * p.y + m.m[10] * p.z + m.m[11];
    r.w = m.m[12] * p.x + m.m[13] * p.y + m.m[14] * p.z + m.m[15];
    return r;
}
static V3 m4_point(M4 m, V3 p) { V4 r = m4_mulp(m, p); return v3(r.x, r.y, r.z); }
static V3 m4_dir(M4 m, V3 d) {
    return v3(m.m[0] * d.x + m.m[1] * d.y + m.m[2] * d.z,
              m.m[4] * d.x + m.m[5] * d.y + m.m[6] * d.z,
              m.m[8] * d.x + m.m[9] * d.y + m.m[10] * d.z);
}
/* View space: x right, y up, z forward (into the screen). */
static M4 m4_lookat(V3 eye, V3 target, V3 up) {
    V3 f = norm(sub(target, eye));
    V3 r = norm(cross(up, f));
    V3 u = cross(f, r);
    M4 m = m4_identity();
    m.m[0] = r.x; m.m[1] = r.y; m.m[2] = r.z; m.m[3] = -dot(r, eye);
    m.m[4] = u.x; m.m[5] = u.y; m.m[6] = u.z; m.m[7] = -dot(u, eye);
    m.m[8] = f.x; m.m[9] = f.y; m.m[10] = f.z; m.m[11] = -dot(f, eye);
    return m;
}
/* Clip-space depth runs 0 (near) .. far, so z/w is in [0,1]; w = view-space z. */
static M4 m4_perspective(float fovy, float aspect, float n, float f) {
    float t = 1.0f / tanf(fovy * 0.5f);
    M4 m; memset(&m, 0, sizeof m);
    m.m[0] = t / aspect; m.m[5] = t;
    m.m[10] = f / (f - n); m.m[11] = -f * n / (f - n);
    m.m[14] = 1.0f;
    return m;
}
static M4 m4_ortho(float radius, float n, float f) {
    M4 m = m4_identity();
    m.m[0] = 1.0f / radius; m.m[5] = 1.0f / radius;
    m.m[10] = 1.0f / (f - n); m.m[11] = -n / (f - n);
    return m;
}

/* ------------------------------------------------------------- meshes */
typedef struct { V3 p, n; float u, v; } Vertex;
typedef struct {
    Vertex *v; int nv, cv;
    uint32_t *i; int ni, ci;
} Mesh;

static void mesh_vert(Mesh *m, V3 p, V3 n, float u, float v) {
    if (m->nv == m->cv) {
        m->cv = m->cv ? m->cv * 2 : 256;
        m->v = realloc(m->v, (size_t)m->cv * sizeof(Vertex));
    }
    Vertex x = {p, n, u, v};
    m->v[m->nv++] = x;
}
static void mesh_tri(Mesh *m, uint32_t a, uint32_t b, uint32_t c) {
    if (m->ni + 3 > m->ci) {
        m->ci = m->ci ? m->ci * 2 : 768;
        m->i = realloc(m->i, (size_t)m->ci * sizeof(uint32_t));
    }
    m->i[m->ni++] = a; m->i[m->ni++] = b; m->i[m->ni++] = c;
}
/* Make every triangle's winding agree with its vertex normals, so culling just works. */
static void fix_winding(Mesh *m) {
    for (int t = 0; t < m->ni; t += 3) {
        Vertex *a = &m->v[m->i[t]], *b = &m->v[m->i[t + 1]], *c = &m->v[m->i[t + 2]];
        V3 gn = cross(sub(b->p, a->p), sub(c->p, a->p));
        V3 nn = add(add(a->n, b->n), c->n);
        if (dot(gn, nn) < 0) { uint32_t s = m->i[t + 1]; m->i[t + 1] = m->i[t + 2]; m->i[t + 2] = s; }
    }
}

static Mesh make_sphere(int stacks, int slices) {
    Mesh m; memset(&m, 0, sizeof m);
    for (int i = 0; i <= stacks; i++)
        for (int j = 0; j <= slices; j++) {
            float th = PI * i / stacks, ph = 2.0f * PI * j / slices;
            V3 p = v3(sinf(th) * cosf(ph), cosf(th), sinf(th) * sinf(ph));
            mesh_vert(&m, p, p, (float)j / slices, (float)i / stacks);
        }
    for (int i = 0; i < stacks; i++)
        for (int j = 0; j < slices; j++) {
            uint32_t a = i * (slices + 1) + j, b = (i + 1) * (slices + 1) + j;
            mesh_tri(&m, a, b, a + 1);
            mesh_tri(&m, b, b + 1, a + 1);
        }
    fix_winding(&m);
    return m;
}

static Mesh make_cube(void) {
    Mesh m; memset(&m, 0, sizeof m);
    V3 axes[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int f = 0; f < 6; f++) {
        V3 n = axes[f];
        V3 t = fabsf(n.y) > 0.9f ? v3(1, 0, 0) : v3(0, 1, 0);
        V3 r = norm(cross(t, n)), u = cross(n, r);
        uint32_t base = (uint32_t)m.nv;
        mesh_vert(&m, add(n, add(mul(r, -1), mul(u, -1))), n, 0, 0);
        mesh_vert(&m, add(n, add(mul(r, 1), mul(u, -1))), n, 1, 0);
        mesh_vert(&m, add(n, add(mul(r, 1), mul(u, 1))), n, 1, 1);
        mesh_vert(&m, add(n, add(mul(r, -1), mul(u, 1))), n, 0, 1);
        mesh_tri(&m, base, base + 1, base + 2);
        mesh_tri(&m, base, base + 2, base + 3);
    }
    fix_winding(&m);
    return m;
}

static Mesh make_floor(float s) {
    Mesh m; memset(&m, 0, sizeof m);
    V3 up = v3(0, 1, 0);
    mesh_vert(&m, v3(-s, 0, -s), up, 0, 0);
    mesh_vert(&m, v3(-s, 0, s), up, 0, 1);
    mesh_vert(&m, v3(s, 0, -s), up, 1, 0);
    mesh_vert(&m, v3(s, 0, s), up, 1, 1);
    mesh_tri(&m, 0, 1, 2);
    mesh_tri(&m, 1, 3, 2);
    return m;
}

static V3 knot_p(float u, float p, float q, float R) {
    float cu = cosf(u), su = sinf(u), qp = q / p * u, cs = cosf(qp);
    return v3(R * (2 + cs) * 0.5f * cu, R * (2 + cs) * 0.5f * su, R * sinf(qp) * 0.5f);
}
static Mesh make_knot(int tubular, int radial, float p, float q, float R, float tube) {
    Mesh m; memset(&m, 0, sizeof m);
    for (int i = 0; i <= tubular; i++) {
        float u = (float)i / tubular * p * 2.0f * PI;
        V3 p1 = knot_p(u, p, q, R), p2 = knot_p(u + 0.01f, p, q, R);
        V3 T = sub(p2, p1), N = add(p2, p1);
        V3 B = cross(T, N);
        N = norm(cross(B, T)); B = norm(B);
        for (int j = 0; j <= radial; j++) {
            float v = (float)j / radial * 2.0f * PI;
            V3 pos = add(p1, add(mul(N, -tube * cosf(v)), mul(B, tube * sinf(v))));
            mesh_vert(&m, pos, norm(sub(pos, p1)), (float)i / tubular, (float)j / radial);
        }
    }
    for (int i = 0; i < tubular; i++)
        for (int j = 0; j < radial; j++) {
            uint32_t a = i * (radial + 1) + j, b = (i + 1) * (radial + 1) + j;
            mesh_tri(&m, a, b, a + 1);
            mesh_tri(&m, b, b + 1, a + 1);
        }
    fix_winding(&m);
    return m;
}

/* Minimal Wavefront OBJ loader: v / vt / vn / f (polygons are fan-triangulated). */
static void obj_corner(const char *tok, int *vi, int *ti, int *ni) {
    char *p;
    *vi = (int)strtol(tok, &p, 10); *ti = 0; *ni = 0;
    if (*p == '/') {
        p++;
        if (*p != '/') *ti = (int)strtol(p, &p, 10);
        if (*p == '/') { p++; *ni = (int)strtol(p, &p, 10); }
    }
}
static int load_obj(const char *path, Mesh *out) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    V3 *vs = NULL, *ns = NULL; float *ts = NULL;
    int nvs = 0, nns = 0, nts = 0, cvs = 0, cns = 0, cts = 0;
    Mesh m; memset(&m, 0, sizeof m);
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "v ", 2)) {
            V3 p; if (sscanf(line + 2, "%f %f %f", &p.x, &p.y, &p.z) == 3) {
                if (nvs == cvs) { cvs = cvs ? cvs * 2 : 1024; vs = realloc(vs, cvs * sizeof(V3)); }
                vs[nvs++] = p;
            }
        } else if (!strncmp(line, "vn ", 3)) {
            V3 p; if (sscanf(line + 3, "%f %f %f", &p.x, &p.y, &p.z) == 3) {
                if (nns == cns) { cns = cns ? cns * 2 : 1024; ns = realloc(ns, cns * sizeof(V3)); }
                ns[nns++] = p;
            }
        } else if (!strncmp(line, "vt ", 3)) {
            float u, v; if (sscanf(line + 3, "%f %f", &u, &v) == 2) {
                if (nts + 2 > cts) { cts = cts ? cts * 2 : 2048; ts = realloc(ts, cts * sizeof(float)); }
                ts[nts++] = u; ts[nts++] = v;
            }
        } else if (!strncmp(line, "f ", 2)) {
            uint32_t first = 0, prev = 0; int count = 0;
            for (char *tok = strtok(line + 2, " \t\r\n"); tok; tok = strtok(NULL, " \t\r\n")) {
                int vi, ti, ni; obj_corner(tok, &vi, &ti, &ni);
                if (vi < 0) vi += nvs + 1;
                if (ti < 0) ti += nts / 2 + 1;
                if (ni < 0) ni += nns + 1;
                if (vi < 1 || vi > nvs) continue;
                V3 n = (ni >= 1 && ni <= nns) ? ns[ni - 1] : v3(0, 0, 0);
                float u = 0, v = 0;
                if (ti >= 1 && ti <= nts / 2) { u = ts[(ti - 1) * 2]; v = ts[(ti - 1) * 2 + 1]; }
                uint32_t idx = (uint32_t)m.nv;
                mesh_vert(&m, vs[vi - 1], n, u, v);
                if (count == 0) first = idx;
                if (count >= 2) mesh_tri(&m, first, prev, idx);
                prev = idx; count++;
            }
        }
    }
    fclose(f);
    free(vs); free(ns); free(ts);
    if (m.ni == 0) return 0;
    /* faces without normals get flat normals from their winding (assumed CCW = outward) */
    for (int t = 0; t < m.ni; t += 3) {
        Vertex *a = &m.v[m.i[t]], *b = &m.v[m.i[t + 1]], *c = &m.v[m.i[t + 2]];
        if (len(a->n) < 1e-6f) {
            V3 fn = norm(cross(sub(b->p, a->p), sub(c->p, a->p)));
            a->n = b->n = c->n = fn;
        }
    }
    /* centre it and scale it to a unit sphere */
    V3 lo = m.v[0].p, hi = m.v[0].p;
    for (int i = 1; i < m.nv; i++) {
        V3 p = m.v[i].p;
        lo = v3(fminf(lo.x, p.x), fminf(lo.y, p.y), fminf(lo.z, p.z));
        hi = v3(fmaxf(hi.x, p.x), fmaxf(hi.y, p.y), fmaxf(hi.z, p.z));
    }
    V3 c = mul(add(lo, hi), 0.5f);
    float r = 0;
    for (int i = 0; i < m.nv; i++) r = fmaxf(r, len(sub(m.v[i].p, c)));
    for (int i = 0; i < m.nv; i++) m.v[i].p = mul(sub(m.v[i].p, c), 1.0f / r);
    fix_winding(&m);
    *out = m;
    return 1;
}

/* -------------------------------------------------- procedural textures */
static float hash3(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 0xffffff) / 16777216.0f;
}
static float vnoise(V3 p) {
    int xi = (int)floorf(p.x), yi = (int)floorf(p.y), zi = (int)floorf(p.z);
    float fx = p.x - xi, fy = p.y - yi, fz = p.z - zi;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    float c[2][2][2];
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int d = 0; d < 2; d++)
        c[a][b][d] = hash3(xi + a, yi + b, zi + d);
    float x00 = mixf(c[0][0][0], c[1][0][0], fx), x10 = mixf(c[0][1][0], c[1][1][0], fx);
    float x01 = mixf(c[0][0][1], c[1][0][1], fx), x11 = mixf(c[0][1][1], c[1][1][1], fx);
    return mixf(mixf(x00, x10, fy), mixf(x01, x11, fy), fz);
}
static float fbm(V3 p) {
    float a = 0.5f, s = 0;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = mul(p, 2.03f); a *= 0.5f; }
    return s;
}

enum { TEX_NONE, TEX_CHECKER, TEX_MARBLE, TEX_WOOD, TEX_PALETTE };
typedef struct {
    V3 color, color2;
    int tex; float tex_scale;
    float spec, shin;   /* Blinn-Phong strength and exponent */
    float refl;         /* Fresnel F0 for sky reflections */
    float metal;        /* 0 = dielectric, 1 = metal (tinted reflection, no diffuse) */
} Material;

static V3 texture_color(const Material *m, float u, float v, V3 wp, float dist) {
    switch (m->tex) {
    case TEX_CHECKER: {
        int iu = (int)floorf(u * m->tex_scale), iv = (int)floorf(v * m->tex_scale);
        float c = ((iu + iv) & 1) ? 1.0f : 0.0f;
        c = mixf(c, 0.5f, sat((dist - 14.0f) / 30.0f)); /* fade to the mean colour: no moiré */
        return mix3(m->color, m->color2, c);
    }
    case TEX_MARBLE: {
        float t = 0.5f + 0.5f * sinf(wp.x * 2.2f + 5.0f * fbm(mul(wp, 1.3f)));
        return mix3(m->color2, m->color, smooth01(0.05f, 0.85f, t));
    }
    case TEX_WOOD: {
        float g = fbm(v3(u * 4.0f, v * 3.0f, 0.0f));
        float t = 0.5f + 0.5f * sinf(v * 42.0f + 7.0f * g);
        return mix3(m->color, m->color2, t * t);
    }
    case TEX_PALETTE: {
        float t = u * 2.0f; /* integer number of cycles: seamless around the knot */
        return v3(0.55f + 0.45f * cosf(2 * PI * (t + 0.00f)),
                  0.55f + 0.45f * cosf(2 * PI * (t + 0.33f)),
                  0.55f + 0.45f * cosf(2 * PI * (t + 0.67f)));
    }
    default: return v3(1, 1, 1);
    }
}

/* ------------------------------------------------------ scene globals */
static V3 L_DIR;               /* unit vector pointing toward the sun */
static const V3 SUN_COL = {3.0f, 2.65f, 2.2f};
static const V3 SKY_HORIZON = {0.62f, 0.72f, 0.88f};
static const V3 SKY_ZENITH = {0.12f, 0.30f, 0.70f};
static const V3 AMB_SKY = {0.20f, 0.27f, 0.38f};
static const V3 AMB_GROUND = {0.14f, 0.12f, 0.10f};
static float *shadow_map;
static M4 light_vp;

static V3 sky(V3 d) {
    V3 c;
    if (d.y >= 0) c = mix3(SKY_HORIZON, SKY_ZENITH, sqrtf(sat(d.y)));
    else c = mix3(SKY_HORIZON, v3(0.16f, 0.15f, 0.14f), sat(-d.y * 3.0f));
    float s = sat(dot(d, L_DIR));
    c = add(c, mul(v3(1.0f, 0.85f, 0.6f), powf(s, 8.0f) * 0.25f + powf(s, 600.0f) * 8.0f));
    return c;
}

static float shadow_at(V3 wp, V3 n) {
    V3 p = add(wp, mul(n, 0.05f)); /* normal offset fights shadow acne */
    V4 lc = m4_mulp(light_vp, p);
    float sx = (lc.x * 0.5f + 0.5f) * SM_RES, sy = (0.5f - lc.y * 0.5f) * SM_RES;
    float z = lc.z - 0.0012f;
    if (sx < 3 || sy < 3 || sx >= SM_RES - 4 || sy >= SM_RES - 4 || z > 1.0f) return 1.0f;
    int cx = (int)sx, cy = (int)sy, lit = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            lit += z <= shadow_map[(cy + dy) * SM_RES + cx + dx];
    return lit / 25.0f;
}

/* ------------------------------------------------------- rasteriser */
typedef struct { V4 c; float v[NV]; } CV;           /* clip-space vertex + varyings */
typedef struct { int w, h; float *depth, *color; } Target;
typedef void (*FragFn)(void *ctx, int x, int y, float z, const float *var);

static CV cv_lerp(const CV *a, const CV *b, float t) {
    CV r;
    r.c.x = mixf(a->c.x, b->c.x, t); r.c.y = mixf(a->c.y, b->c.y, t);
    r.c.z = mixf(a->c.z, b->c.z, t); r.c.w = mixf(a->c.w, b->c.w, t);
    for (int k = 0; k < NV; k++) r.v[k] = mixf(a->v[k], b->v[k], t);
    return r;
}
/* Clip a triangle against the near plane (clip z >= 0). Returns 0, 3 or 4 vertices. */
static int clip_near(const CV *in, CV *out) {
    int n = 0;
    for (int i = 0; i < 3; i++) {
        const CV *a = &in[i], *b = &in[(i + 1) % 3];
        int ina = a->c.z >= 0, inb = b->c.z >= 0;
        if (ina) out[n++] = *a;
        if (ina != inb) out[n++] = cv_lerp(a, b, a->c.z / (a->c.z - b->c.z));
    }
    return n;
}

static void raster_tri(Target *t, const CV *a, const CV *b, const CV *c, FragFn fn, void *ctx) {
    const CV *v[3] = {a, b, c};
    float sx[3], sy[3], sz[3], iw[3], pv[3][NV];
    for (int i = 0; i < 3; i++) {
        iw[i] = 1.0f / v[i]->c.w;
        sx[i] = (v[i]->c.x * iw[i] * 0.5f + 0.5f) * t->w;
        sy[i] = (0.5f - v[i]->c.y * iw[i] * 0.5f) * t->h;
        sz[i] = v[i]->c.z * iw[i];
        for (int k = 0; k < NV; k++) pv[i][k] = v[i]->v[k] * iw[i]; /* attribute / w */
    }
    float area = (sx[2] - sx[0]) * (sy[1] - sy[0]) - (sy[2] - sy[0]) * (sx[1] - sx[0]);
    if (fabsf(area) < 1e-9f) return;
    float inv = 1.0f / area;
    int x0 = (int)fmaxf(0.0f, floorf(fminf(sx[0], fminf(sx[1], sx[2]))));
    int x1 = (int)fminf((float)t->w - 1, ceilf(fmaxf(sx[0], fmaxf(sx[1], sx[2]))));
    int y0 = (int)fmaxf(0.0f, floorf(fminf(sy[0], fminf(sy[1], sy[2]))));
    int y1 = (int)fminf((float)t->h - 1, ceilf(fmaxf(sy[0], fmaxf(sy[1], sy[2]))));
    float var[NV];
    for (int y = y0; y <= y1; y++) {
        float py = y + 0.5f;
        for (int x = x0; x <= x1; x++) {
            float px = x + 0.5f;
            /* edge functions = barycentric coordinates in screen space */
            float w0 = ((px - sx[1]) * (sy[2] - sy[1]) - (py - sy[1]) * (sx[2] - sx[1])) * inv;
            float w1 = ((px - sx[2]) * (sy[0] - sy[2]) - (py - sy[2]) * (sx[0] - sx[2])) * inv;
            float w2 = ((px - sx[0]) * (sy[1] - sy[0]) - (py - sy[0]) * (sx[1] - sx[0])) * inv;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            float z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
            if (z < 0 || z > 1) continue;
            size_t di = (size_t)y * t->w + x;
            if (z >= t->depth[di]) continue;
            t->depth[di] = z;
            if (!fn) continue;
            float iwp = w0 * iw[0] + w1 * iw[1] + w2 * iw[2], rw = 1.0f / iwp;
            for (int k = 0; k < NV; k++)
                var[k] = (w0 * pv[0][k] + w1 * pv[1][k] + w2 * pv[2][k]) * rw; /* perspective-correct */
            fn(ctx, x, y, z, var);
        }
    }
}

static void draw_mesh(Target *t, const Mesh *m, M4 model, M4 vp, V3 eye, int cull, FragFn fn, void *ctx) {
    CV *cv = malloc((size_t)m->nv * sizeof(CV));
    for (int i = 0; i < m->nv; i++) {
        V3 wp = m4_point(model, m->v[i].p);
        V3 wn = norm(m4_dir(model, m->v[i].n)); /* fine for rotation + uniform scale */
        cv[i].c = m4_mulp(vp, wp);
        cv[i].v[0] = wp.x; cv[i].v[1] = wp.y; cv[i].v[2] = wp.z;
        cv[i].v[3] = wn.x; cv[i].v[4] = wn.y; cv[i].v[5] = wn.z;
        cv[i].v[6] = m->v[i].u; cv[i].v[7] = m->v[i].v;
    }
    for (int k = 0; k < m->ni; k += 3) {
        CV tri[3] = {cv[m->i[k]], cv[m->i[k + 1]], cv[m->i[k + 2]]};
        if (cull) {
            V3 p0 = v3(tri[0].v[0], tri[0].v[1], tri[0].v[2]);
            V3 p1 = v3(tri[1].v[0], tri[1].v[1], tri[1].v[2]);
            V3 p2 = v3(tri[2].v[0], tri[2].v[1], tri[2].v[2]);
            if (dot(cross(sub(p1, p0), sub(p2, p0)), sub(eye, p0)) <= 0) continue;
        }
        CV poly[4];
        int n = clip_near(tri, poly);
        for (int j = 1; j + 1 < n; j++) raster_tri(t, &poly[0], &poly[j], &poly[j + 1], fn, ctx);
    }
    free(cv);
}

/* ---------------------------------------------------------- shading */
typedef struct { Target *t; const Material *mat; V3 eye; } MainCtx;

static void frag_main(void *vctx, int x, int y, float z, const float *var) {
    (void)z;
    MainCtx *c = vctx;
    const Material *m = c->mat;
    V3 wp = v3(var[0], var[1], var[2]);
    V3 n = norm(v3(var[3], var[4], var[5]));
    V3 view = sub(c->eye, wp);
    float dist = len(view);
    view = mul(view, 1.0f / dist);

    V3 albedo = vmul(texture_color(m, var[6], var[7], wp, dist), v3(1, 1, 1));
    if (m->tex == TEX_NONE) albedo = m->color;

    float sh = shadow_at(wp, n);
    float ndl = sat(dot(n, L_DIR));
    V3 H = norm(add(L_DIR, view));
    float spec = powf(sat(dot(n, H)), m->shin) * m->spec * (ndl > 0 ? 1.0f : 0.0f);

    V3 sun = mul(SUN_COL, sh);
    V3 hemi = mix3(AMB_GROUND, AMB_SKY, n.y * 0.5f + 0.5f);
    V3 diffuse = vmul(mul(albedo, 1.0f - m->metal), add(mul(sun, ndl), hemi));
    V3 col = add(diffuse, mul(sun, spec));

    /* Fresnel-weighted reflection of the sky */
    float ndv = sat(dot(n, view));
    float fres = m->refl + (1.0f - m->refl) * powf(1.0f - ndv, 5.0f);
    V3 R = sub(mul(n, 2.0f * dot(n, view)), view);
    V3 tint = mix3(v3(1, 1, 1), albedo, m->metal);
    col = add(mul(col, 1.0f - fres), mul(vmul(sky(R), tint), fres * (0.35f + 0.65f * sat(0.4f + sh))));

    float fog = 1.0f - expf(-dist * 0.0055f);
    col = mix3(col, SKY_HORIZON, fog);

    float *px = c->t->color + ((size_t)y * c->t->w + x) * 3;
    px[0] = col.x; px[1] = col.y; px[2] = col.z;
}

/* ------------------------------------------------------------ PNG out */
static uint32_t crc_tab[256];
static void crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crc_tab[n] = c;
    }
}
static uint32_t crc_update(uint32_t c, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ b[i]) & 0xff] ^ (c >> 8);
    return c;
}
static void put32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void write_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t hdr[8];
    put32(hdr, n); memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    uint32_t c = crc_update(0xffffffffu, (const uint8_t *)type, 4);
    if (n) c = crc_update(c, data, n);
    uint8_t tail[4]; put32(tail, ~c);
    fwrite(tail, 1, 4, f);
}
static int write_png(const char *path, int w, int h, const uint8_t *rgb) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    crc_init();
    size_t rowb = (size_t)w * 3 + 1, raw_n = rowb * h;
    uint8_t *raw = malloc(raw_n);
    for (int y = 0; y < h; y++) {
        raw[y * rowb] = 0; /* filter: none */
        memcpy(raw + y * rowb + 1, rgb + (size_t)y * w * 3, (size_t)w * 3);
    }
    size_t nblocks = raw_n / 65535 + 1;
    size_t z_n = 2 + raw_n + nblocks * 5 + 4;
    uint8_t *z = malloc(z_n), *zp = z;
    *zp++ = 0x78; *zp++ = 0x01;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw_n; i++) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
    for (size_t off = 0; off < raw_n || off == 0;) {
        size_t n = raw_n - off > 65535 ? 65535 : raw_n - off;
        *zp++ = (off + n >= raw_n) ? 1 : 0;
        *zp++ = n & 0xff; *zp++ = n >> 8; *zp++ = ~n & 0xff; *zp++ = (~n >> 8) & 0xff;
        memcpy(zp, raw + off, n); zp += n; off += n;
        if (n == 0) break;
    }
    put32(zp, (b << 16) | a); zp += 4;
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13]; put32(ihdr, w); put32(ihdr + 4, h);
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    write_chunk(f, "IHDR", ihdr, 13);
    write_chunk(f, "IDAT", z, (uint32_t)(zp - z));
    write_chunk(f, "IEND", NULL, 0);
    fclose(f); free(raw); free(z);
    return 1;
}

/* ------------------------------------------------------------ scene */
typedef struct { const Mesh *mesh; M4 model; Material mat; int cast_shadow, cull; } Obj;

static Mesh g_sphere, g_cube, g_floor, g_hero;
static int g_hero_is_obj;

static float aces(float x) {
    x *= 0.95f;
    return sat((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

static void render_frame(float t, int W, int H, int SS, uint8_t *out) {
    int RW = W * SS, RH = H * SS;
    size_t npix = (size_t)RW * RH;
    float *color = calloc(npix * 3, sizeof(float));
    float *depth = malloc(npix * sizeof(float));
    for (size_t i = 0; i < npix; i++) depth[i] = 2.0f;

    /* ---- build the scene for this moment in time ---- */
    Obj objs[32]; int no = 0;
    Material floor_m = {{0.78f, 0.75f, 0.70f}, {0.07f, 0.08f, 0.11f}, TEX_CHECKER, 40.0f, 0.15f, 60.0f, 0.04f, 0.0f};
    objs[no++] = (Obj){&g_floor, m4_identity(), floor_m, 0, 0};

    Material hero_m = {{0.9f, 0.9f, 0.9f}, {0, 0, 0}, TEX_PALETTE, 1.0f, 1.6f, 120.0f, 0.5f, 0.9f};
    if (g_hero_is_obj) { hero_m.tex = TEX_NONE; hero_m.color = v3(1.0f, 0.72f, 0.35f); }
    M4 hero = m4_mul(m4_translate(v3(0, 1.85f, 0)), m4_mul(m4_roty(t), m4_mul(m4_rotx(1.15f), m4_scale(g_hero_is_obj ? 1.6f : 0.95f))));
    objs[no++] = (Obj){&g_hero, hero, hero_m, 1, 1};

    Material marble = {{0.93f, 0.93f, 0.95f}, {0.10f, 0.22f, 0.28f}, TEX_MARBLE, 1.0f, 0.9f, 120.0f, 0.05f, 0.0f};
    objs[no++] = (Obj){&g_sphere, m4_mul(m4_translate(v3(3.1f, 0.95f, 1.2f)), m4_scale(0.95f)), marble, 1, 1};

    Material wood = {{0.62f, 0.38f, 0.18f}, {0.28f, 0.14f, 0.06f}, TEX_WOOD, 1.0f, 0.25f, 30.0f, 0.04f, 0.0f};
    objs[no++] = (Obj){&g_cube, m4_mul(m4_translate(v3(-3.0f, 0.75f, -0.4f)), m4_mul(m4_roty(t + 0.6f), m4_scale(0.75f))), wood, 1, 1};

    for (int i = 0; i < 7; i++) {
        float ang = (float)i / 7.0f * 2.0f * PI + t;
        float hue = (float)i / 7.0f;
        Material bead = {{0.55f + 0.45f * cosf(2 * PI * hue), 0.55f + 0.45f * cosf(2 * PI * (hue + 0.33f)),
                          0.55f + 0.45f * cosf(2 * PI * (hue + 0.67f))},
                         {0, 0, 0}, TEX_NONE, 1.0f, 1.0f, 140.0f, 0.12f, 0.0f};
        objs[no++] = (Obj){&g_sphere, m4_mul(m4_translate(v3(5.6f * cosf(ang), 0.38f, 5.6f * sinf(ang))), m4_scale(0.38f)), bead, 1, 1};
    }

    /* ---- shadow pass: depth-only render from the sun ---- */
    V3 centre = v3(0, 1.0f, 0);
    V3 light_pos = add(centre, mul(L_DIR, 15.0f));
    light_vp = m4_mul(m4_ortho(SM_RADIUS, 1.0f, 30.0f), m4_lookat(light_pos, centre, v3(0, 1, 0)));
    for (int i = 0; i < SM_RES * SM_RES; i++) shadow_map[i] = 2.0f;
    Target st = {SM_RES, SM_RES, shadow_map, NULL};
    for (int i = 0; i < no; i++)
        if (objs[i].cast_shadow) draw_mesh(&st, objs[i].mesh, objs[i].model, light_vp, light_pos, 0, NULL, NULL);

    /* ---- main pass ---- */
    float angle = 0.6f + t;
    V3 eye = v3(8.0f * sinf(angle), 2.9f, 8.0f * cosf(angle));
    V3 target = v3(0, 1.2f, 0);
    float fov = 0.68f, aspect = (float)W / H;
    M4 vp = m4_mul(m4_perspective(fov, aspect, 0.1f, 80.0f), m4_lookat(eye, target, v3(0, 1, 0)));
    Target mt = {RW, RH, depth, color};
    for (int i = 0; i < no; i++) {
        MainCtx ctx = {&mt, &objs[i].mat, eye};
        draw_mesh(&mt, objs[i].mesh, objs[i].model, vp, eye, objs[i].cull, frag_main, &ctx);
    }

    /* ---- sky behind everything ---- */
    V3 f = norm(sub(target, eye)), r = norm(cross(v3(0, 1, 0), f)), u = cross(f, r);
    float th = tanf(fov * 0.5f);
    for (int y = 0; y < RH; y++)
        for (int x = 0; x < RW; x++) {
            size_t i = (size_t)y * RW + x;
            if (depth[i] <= 1.5f) continue;
            float nx = ((x + 0.5f) / RW) * 2.0f - 1.0f, ny = 1.0f - ((y + 0.5f) / RH) * 2.0f;
            V3 d = norm(add(f, add(mul(r, nx * aspect * th), mul(u, ny * th))));
            V3 c = sky(d);
            color[i * 3] = c.x; color[i * 3 + 1] = c.y; color[i * 3 + 2] = c.z;
        }

    /* ---- tone map, gamma, supersample down, vignette ---- */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float acc[3] = {0, 0, 0};
            for (int sy = 0; sy < SS; sy++)
                for (int sx = 0; sx < SS; sx++) {
                    size_t i = ((size_t)(y * SS + sy) * RW + (x * SS + sx)) * 3;
                    for (int k = 0; k < 3; k++) acc[k] += powf(aces(color[i + k]), 1.0f / 2.2f);
                }
            float nx = ((x + 0.5f) / W) * 2.0f - 1.0f, ny = ((y + 0.5f) / H) * 2.0f - 1.0f;
            float vig = 1.0f - 0.22f * (nx * nx + ny * ny) * 0.5f;
            for (int k = 0; k < 3; k++)
                out[((size_t)y * W + x) * 3 + k] = (uint8_t)(sat(acc[k] / (SS * SS) * vig) * 255.0f + 0.5f);
        }
    free(color); free(depth);
}

int main(int argc, char **argv) {
    int W = 1280, H = 720, SS = 2, frames = 0;
    const char *out = NULL, *obj_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) SS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--obj") && i + 1 < argc) obj_path = argv[++i];
        else { fprintf(stderr, "usage: %s [-w N] [-h N] [-s ssaa] [--frames N] [-o name] [--obj file.obj]\n", argv[0]); return 1; }
    }
    if (W < 16 || H < 16 || SS < 1 || SS > 6) { fprintf(stderr, "bad size\n"); return 1; }

    L_DIR = norm(v3(0.55f, 0.85f, 0.38f));
    g_sphere = make_sphere(48, 96);
    g_cube = make_cube();
    g_floor = make_floor(60.0f);
    if (obj_path && load_obj(obj_path, &g_hero)) { g_hero_is_obj = 1; printf("loaded %s: %d triangles\n", obj_path, g_hero.ni / 3); }
    else {
        if (obj_path) fprintf(stderr, "could not read %s, using the torus knot\n", obj_path);
        g_hero = make_knot(480, 28, 2.0f, 3.0f, 1.0f, 0.3f);
    }
    shadow_map = malloc((size_t)SM_RES * SM_RES * sizeof(float));
    uint8_t *img = malloc((size_t)W * H * 3);

    int n = frames > 0 ? frames : 1;
    for (int i = 0; i < n; i++) {
        float t = frames > 0 ? 2.0f * PI * i / frames : 0.0f;
        char name[512];
        if (frames > 0) snprintf(name, sizeof name, "%s_%03d.png", out ? out : "frame", i);
        else snprintf(name, sizeof name, "%s", out ? out : "render.png");
        clock_t t0 = clock();
        render_frame(t, W, H, SS, img);
        if (!write_png(name, W, H, img)) { fprintf(stderr, "cannot write %s\n", name); return 1; }
        printf("%s  %dx%d  %dx SSAA  %.2fs\n", name, W, H, SS, (double)(clock() - t0) / CLOCKS_PER_SEC);
    }
    return 0;
}
