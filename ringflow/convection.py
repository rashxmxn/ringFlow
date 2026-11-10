"""Natural (free) convection of a viscous fluid in a ring-shaped domain
--  port of the three ``Dvizh_psi_om_Kolc*.for`` programs.

Physics
-------
2-D incompressible flow in stream-function / vorticity form, Boussinesq
buoyancy, written in general curvilinear coordinates (xi, eta) on a
structured O-grid (i periodic around the ring, j from wall to wall):

    dw/dt + (U.grad) w = nu * Lap(w) + beta * dT/dx      (vorticity)
    Lap(psi) = w                                          (stream function)
    dT/dt + (U.grad) T = nu * Lap(T)                      (temperature)
    u = dpsi/dy,  v = -dpsi/dx                            (velocity)

Symbols in the Fortran code:  w = vorticity, p = psi (stream function),
tt = temperature, tmx = nu (diffusion coefficient), bt = beta (buoyancy
number), t = time step, jakob = 1/J, a11/a12/a22 = metric tensor terms,
a1/a2 = contravariant velocities.

Time step (one "n" iteration), exactly as in the original:

1. stage 1 -- explicit update of the vorticity (upwind convection,
   central diffusion incl. the mixed derivative, buoyancy source);
2. stage 2 -- solve the Poisson equation for psi
      * Gauss-Seidel iterations (``..._jav.for``  -> solver "gs"), or
      * ADI pseudo-time iterations (``Dvizh_psi_om_Kolc.for`` -> "adi");
   then velocities from psi; then wall vorticity (Thom-like condition);
3. stage 3 -- explicit update of the temperature;
4. stop when max|T_new - T_old| (or |w_new - w_old|) < eps.

The three original programs are available as presets:

=========  ======================================  =========================
preset     original file                           what it is
=========  ======================================  =========================
ring       Dvizh_zhid/Dvizh_psi_om_Kolc_jav.for    circular annulus 101x51,
                                                   heated inner wall, GS
ring_adi   Dvizh_zhid/Dvizh_psi_om_Kolc.for        annulus 101x101, ADI psi
                                                   solver (test version: 3
                                                   steps, no wall vorticity)
curved     Dvizh_zhid/Dv_kv_set/..._jav.for        curvilinear ring 51x26,
                                                   grid from xxx/yyy.dat
                                                   (output of mesh.py)
=========  ======================================  =========================
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
import time

import numpy as np

from ._jit import njit
from .tridiag import cyclic_sweep, dirichlet_sweep
from . import io

PI_F77 = 3.141592655  # the value of pi used in the Fortran code


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class FlowConfig:
    preset: str = "ring"
    n1: int = 101
    n2: int = 51
    geometry: str = "ring"          # "ring" (analytic annulus) | "file" (xxx/yyy.dat)
    mesh_x: str = "xxx.dat"
    mesh_y: str = "yyy.dat"
    tau: float = 1e-3               # t      : time step
    beta: float = 10.0              # bt     : buoyancy number
    viscosity: float = 1e-3         # tmx    : diffusion coefficient
    eps: float = 1e-6               # stop criterion for the outer loop
    eps_psi: float = 1e-6           # epsp   : stop criterion for psi iterations
    max_steps: int = 20000          # m
    max_psi_iter: int = 1000        # mp
    psi_solver: str = "gs"          # "gs" | "adi"
    gs_variant: str = "ring"        # boundary rows handling, see _psi_gs
    stop_on: str = "T"              # "T" or "w"
    wall_vorticity: bool = True
    T_init: str = "1 - eta"         # see _initial_temperature
    psi_init: str = "0"             # "0" | "0.25*(1-eta)"
    snapshot_every: int = 5         # write a Tecplot zone every N steps (0 = off)
    write_initial: bool = True
    write_final_fields: bool = False
    write_metrics: bool = True      # metr_koef.dat
    fix_metric_stencil: bool = False
    print_every: int = 1


PRESETS = {
    "ring": FlowConfig(),
    "ring_adi": FlowConfig(
        preset="ring_adi", n2=101, beta=1e-4, eps=1e-8, eps_psi=1e-10,
        max_steps=3, max_psi_iter=3, psi_solver="adi", wall_vorticity=False,
        T_init="0.7 - 0.05*eta", psi_init="0.25*(1-eta)", snapshot_every=0,
        write_initial=False, write_final_fields=True, write_metrics=False),
    "curved": FlowConfig(
        preset="curved", n1=51, n2=26, geometry="file", max_psi_iter=10000,
        gs_variant="curved", stop_on="w", T_init="eta",
        # In the original the line "tmx(i,j)=0.001" is commented out, so tmx
        # is an uninitialised array (zero with Compaq Visual Fortran):
        # no viscosity / heat conduction at all.  Kept for fidelity; pass
        # --viscosity 0.001 to get what was probably intended.
        viscosity=0.0),
}


def preset(name, **overrides):
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return replace(PRESETS[name], **overrides)


# ===========================================================================
# Geometry and metric coefficients
# ===========================================================================

@dataclass
class Geometry:
    x: np.ndarray       # all 1-based (n1+1, n2+1)
    y: np.ndarray
    xx: np.ndarray      # dx/dxi
    xy: np.ndarray      # dx/deta
    yx: np.ndarray      # dy/dxi
    yy: np.ndarray      # dy/deta
    jakob: np.ndarray   # 1 / (x_xi*y_eta - x_eta*y_xi)
    a11: np.ndarray
    a22: np.ndarray
    a12: np.ndarray


def _pad(a):
    n1, n2 = a.shape
    out = np.zeros((n1 + 1, n2 + 1))
    out[1:, 1:] = a
    return out


def ring_geometry(n1, n2):
    """Annulus 0.5 <= r <= 2 with analytic metric terms (as in the original)."""
    h1 = 1.0 / (n1 - 1)
    h2 = 1.0 / (n2 - 1)
    i = np.arange(n1)[:, None]
    j = np.arange(n2)[None, :]
    r = 0.5 + j * 1.5 * h2
    phi = i * 2 * PI_F77 * h1
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    xx = -r * np.sin(phi)
    yx = r * np.cos(phi)
    xy = np.cos(phi) * np.ones_like(r)
    yy = np.sin(phi) * np.ones_like(r)
    return _finish_geometry(x, y, xx, xy, yx, yy)


def file_geometry(x, y, fix_metric_stencil=False):
    """Grid read from files, metric terms by finite differences (curved preset)."""
    from .mesh import _metrics
    x = np.array(x, float)
    y = np.array(y, float)
    x[-1, :] = x[0, :]
    y[-1, :] = y[0, :]
    n1, n2 = x.shape
    X, Y = _pad(x), _pad(y)
    xx = np.zeros_like(X); xy = np.zeros_like(X); yx = np.zeros_like(X); yy = np.zeros_like(X)
    _metrics(X, Y, n1, n2, 1.0 / (n1 - 1), 1.0 / (n2 - 1), xx, xy, yx, yy,
             0.5 if fix_metric_stencil else 1.0)
    return _finish_geometry(x, y, xx[1:, 1:], xy[1:, 1:], yx[1:, 1:], yy[1:, 1:])


def _finish_geometry(x, y, xx, xy, yx, yy):
    jak = 1.0 / (xx * yy - xy * yx)
    a11 = jak * (yy ** 2 + xy ** 2)
    a22 = jak * (yx ** 2 + xx ** 2)
    a12 = jak * (yx * yy + xx * xy)
    return Geometry(*(_pad(a) for a in (x, y, xx, xy, yx, yy, jak, a11, a22, a12)))


# ===========================================================================
# Kernels (1-based arrays; loops mirror the Fortran)
# ===========================================================================

@njit
def _transport(f, f1, a1, a2, a11, a22, a12, jak, tmx, yy, yx, tt,
               n1, n2, h1, h2, t, bt, buoyancy):
    """Stage 1 (vorticity, buoyancy=True) / stage 3 (temperature)."""
    n12 = n1 - 1
    for i in range(1, n12 + 1):
        im = n12 if i == 1 else i - 1
        ip = i + 1
        for j in range(2, n2):
            j1 = j + 1
            j2 = j - 1
            # convection, first-order upwind
            px = 0.5 * t / h1
            px1 = 0.5 * (a1[ip, j] + a1[i, j]); px1 = px1 - abs(px1)
            px2 = 0.5 * (a1[i, j] + a1[im, j]); px2 = px2 + abs(px2)
            py = 0.5 * t / h2
            py1 = 0.5 * (a2[i, j1] + a2[i, j]); py1 = py1 - abs(py1)
            py2 = 0.5 * (a2[i, j] + a2[i, j2]); py2 = py2 + abs(py2)
            val = f[i, j] - px * (px1 * (f[ip, j] - f[i, j]) + px2 * (f[i, j] - f[im, j])) \
                          - py * (py1 * (f[i, j1] - f[i, j]) + py2 * (f[i, j] - f[i, j2]))
            # diffusion, main terms
            px = t * tmx[i, j] * jak[i, j] / h1 ** 2
            px1 = 0.5 * (a11[ip, j] + a11[i, j])
            px2 = 0.5 * (a11[i, j] + a11[im, j])
            py = t * tmx[i, j] * jak[i, j] / h2 ** 2
            py1 = 0.5 * (a22[i, j1] + a22[i, j])
            py2 = 0.5 * (a22[i, j] + a22[i, j2])
            val = val + px * (px1 * (f[ip, j] - f[i, j]) - px2 * (f[i, j] - f[im, j])) \
                      + py * (py1 * (f[i, j1] - f[i, j]) - py2 * (f[i, j] - f[i, j2]))
            # diffusion, mixed derivative
            px = t * tmx[i, j] * jak[i, j] / (4 * h1 * h2)
            px1 = a12[ip, j]
            px2 = a12[im, j]
            py1 = a12[i, j1]
            py2 = a12[i, j2]
            val = val - px * (px1 * (f[ip, j1] - f[ip, j2]) - px2 * (f[im, j1] - f[im, j2])) \
                      - px * (py1 * (f[ip, j1] - f[im, j1]) - py2 * (f[ip, j2] - f[im, j2]))
            if buoyancy:
                px = t * bt * jak[i, j]
                px1 = 0.5 * yy[i, j] / h1
                px2 = 0.5 * yx[i, j] / h2
                val = val + px * (px1 * (tt[ip, j] - tt[im, j]) - px2 * (tt[i, j1] - tt[i, j2]))
            f1[i, j] = val
            if i == 1:
                f1[n1, j] = val


@njit
def _psi_gs(p1, w1, jak, a11, a22, a12, n1, n2, h1, h2, mp, epsp, variant):
    """Stage 2, Gauss-Seidel solver for psi.  Returns (k, horp).

    variant 0 ("ring"):   i=1 column for j=2..n2-2, rows j=1,2 := p1(1,2),
                          rows j=n2-1, n2 never updated (stay = initial value)
    variant 1 ("curved"): i=1 column for j=3..n2-1, rows j=n2-1..n2 := p1(1,n2-1),
                          rows j=1,2 never updated
    interior always i=2..n1-1, j=3..n2-2.
    """
    n12 = n1 - 1
    n22 = n2 - 1
    c4 = 4 * h1 * h2
    k = 0
    horp = 0.0
    while True:
        k += 1
        if k > mp:
            break
        p = p1.copy()
        jlo = 2 if variant == 0 else 3
        jhi = n22 - 1 if variant == 0 else n22
        for j in range(jlo, jhi + 1):
            j1 = j + 1; j2 = j - 1
            px1 = 0.5 * (a11[2, j] + a11[1, j]) / h1 ** 2
            px2 = 0.5 * (a11[1, j] + a11[n12, j]) / h1 ** 2
            py1 = 0.5 * (a22[1, j1] + a22[1, j]) / h2 ** 2
            py2 = 0.5 * (a22[1, j] + a22[1, j2]) / h2 ** 2
            d = px1 + px2 + py1 + py2
            f = w1[1, j] / jak[1, j]
            s = px1 * p[2, j] + px2 * p[n12, j] + py1 * p[1, j1] + py2 * p[1, j2]
            cx1 = a12[2, j] / c4; cx2 = a12[n12, j] / c4
            cy1 = a12[1, j1] / c4; cy2 = a12[1, j2] / c4
            s = s - (cx1 * (p[2, j1] - p[2, j2]) - cx2 * (p[n12, j1] - p[n12, j2])) \
                  - (cy1 * (p[2, j1] - p[n12, j1]) - cy2 * (p[2, j2] - p[n12, j2]))
            p1[1, j] = (s + f) / d
            p1[n1, j] = p1[1, j]
        if variant == 0:
            v = p1[1, 2]
            for i in range(1, n1 + 1):
                p1[i, 1] = v
                p1[i, 2] = v
        else:
            v = p1[1, n22]
            for i in range(1, n1 + 1):
                p1[i, n22] = v
                p1[i, n2] = v
        for i in range(2, n12 + 1):
            i1 = i + 1; i2 = i - 1
            for j in range(3, n22):
                j1 = j + 1; j2 = j - 1
                px1 = 0.5 * (a11[i1, j] + a11[i, j]) / h1 ** 2
                px2 = 0.5 * (a11[i, j] + a11[i2, j]) / h1 ** 2
                py1 = 0.5 * (a22[i, j1] + a22[i, j]) / h2 ** 2
                py2 = 0.5 * (a22[i, j] + a22[i, j2]) / h2 ** 2
                d = px1 + px2 + py1 + py2
                f = w1[i, j] / jak[i, j]
                s = px1 * p[i1, j] + px2 * p1[i2, j] + py1 * p[i, j1] + py2 * p1[i, j2]
                cx1 = a12[i1, j] / c4; cx2 = a12[i2, j] / c4
                cy1 = a12[i, j1] / c4; cy2 = a12[i, j2] / c4
                s = s - (cx1 * (p[i1, j1] - p[i1, j2]) - cx2 * (p[i2, j1] - p1[i2, j2])) \
                      - (cy1 * (p[i1, j1] - p[i2, j1]) - cy2 * (p[i1, j2] - p1[i2, j2]))
                p1[i, j] = (s + f) / d
        horp = 0.0
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                ps = abs(p1[i, j] - p[i, j])
                if ps > horp:
                    horp = ps
        if not horp > epsp:
            break
    return k, horp


@njit
def _psi_adi(p1, p2, w1, jak, a11, a22, a12, n1, n2, h1, h2, t, mp, epsp):
    """Stage 2, ADI pseudo-time solver for psi (Dvizh_psi_om_Kolc.for).

    p2 holds psi (in/out); p1 is the intermediate (half-step) array.
    """
    n12 = n1 - 1
    n22 = n2 - 1
    c4 = 4 * h1 * h2
    A = np.zeros(n1 + n2 + 2); B = np.zeros(n1 + n2 + 2); F = np.zeros(n1 + n2 + 2)
    out = np.zeros(n1 + n2 + 2)
    k = 0
    horp = 0.0
    while True:
        k += 1
        if k > mp:
            break
        p = p2.copy()
        # I: implicit along i (periodic), explicit mixed term
        for j in range(2, n22 + 1):
            j1 = j + 1; j2 = j - 1
            for i in range(1, n12 + 1):
                im = n12 if i == 1 else i - 1
                ip = i + 1
                A[i] = 0.5 * t * (a11[i, j] + a11[im, j]) / h1 ** 2
                B[i] = 0.5 * t * (a11[ip, j] + a11[i, j]) / h1 ** 2
                py1 = t * a12[ip, j] / c4
                py2 = t * a12[im, j] / c4
                F[i] = p[i, j] - (py1 * (p[ip, j1] - p[ip, j2]) - py2 * (p[im, j1] - p[im, j2]))
            cyclic_sweep(A, B, F, n1, out)
            for i in range(1, n1 + 1):
                p1[i, j] = out[i]
        # II: implicit along j
        for kk in range(1, n12 + 1):
            i = kk + 1 if kk < n12 else 1
            im = n12 if i == 1 else i - 1
            ip = i + 1
            for j in range(2, n22 + 1):
                j1 = j + 1; j2 = j - 1
                A[j] = 0.5 * t * (a22[i, j] + a22[i, j2]) / h2 ** 2
                B[j] = 0.5 * t * (a22[i, j1] + a22[i, j]) / h2 ** 2
                py1 = t * a12[i, j1] / c4
                py2 = t * a12[i, j2] / c4
                F[j] = p1[i, j] - (py1 * (p1[ip, j1] - p1[im, j1]) - py2 * (p1[ip, j2] - p1[im, j2]))
            dirichlet_sweep(A, B, F, n2, p[i, 1], p[i, n2], out)
            for j in range(1, n2 + 1):
                p2[i, j] = out[j]
        for j in range(1, n2 + 1):
            p2[n1, j] = p2[1, j]
        horp = 0.0
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                p2[i, j] = p2[i, j] + t * w1[i, j] / jak[i, j]
                ps = abs(p2[i, j] - p[i, j])
                if ps > horp:
                    horp = ps
        if not horp > epsp:
            break
    return k, horp


@njit
def _velocity(p, u, v, jak, xx, xy, yx, yy, n1, n2, h1, h2):
    n12 = n1 - 1
    for i in range(1, n12 + 1):
        im = n12 if i == 1 else i - 1
        ip = i + 1
        for j in range(2, n2):
            j1 = j + 1; j2 = j - 1
            px1 = 0.5 * jak[i, j] * xy[i, j] / h1
            px2 = 0.5 * jak[i, j] * xx[i, j] / h2
            u[i, j] = -px1 * (p[ip, j] - p[im, j]) + px2 * (p[i, j1] - p[i, j2])
            py1 = 0.5 * jak[i, j] * yy[i, j] / h1
            py2 = 0.5 * jak[i, j] * yx[i, j] / h2
            v[i, j] = -py1 * (p[ip, j] - p[im, j]) + py2 * (p[i, j1] - p[i, j2])
            if i == 1:
                u[n1, j] = u[1, j]
                v[n1, j] = v[1, j]


@njit
def _wall_vorticity(w1, u, v, jak, xx, yx, n1, n2, h2):
    n22 = n2 - 1
    for i in range(1, n1 + 1):
        px1 = jak[i, 1] * yx[i, 1] / h2
        py1 = jak[i, 1] * xx[i, 1] / h2
        w1[i, 1] = -(px1 * v[i, 2] + py1 * u[i, 2])
        px2 = jak[i, n2] * yx[i, n2] / h2
        py2 = jak[i, n2] * xx[i, n2] / h2
        w1[i, n2] = px2 * v[i, n22] + py2 * u[i, n22]


@njit
def _contravariant(u, v, jak, xx, xy, yx, yy, a1, a2):
    a1[:, :] = jak * (yy * u - xy * v)
    a2[:, :] = jak * (xx * v - yx * u)


# ===========================================================================
# Driver
# ===========================================================================

@dataclass
class FlowResult:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray       # vorticity
    psi: np.ndarray     # stream function
    T: np.ndarray       # temperature
    steps: int
    converged: bool
    history: np.ndarray  # columns: n, k, horw, hortt, horp
    config: FlowConfig = None
    geometry: Geometry = None


def _initial_fields(cfg, n1, n2):
    eta = (np.arange(n2)[None, :] / (n2 - 1)) * np.ones((n1, 1))  # (j-1)/n22
    T = {"1 - eta": 1.0 - eta,
         "eta": eta,
         "0.7 - 0.05*eta": 0.2 + 0.5 - 0.05 * eta}.get(cfg.T_init)
    if T is None:
        raise ValueError(f"unknown T_init {cfg.T_init!r}")
    psi = {"0": np.zeros((n1, n2)),
           "0.25*(1-eta)": 0.25 - 0.25 * eta}.get(cfg.psi_init)
    if psi is None:
        raise ValueError(f"unknown psi_init {cfg.psi_init!r}")
    return T, psi


def _fields_zone(fh, g, u, v, w, p, T, title=True):
    io.write_tecplot_zone(
        fh, ["X", "Y", "U", "V", "W", "P", "T"],
        [g.x[1:, 1:], g.y[1:, 1:], u[1:, 1:], v[1:, 1:], w[1:, 1:], p[1:, 1:], T[1:, 1:]],
        title=title, fmts=["%8.4f", "%8.4f"] + ["%15.7f"] * 5)


def solve(cfg=None, out_dir=None, verbose=True, geometry=None, callback=None):
    """Run the flow solver.

    cfg      : FlowConfig (default: preset "ring")
    out_dir  : where to write rezsurf.dat / metr_koef.dat / result.npz
               (None = write nothing)
    geometry : optional Geometry (otherwise built from cfg)
    callback : optional f(n, fields_dict) called after every step;
               returning True stops the run
    """
    cfg = cfg or preset("ring")
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    if geometry is None:
        if cfg.geometry == "ring":
            geometry = ring_geometry(cfg.n1, cfg.n2)
        else:
            gx = io.read_grid_array(cfg.mesh_x, cfg.n1, cfg.n2)
            gy = io.read_grid_array(cfg.mesh_y, cfg.n1, cfg.n2)
            geometry = file_geometry(gx, gy, cfg.fix_metric_stencil)
    g = geometry
    n1, n2 = g.x.shape[0] - 1, g.x.shape[1] - 1
    h1 = 1.0 / (n1 - 1)
    h2 = 1.0 / (n2 - 1)
    t, bt = cfg.tau, cfg.beta

    T0, psi0 = _initial_fields(cfg, n1, n2)
    tt1 = _pad(T0)
    Z = lambda: np.zeros((n1 + 1, n2 + 1))
    u, v, w, w1, tt, a1, a2 = Z(), Z(), Z(), Z(), Z(), Z(), Z()
    tmx = np.full((n1 + 1, n2 + 1), cfg.viscosity)
    if cfg.psi_solver == "adi":
        psi = _pad(psi0)          # p2 in the Fortran
        half = Z()                # p1 in the Fortran
    else:
        psi = _pad(psi0)          # p1 in the Fortran
        half = None
    variant = 0 if cfg.gs_variant == "ring" else 1

    if cfg.psi_solver == "gs":
        _velocity(psi, u, v, g.jakob, g.xx, g.xy, g.yx, g.yy, n1, n2, h1, h2)

    fh = None
    if out is not None and (cfg.write_initial or cfg.snapshot_every or cfg.write_final_fields):
        fh = open(out / "rezsurf.dat", "w")
    if fh is not None and cfg.write_initial:
        _fields_zone(fh, g, u, v, w1, psi, tt1)

    hist = []
    converged = False
    n = 0
    t0 = time.time()
    try:
        while True:
            n += 1
            if n > cfg.max_steps:
                n = cfg.max_steps
                break
            w[:, :] = w1
            tt[:, :] = tt1
            _contravariant(u, v, g.jakob, g.xx, g.xy, g.yx, g.yy, a1, a2)

            # stage 1: vorticity
            _transport(w, w1, a1, a2, g.a11, g.a22, g.a12, g.jakob, tmx, g.yy, g.yx, tt,
                       n1, n2, h1, h2, t, bt, True)
            # stage 2: stream function, velocity, wall vorticity
            if cfg.psi_solver == "gs":
                k, horp = _psi_gs(psi, w1, g.jakob, g.a11, g.a22, g.a12, n1, n2, h1, h2,
                                  cfg.max_psi_iter, cfg.eps_psi, variant)
            else:
                k, horp = _psi_adi(half, psi, w1, g.jakob, g.a11, g.a22, g.a12, n1, n2,
                                   h1, h2, t, cfg.max_psi_iter, cfg.eps_psi)
            _velocity(psi, u, v, g.jakob, g.xx, g.xy, g.yx, g.yy, n1, n2, h1, h2)
            if cfg.wall_vorticity:
                _wall_vorticity(w1, u, v, g.jakob, g.xx, g.yx, n1, n2, h2)
            # stage 3: temperature
            _transport(tt, tt1, a1, a2, g.a11, g.a22, g.a12, g.jakob, tmx, g.yy, g.yx, tt,
                       n1, n2, h1, h2, t, bt, False)

            horw = float(np.max(np.abs(w1 - w)))
            hortt = float(np.max(np.abs(tt1 - tt)))
            hist.append((n, k, horw, hortt, horp))

            if fh is not None and cfg.snapshot_every and n % cfg.snapshot_every == 0:
                _fields_zone(fh, g, u, v, w1, psi, tt1)
            if verbose and n % cfg.print_every == 0:
                print(f"{n:6d}{k:6d}{horw:15.10f}{hortt:15.10f}{horp:15.10f}")
            if callback is not None:
                # callback may return True to stop the calculation (GUI "Stop")
                if callback(n, dict(u=u[1:, 1:], v=v[1:, 1:], w=w1[1:, 1:],
                                    psi=psi[1:, 1:], T=tt1[1:, 1:],
                                    k=k, horw=horw, hortt=hortt, horp=horp)) is True:
                    break
            crit = hortt if cfg.stop_on == "T" else horw
            if not crit > cfg.eps:
                converged = True
                break
    finally:
        if fh is not None:
            if cfg.write_final_fields:
                _fields_zone(fh, g, u, v, w1, psi, tt1)
            fh.close()

    if verbose:
        print(f"finished after {n} steps ({'converged' if converged else 'step limit reached'}),"
              f" {time.time() - t0:.1f} s")

    res = FlowResult(g.x[1:, 1:].copy(), g.y[1:, 1:].copy(), u[1:, 1:].copy(),
                     v[1:, 1:].copy(), w1[1:, 1:].copy(), psi[1:, 1:].copy(),
                     tt1[1:, 1:].copy(), n, converged, np.array(hist), cfg, g)

    if out is not None:
        if cfg.write_metrics:
            with open(out / "metr_koef.dat", "w") as fm:
                io.write_tecplot_zone(
                    fm, ["X", "Y", "U", "V", "W", "P", "T"],
                    [g.x[1:, 1:], g.y[1:, 1:], g.xx[1:, 1:], g.xy[1:, 1:], g.yx[1:, 1:],
                     g.yy[1:, 1:], g.jakob[1:, 1:]],
                    fmts=["%8.4f", "%8.4f"] + ["%15.7f"] * 5)
        np.savez(out / "result.npz", x=res.x, y=res.y, u=res.u, v=res.v, w=res.w,
                 psi=res.psi, T=res.T, history=res.history)
    return res
