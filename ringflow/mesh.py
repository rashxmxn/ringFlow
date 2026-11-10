"""Elliptic mesh generator for a doubly-connected (ring-like) domain
--  port of ``Dvusv_set_dvish/Setka_ED2_II.for`` (+ subroutine KRIT_KA).

What it does
------------
Builds a structured, O-type curvilinear grid (n1 x n2 = 51 x 26 nodes) between
an outer and an inner closed contour (read from ``rez_param.dat``):

* i = 1..n1 runs around the ring (periodic: node n1 == node 1),
* j = 1 is the outer contour, j = n2 the inner contour.

Start: straight lines between matching boundary points.  Then the interior
nodes are moved by solving the (inverse) elliptic grid equations

    d/dxi (g22 dX/dxi) + d/deta (g11 dX/deta) = 0,   X = x, y

with a pseudo-time ADI scheme (implicit periodic sweep along i, then an
implicit sweep along j) until the grid stops moving (max node shift < eps)
or ``max_iter`` iterations are done.

After every iteration four grid-quality criteria are evaluated per cell
(KRIT_KA, all in [0, 1], 1 = perfect):

* Q1: 2*min(triangle areas)/cell area  -- convexity / skewness of the quad
* Q2: min sine of the four cell angles  -- orthogonality
* Q3: min edge / max edge               -- aspect ratio
* Q4: min(S/S_avg, S_avg/S)             -- area uniformity

For every criterion the program remembers the grid on which the worst cell
was best ("max-min"), and that is what is written out.  ``xxx.dat`` /
``yyy.dat`` hold the best grid according to Q1; they are the input of the
flow solver in the curved domain.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._jit import njit
from .tridiag import cyclic_sweep, dirichlet_sweep
from . import io


# ---------------------------------------------------------------------------
# kernels (1-based arrays, see tridiag.py)
# ---------------------------------------------------------------------------

@njit
def _metrics(x2, y2, n1, n2, h1, h2, xx, xy, yx, yy, interior_factor):
    n12 = n1 - 1
    n22 = n2 - 1
    for i in range(2, n12 + 1):
        for j in range(1, n2 + 1):
            # NOTE: the original has no 0.5 here (unlike the periodic node
            # below).  interior_factor=1.0 reproduces it, 0.5 fixes it.
            xx[i, j] = interior_factor * (x2[i + 1, j] - x2[i - 1, j]) / h1
            yx[i, j] = interior_factor * (y2[i + 1, j] - y2[i - 1, j]) / h1
    for j in range(1, n2 + 1):
        xx[1, j] = 0.5 * (x2[2, j] - x2[n12, j]) / h1
        xx[n1, j] = xx[1, j]
        yx[1, j] = 0.5 * (y2[2, j] - y2[n12, j]) / h1
        yx[n1, j] = yx[1, j]
    for i in range(1, n1 + 1):
        for j in range(2, n22 + 1):
            xy[i, j] = 0.5 * (x2[i, j + 1] - x2[i, j - 1]) / h2
            yy[i, j] = 0.5 * (y2[i, j + 1] - y2[i, j - 1]) / h2
    for i in range(1, n1 + 1):
        xy[i, 1] = (x2[i, 2] - x2[i, 1]) / h2
        xy[i, n2] = (x2[i, n2] - x2[i, n22]) / h2
        yy[i, 1] = (y2[i, 2] - y2[i, 1]) / h2
        yy[i, n2] = (y2[i, n2] - y2[i, n22]) / h2


@njit
def _coef(xx, xy, yx, yy, jak, i, im, ip, j, h1, h2):
    """px1, px2, py1, py2 of the original (face-averaged metric weights)."""
    a12 = (xy[i, j] + xy[im, j]) / 2
    a22 = (yy[i, j] + yy[im, j]) / 2
    jk = (jak[i, j] + jak[im, j]) / 2
    px1 = jk * (a12 ** 2 + a22 ** 2) / h1 ** 2

    a12 = (xy[ip, j] + xy[i, j]) / 2
    a22 = (yy[ip, j] + yy[i, j]) / 2
    jk = (jak[ip, j] + jak[i, j]) / 2
    px2 = jk * (a12 ** 2 + a22 ** 2) / h1 ** 2

    a11 = (xx[i, j] + xx[i, j - 1]) / 2
    a21 = (yx[i, j] + yx[i, j - 1]) / 2
    jk = (jak[i, j] + jak[i, j - 1]) / 2
    py1 = jk * (a11 ** 2 + a21 ** 2) / h2 ** 2

    a11 = (xx[i, j + 1] + xx[i, j]) / 2
    a21 = (yx[i, j + 1] + yx[i, j]) / 2
    jk = (jak[i, j + 1] + jak[i, j]) / 2
    py2 = jk * (a11 ** 2 + a21 ** 2) / h2 ** 2
    return px1, px2, py1, py2


@njit
def _drov1(wb, w1, xx, xy, yx, yy, jak, n1, n2, h1, h2, t):
    """Implicit periodic sweep along i for every interior line j."""
    n12 = n1 - 1
    A = np.zeros(n1 + 1)
    B = np.zeros(n1 + 1)
    F = np.zeros(n1 + 1)
    out = np.zeros(n1 + 1)
    for j in range(2, n2):
        for i in range(1, n12 + 1):
            im = n12 if i == 1 else i - 1
            ip = i + 1
            px1, px2, py1, py2 = _coef(xx, xy, yx, yy, jak, i, im, ip, j, h1, h2)
            A[i] = t * px1
            B[i] = t * px2
            F[i] = wb[i, j] + t * (py2 * (wb[i, j + 1] - wb[i, j])
                                   - py1 * (wb[i, j] - wb[i, j - 1]))
        cyclic_sweep(A, B, F, n1, out)
        for i in range(1, n1 + 1):
            w1[i, j] = out[i]


@njit
def _drov2(w, w1, xx, xy, yx, yy, jak, n1, n2, h1, h2, t):
    """Implicit sweep along j (Dirichlet ends) for every line i."""
    n12 = n1 - 1
    A = np.zeros(n2 + 1)
    B = np.zeros(n2 + 1)
    F = np.zeros(n2 + 1)
    out = np.zeros(n2 + 1)
    # Fortran order: i = 2..n12 first, then i = 1 (periodic neighbour)
    for k in range(1, n12 + 1):
        i = k + 1 if k < n12 else 1
        im = n12 if i == 1 else i - 1
        ip = i + 1
        for j in range(2, n2):
            px1, px2, py1, py2 = _coef(xx, xy, yx, yy, jak, i, im, ip, j, h1, h2)
            A[j] = t * py1
            B[j] = t * py2
            F[j] = w1[i, j] + t * (px2 * (w1[ip, j] - w1[i, j])
                                   - px1 * (w1[i, j] - w1[im, j]))
        dirichlet_sweep(A, B, F, n2, w1[i, 1], w1[i, n2], out)
        if i == 1:
            for l in range(1, n2):
                w[1, l] = out[l]
                w[n1, l] = out[l]
            w[1, n2] = out[n2]
        else:
            for l in range(1, n2 + 1):
                w[i, l] = out[l]


@njit
def mesh_iteration(x2, y2, n1, n2, h1, h2, t, jak, xx, xy, yx, yy, interior_factor):
    """One full iteration (x then y); updates x2, y2 in place; returns max shift."""
    _metrics(x2, y2, n1, n2, h1, h2, xx, xy, yx, yy, interior_factor)
    xb = x2.copy()
    yb = y2.copy()
    x1 = x2.copy()
    y1 = y2.copy()
    _drov1(xb, x1, xx, xy, yx, yy, jak, n1, n2, h1, h2, t)
    _drov2(x2, x1, xx, xy, yx, yy, jak, n1, n2, h1, h2, t)
    _drov1(yb, y1, xx, xy, yx, yy, jak, n1, n2, h1, h2, t)
    _drov2(y2, y1, xx, xy, yx, yy, jak, n1, n2, h1, h2, t)
    hor = 0.0
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            ps = np.sqrt((x2[i, j] - xb[i, j]) ** 2 + (y2[i, j] - yb[i, j]) ** 2)
            if ps > hor:
                hor = ps
    return hor


@njit
def quality(x2, y2, n1, n2):
    """KRIT_KA: four quality criteria per node (1-based (n1+1, n2+1) arrays)."""
    n12 = n1 - 1
    n22 = n2 - 1
    q1 = np.zeros((n1 + 1, n2 + 1))
    q2 = np.zeros((n1 + 1, n2 + 1))
    q3 = np.zeros((n1 + 1, n2 + 1))
    q4 = np.zeros((n1 + 1, n2 + 1))
    ss = 0.0
    for i in range(1, n12 + 1):
        for j in range(1, n22 + 1):
            xt1 = x2[i, j]; xt2 = x2[i + 1, j]; xt3 = x2[i + 1, j + 1]; xt4 = x2[i, j + 1]
            yt1 = y2[i, j]; yt2 = y2[i + 1, j]; yt3 = y2[i + 1, j + 1]; yt4 = y2[i, j + 1]
            s1 = 0.5 * ((xt2 - xt1) * (yt3 - yt1) - (xt3 - xt1) * (yt2 - yt1))
            s2 = 0.5 * ((xt3 - xt1) * (yt4 - yt1) - (xt4 - xt1) * (yt3 - yt1))
            ss += s1 + s2
    ss = ss / (n12 * n22)
    for i in range(1, n12 + 1):
        for j in range(1, n22 + 1):
            xt1 = x2[i, j]; xt2 = x2[i + 1, j]; xt3 = x2[i + 1, j + 1]; xt4 = x2[i, j + 1]
            yt1 = y2[i, j]; yt2 = y2[i + 1, j]; yt3 = y2[i + 1, j + 1]; yt4 = y2[i, j + 1]
            s1 = 0.5 * ((xt2 - xt1) * (yt3 - yt1) - (xt3 - xt1) * (yt2 - yt1))
            s2 = 0.5 * ((xt3 - xt1) * (yt4 - yt1) - (xt4 - xt1) * (yt3 - yt1))
            s3 = 0.5 * ((xt2 - xt1) * (yt4 - yt1) - (xt4 - xt1) * (yt2 - yt1))
            s4 = 0.5 * ((xt3 - xt2) * (yt4 - yt2) - (xt4 - xt2) * (yt3 - yt2))
            sabcd = s1 + s2
            pl1 = np.sqrt((xt2 - xt1) ** 2 + (yt2 - yt1) ** 2)
            pl2 = np.sqrt((xt3 - xt2) ** 2 + (yt3 - yt2) ** 2)
            pl3 = np.sqrt((xt4 - xt3) ** 2 + (yt4 - yt3) ** 2)
            pl4 = np.sqrt((xt1 - xt4) ** 2 + (yt1 - yt4) ** 2)
            sinf1 = 2 * s3 / (pl1 * pl4)
            sinf2 = 2 * s1 / (pl1 * pl2)
            sinf3 = 2 * s4 / (pl2 * pl3)
            sinf4 = 2 * s2 / (pl3 * pl4)
            q1[i, j] = 2 * min(min(s1, s2), min(s3, s4)) / (s1 + s2)
            q2[i, j] = min(min(sinf1, sinf2), min(sinf3, sinf4))
            q3[i, j] = min(min(pl1, pl2), min(pl3, pl4)) / max(max(pl1, pl2), max(pl3, pl4))
            q4[i, j] = min(sabcd / ss, ss / sabcd)
    for i in range(1, n1 + 1):
        q1[i, n2] = q1[i, n22]; q2[i, n2] = q2[i, n22]
        q3[i, n2] = q3[i, n22]; q4[i, n2] = q4[i, n22]
    for j in range(1, n2 + 1):
        q1[n1, j] = q1[1, j]; q2[n1, j] = q2[1, j]
        q3[n1, j] = q3[1, j]; q4[n1, j] = q4[1, j]
    return q1, q2, q3, q4


def grid_quality(x, y):
    """Public helper: quality criteria Q1..Q4 for a 0-based (n1, n2) grid."""
    n1, n2 = x.shape
    X = np.zeros((n1 + 1, n2 + 1)); X[1:, 1:] = x
    Y = np.zeros((n1 + 1, n2 + 1)); Y[1:, 1:] = y
    return tuple(q[1:, 1:] for q in quality(X, Y, n1, n2))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

@dataclass
class MeshConfig:
    n1: int = 51
    n2: int = 26
    tau: float = 1e-5          # pseudo-time step   (t)
    eps: float = 1e-4          # stop when max node shift < eps
    max_iter: int = 5000       # m
    snapshot_at: int = 4969    # iteration written to rezsurf1.dat (0 = never)
    fix_metric_stencil: bool = False   # True -> use 0.5 in the interior d/dxi


@dataclass
class MeshResult:
    x: np.ndarray                     # final grid (n1, n2)
    y: np.ndarray
    best: dict                        # 'Q1'..'Q4' -> dict(x, y, q, iteration, value)
    history: np.ndarray               # columns: n, hor, q1, q2, q3, q4
    iterations: int


def initial_grid(xo, yo, xi, yi, n2):
    """Straight-line interpolation between outer (j=1) and inner (j=n2) contour."""
    # same arithmetic as Fortran: x1 + (j-1)*du
    du = (np.asarray(xi) - np.asarray(xo)) / (n2 - 1)
    dv = (np.asarray(yi) - np.asarray(yo)) / (n2 - 1)
    j = np.arange(n2)
    x = np.asarray(xo)[:, None] + j[None, :] * du[:, None]
    y = np.asarray(yo)[:, None] + j[None, :] * dv[:, None]
    x[:, -1] = xi
    y[:, -1] = yi
    return x, y


def generate(xo, yo, xi, yi, cfg=MeshConfig(), verbose=True, on_snapshot=None,
             print_every=1, callback=None):
    """Run the mesh generator.  xo/yo: outer contour, xi/yi: inner contour."""
    n1, n2 = cfg.n1, cfg.n2
    h1 = 1.0 / (n1 - 1)
    h2 = 1.0 / (n2 - 1)
    x0, y0 = initial_grid(xo, yo, xi, yi, n2)
    X = np.zeros((n1 + 1, n2 + 1)); X[1:, 1:] = x0
    Y = np.zeros((n1 + 1, n2 + 1)); Y[1:, 1:] = y0
    jak = np.ones((n1 + 1, n2 + 1))    # weight function (=1 in the original)
    xx = np.zeros_like(X); xy = np.zeros_like(X); yx = np.zeros_like(X); yy = np.zeros_like(X)
    factor = 0.5 if cfg.fix_metric_stencil else 1.0

    names = ("Q1", "Q2", "Q3", "Q4")
    best = {k: dict(x=np.zeros((n1, n2)), y=np.zeros((n1, n2)), q=np.zeros((n1, n2)),
                    iteration=0, value=0.0) for k in names}
    hist = []
    n = 0
    while True:
        n += 1
        if n > cfg.max_iter:
            n = cfg.max_iter
            break
        hor = mesh_iteration(X, Y, n1, n2, h1, h2, cfg.tau, jak, xx, xy, yx, yy, factor)
        qs = quality(X, Y, n1, n2)
        mins = []
        for k, q in zip(names, qs):
            qmin = min(1.0, float(q[1:, 1:].min()))
            mins.append(qmin)
            if qmin > best[k]["value"]:
                best[k].update(x=X[1:, 1:].copy(), y=Y[1:, 1:].copy(),
                               q=q[1:, 1:].copy(), iteration=n, value=qmin)
        if cfg.snapshot_at and n == cfg.snapshot_at and on_snapshot is not None:
            on_snapshot(X[1:, 1:].copy(), Y[1:, 1:].copy(), [q[1:, 1:].copy() for q in qs])
        hist.append([n, hor] + mins)
        if verbose and (n % print_every == 0):
            print(f"{n:6d}{hor:15.10f}" + "".join(f"{v:10.4f}" for v in mins))
        if callback is not None and callback(n, hor, mins) is True:
            break
        if not hor > cfg.eps:
            break

    if verbose:
        print("L1={:4d}   L2={:4d}   L3={:4d}   L4={:4d}".format(
            *[best[k]["iteration"] for k in names]))
        print("Q1={:10.4f}   Q2={:10.4f}   Q3={:10.4f}   Q4={:10.4f}".format(
            *[best[k]["value"] for k in names]))
    return MeshResult(X[1:, 1:].copy(), Y[1:, 1:].copy(), best, np.array(hist), n)


def read_boundary(path, n1):
    """rez_param.dat -> (outer x, outer y, inner x, inner y), n1 points each."""
    x, y = io.read_points(path)
    return x[:n1], y[:n1], x[n1:2 * n1], y[n1:2 * n1]


def run(boundary_file, out_dir=".", cfg=MeshConfig(), verbose=True, print_every=1,
        callback=None):
    """Full program: same output files as Setka_ED2_II.exe."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xo, yo, xi, yi = read_boundary(boundary_file, cfg.n1)

    def snap(x, y, qs):
        z = np.zeros_like(x)
        with open(out / "rezsurf1.dat", "w") as fh:
            io.write_tecplot_zone(fh, ["X", "Y", "Q1", "Q2", "Q3", "Q4"], [x, y, *qs],
                                  fmts=["%15.5f"] * 6)
            io.write_tecplot_zone(fh, ["X", "Y", "Q1", "Q2", "Q3", "Q4"], [x, y, z, z, z, z],
                                  title=False, fmts=["%15.5f"] * 6)

    res = generate(xo, yo, xi, yi, cfg, verbose, on_snapshot=snap, print_every=print_every,
                   callback=callback)

    b1 = res.best["Q1"]
    z = np.zeros((cfg.n1, cfg.n2))
    io.write_grid_array(out / "xxx.dat", b1["x"])
    io.write_grid_array(out / "yyy.dat", b1["y"])
    io.write_grid_array(out / "zzz.dat", z)
    for k, name in enumerate(("Q1", "Q2", "Q3", "Q4"), start=1):
        b = res.best[name]
        with open(out / f"surf{k}.dat", "w") as fh:
            io.write_tecplot_zone(fh, ["X", "Y", "T"], [b["x"], b["y"], z], fmts=["%15.5f"] * 3)
            io.write_tecplot_zone(fh, ["X", "Y", "T"], [b["x"], b["y"], b["q"]], title=False,
                                  fmts=["%15.5f"] * 3)
    np.savez(out / "mesh_result.npz", x=res.x, y=res.y, history=res.history,
             **{f"best_{k}_{f}": res.best[k][f] for k in res.best for f in ("x", "y", "q")})
    return res
