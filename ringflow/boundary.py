"""Boundary point redistribution  --  port of ``physical_area/interp.for``.

What it does
------------
The physical domain is a curvilinear ring bounded by two closed polylines:
``bottom_border.dat`` (outer contour) and ``top_border.dat`` (inner contour,
the "hole").  Those polylines are digitised with an arbitrary number of
points.  The mesh generator needs exactly ``n1`` points on each contour, so
this program places ``n1`` points along each polyline, equally distributed
with respect to a weight  w = |dr/ds|  (the ratio between the chord length of
the piecewise-linear interpolant and the arc-length parameter).  The weights
depend on the current point positions, so the 1-D equidistribution equation

    w[i-1] * (s[i] - s[i-1]) = w[i] * (s[i+1] - s[i])

is solved by simple iteration with a tridiagonal (progonka) solve per step.

Outputs (same names as the Fortran program)
-------------------------------------------
* ``rez_param.dat`` : n1 points on the outer contour, then n1 on the inner
  one.  This is the input of the mesh generator.
* ``rez_kord.dat``  : the raw input polylines, copied (for plotting).

Note: the original file has ``n1=21``; the ``rez_param.dat`` actually used by
the mesh generator in Dvusv_set_dvish was produced with ``n1=51``.
"""

from dataclasses import dataclass, field

import numpy as np

from . import io


@dataclass
class BoundaryResult:
    x: np.ndarray            # (n1,) redistributed points
    y: np.ndarray
    s: np.ndarray            # arc-length parameters of the points
    iterations: int
    history: list = field(default_factory=list)  # max |s_new - s_old| per iteration


def _locate(p, dl, xs, ys, fp1, fp2, n1):
    """Linear interpolation of the polyline at arc-length positions p.

    Faithful to the Fortran search: if p is not inside any [dl(j), dl(j+1))
    interval the previous value of fp is kept.
    """
    nb = dl.size
    for i in range(n1):
        for j in range(nb - 1):
            if dl[j] <= p[i] < dl[j + 1]:
                r = (p[i] - dl[j]) / (dl[j + 1] - dl[j])
                fp1[i] = xs[j] + (xs[j + 1] - xs[j]) * r
                fp2[i] = ys[j] + (ys[j + 1] - ys[j]) * r


def redistribute(xs, ys, n1=51, max_iter=1000, eps=1e-5, verbose=False,
                 _fp=None, closed_fix_first_to_last=False):
    """Place n1 points along the polyline (xs, ys).  See module docstring."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    nb = xs.size
    n12 = n1 - 1
    h1 = 1.0 / n12

    dl = np.zeros(nb)
    dl[1:] = np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))
    L = dl[-1]

    # fp arrays are static in the Fortran program and keep values between
    # the outer- and inner-contour passes; mimic that when _fp is passed.
    fp1, fp2 = _fp if _fp is not None else (np.zeros(n1), np.zeros(n1))

    p2 = L * np.arange(n1) * h1
    history = []
    n = 0
    while True:
        n += 1
        if n > max_iter:
            break
        p1 = p2.copy()
        _locate(p1, dl, xs, ys, fp1, fp2, n1)
        fp1[n1 - 1] = xs[-1]
        fp2[n1 - 1] = ys[-1]

        # 1-based work arrays (index 0 unused) so this reads like the Fortran
        W = np.zeros(n1 + 1)
        for i in range(1, n1):
            dp = p1[i] - p1[i - 1]                 # p1(i+1)-p1(i)
            px = (fp1[i] - fp1[i - 1]) / dp
            py = (fp2[i] - fp2[i - 1]) / dp
            W[i] = np.sqrt(px * px + py * py)      # wjak(i)

        alf = np.zeros(n1 + 2)
        bet = np.zeros(n1 + 2)
        for i in range(2, n12 + 1):
            ai = W[i - 1] / h1 ** 2
            bi = W[i] / h1 ** 2
            ci = ai + bi
            alf[i + 1] = bi / (ci - ai * alf[i])
            bet[i + 1] = ai * bet[i] / (ci - ai * alf[i])
        P = np.zeros(n1 + 2)
        P[n1] = L
        for i in range(1, n12 + 1):
            l = n1 - i
            P[l] = alf[l + 1] * P[l + 1] + bet[l + 1]
        p2 = P[1:n1 + 1].copy()

        hor = float(np.max(np.abs(p2 - p1)))
        history.append(hor)
        if verbose:
            print(f"{n:4d}{hor:15.8f}")
        if hor < eps:
            break

    _locate(p2, dl, xs, ys, fp1, fp2, n1)
    if closed_fix_first_to_last:
        # Fortran (inner contour) sets the first point to the LAST input
        # point; identical for a closed contour.
        fp1[0], fp2[0] = xs[-1], ys[-1]
    else:
        fp1[0], fp2[0] = xs[0], ys[0]
    fp1[n1 - 1], fp2[n1 - 1] = xs[-1], ys[-1]
    return BoundaryResult(fp1.copy(), fp2.copy(), p2, min(n, max_iter), history)


def run(bottom_file, top_file, out_dir=".", n1=51, max_iter=1000, eps=1e-5,
        verbose=True):
    """Full program: both contours -> rez_param.dat + rez_kord.dat."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xb, yb = io.read_points(bottom_file)
    xt, yt = io.read_points(top_file)
    fp = (np.zeros(n1), np.zeros(n1))
    outer = redistribute(xb, yb, n1, max_iter, eps, verbose, _fp=fp)
    inner = redistribute(xt, yt, n1, max_iter, eps, verbose, _fp=fp,
                         closed_fix_first_to_last=True)

    io.write_points(out / "rez_kord.dat", np.r_[xb, xt], np.r_[yb, yt])
    io.write_points(out / "rez_param.dat", np.r_[outer.x, inner.x], np.r_[outer.y, inner.y])
    return outer, inner
