"""Ready-made doubly-connected domains (outer + inner contour).

Each function returns four arrays (xo, yo, xi, yi): closed polylines
(last point == first point) in the same orientation and with the same
starting point as the dissertation data (start at the leftmost point,
counter-clockwise), so they can be fed to boundary.redistribute /
mesh.generate exactly like bottom_border.dat / top_border.dat.
"""

import numpy as np

from . import io


def _angles(n):
    # start at the leftmost point (angle pi), counter-clockwise
    return np.pi + 2 * np.pi * np.arange(n + 1) / n


def ring(r_out=2.0, r_in=0.5, n=200):
    """Circular ring (concentric circles)."""
    t = _angles(n)
    return r_out * np.cos(t), r_out * np.sin(t), r_in * np.cos(t), r_in * np.sin(t)


def elliptic(a_out=3.0, b_out=2.0, a_in=1.0, b_in=0.5, n=200):
    """Elliptic ring: outer ellipse with semi-axes a_out, b_out; inner a_in, b_in."""
    t = _angles(n)
    return a_out * np.cos(t), b_out * np.sin(t), a_in * np.cos(t), b_in * np.sin(t)


def eccentric(r_out=2.0, r_in=0.6, shift=0.6, n=200):
    """Outer circle, inner circle shifted by `shift` along x."""
    t = _angles(n)
    return r_out * np.cos(t), r_out * np.sin(t), shift + r_in * np.cos(t), r_in * np.sin(t)


def square_hole(side=4.0, r_in=0.7, n=200):
    """Square with a circular hole in the middle."""
    n = int(np.ceil(n / 8) * 8)          # corners fall exactly on grid angles
    t = _angles(n)
    r = (side / 2) / np.maximum(np.abs(np.cos(t)), np.abs(np.sin(t)))
    return r * np.cos(t), r * np.sin(t), r_in * np.cos(t), r_in * np.sin(t)


# name -> (function, [(parameter, label, default), ...])
SHAPES = {
    "ring": (ring, [("r_out", "Внешний радиус R", 2.0), ("r_in", "Внутренний радиус r", 0.5)]),
    "elliptic": (elliptic, [("a_out", "Внешняя полуось a", 3.0), ("b_out", "Внешняя полуось b", 2.0),
                            ("a_in", "Внутренняя полуось a", 1.0), ("b_in", "Внутренняя полуось b", 0.5)]),
    "eccentric": (eccentric, [("r_out", "Внешний радиус R", 2.0), ("r_in", "Внутренний радиус r", 0.6),
                              ("shift", "Смещение отверстия", 0.6)]),
    "square_hole": (square_hole, [("side", "Сторона квадрата", 4.0), ("r_in", "Радиус отверстия", 0.7)]),
}

TITLES = {
    "ring": "Круговое кольцо",
    "elliptic": "Эллиптическое кольцо",
    "eccentric": "Кольцо со смещенным отверстием",
    "square_hole": "Квадрат с круглым отверстием",
}


def check(xo, yo, xi, yi):
    """Basic sanity check: the inner contour must lie inside the outer one."""
    from matplotlib.path import Path as MPath
    outer = MPath(np.column_stack([xo, yo]))
    if not outer.contains_points(np.column_stack([xi, yi])).all():
        raise ValueError("Внутренний контур должен целиком лежать внутри внешнего.")


def make(kind, **params):
    func, spec = SHAPES[kind]
    kw = {k: params.get(k, d) for k, _, d in spec}
    xo, yo, xi, yi = func(**kw)
    check(xo, yo, xi, yi)
    return xo, yo, xi, yi


def write(kind, out_dir, **params):
    """Write bottom_border.dat (outer) and top_border.dat (inner); return paths."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xo, yo, xi, yi = make(kind, **params)
    io.write_points(out / "bottom_border.dat", xo, yo)
    io.write_points(out / "top_border.dat", xi, yi)
    return out / "bottom_border.dat", out / "top_border.dat"
