"""Regression tests: the Python port must reproduce the original Fortran.

Reference data in tests/data was produced by compiling the ORIGINAL .for
files with gfortran (double precision, zero-initialised memory) and running
them on the original input files.

Run:  python -m pytest -q
"""

from pathlib import Path

import numpy as np
import pytest

from ringflow import boundary, mesh, convection, io
from ringflow.tridiag import cyclic_sweep, dirichlet_sweep

DATA = Path(__file__).parent / "data"


def _log_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            pass
    return rows


# --------------------------------------------------------------------------
# tridiagonal solvers against dense linear algebra
# --------------------------------------------------------------------------

def test_dirichlet_sweep_matches_dense_solve():
    rng = np.random.default_rng(0)
    n = 12
    A = np.zeros(n + 1); B = np.zeros(n + 1); F = np.zeros(n + 1)
    A[2:n] = rng.random(n - 2); B[2:n] = rng.random(n - 2); F[2:n] = rng.random(n - 2)
    left, right = 0.3, -1.2
    out = np.zeros(n + 1)
    dirichlet_sweep(A, B, F, n, left, right, out)
    M = np.zeros((n, n)); r = np.zeros(n)
    M[0, 0] = 1; r[0] = left
    M[-1, -1] = 1; r[-1] = right
    for k in range(2, n):
        M[k - 1, k - 2] = -A[k]; M[k - 1, k - 1] = 1 + A[k] + B[k]; M[k - 1, k] = -B[k]
        r[k - 1] = F[k]
    assert np.allclose(out[1:], np.linalg.solve(M, r))


def test_cyclic_sweep_matches_dense_solve():
    rng = np.random.default_rng(1)
    n1 = 15
    m = n1 - 1                         # unique nodes 1..n1-1
    A = np.zeros(n1 + 1); B = np.zeros(n1 + 1); F = np.zeros(n1 + 1)
    A[1:n1] = rng.random(m); B[1:n1] = rng.random(m); F[1:n1] = rng.random(m)
    out = np.zeros(n1 + 1)
    cyclic_sweep(A, B, F, n1, out)
    # What the original algorithm really solves: a ring of n1 unknowns where
    # the seam node n1 is a separate node carrying node 1's coefficients
    # (left neighbour n1-1, right neighbour 1).  See tridiag.cyclic_sweep.
    N = n1
    M = np.zeros((N, N)); r = np.zeros(N)
    for k in range(N):
        i = k + 1 if k < N - 1 else 1          # node n1 uses row-1 coefficients
        M[k, k] = 1 + A[i] + B[i]
        M[k, (k - 1) % N] -= A[i]
        M[k, (k + 1) % N] -= B[i]
        r[k] = F[i]
    sol = np.linalg.solve(M, r)
    assert np.allclose(out[1:], sol)


# --------------------------------------------------------------------------
# full programs against Fortran output
# --------------------------------------------------------------------------

def test_boundary_matches_fortran(tmp_path):
    d = DATA / "boundary"
    boundary.run(d / "bottom_border.dat", d / "top_border.dat", tmp_path, n1=51, verbose=False)
    a = np.loadtxt(d / "rez_param.dat")
    b = np.loadtxt(tmp_path / "rez_param.dat")
    assert np.abs(a - b).max() < 1e-8


def test_mesh_matches_fortran(tmp_path):
    d = DATA / "mesh"
    res = mesh.run(d / "rez_param.dat", tmp_path, mesh.MeshConfig(), verbose=False)
    for f in ("xxx.dat", "yyy.dat"):
        a = np.array((d / f).read_text().split(), float)
        b = np.array((tmp_path / f).read_text().split(), float)
        assert np.abs(a - b).max() < 1e-9, f
    # best iterations reported by the Fortran program: L1=4969 L2=3380 L3=2 L4=595
    assert [res.best[k]["iteration"] for k in ("Q1", "Q2", "Q3", "Q4")] == [4969, 3380, 2, 595]
    # iteration log (n, hor, q1..q4) identical
    ref = np.array([r for r in _log_rows(d / "log.txt") if len(r) == 6])
    assert ref.shape == res.history.shape
    assert np.abs(ref - res.history).max() < 1e-4


@pytest.mark.parametrize("case", ["ring20", "curved20"])
def test_flow_matches_fortran(case, tmp_path):
    d = DATA / case
    if case == "ring20":
        cfg = convection.preset("ring", max_steps=20, snapshot_every=0, write_initial=False)
    else:
        cfg = convection.preset("curved", max_steps=20, snapshot_every=0, write_initial=False,
                                mesh_x=str(d / "xxx.dat"), mesh_y=str(d / "yyy.dat"))
    res = convection.solve(cfg, None, verbose=False)
    ref = np.load(d / "fortran_step20.npz")
    mine = dict(X=res.x, Y=res.y, U=res.u, V=res.v, W=res.w, P=res.psi, T=res.T)
    for k in mine:
        # reference was printed with 4 (X, Y) or 7 decimals
        tol = 6e-5 if k in "XY" else 6e-8
        assert np.abs(ref[k] - mine[k]).max() < tol, k
    log = np.array([r for r in _log_rows(d / "log.txt") if len(r) == 5])
    assert np.abs(log - res.history).max() < 1e-9


def test_tecplot_roundtrip(tmp_path):
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0), indexing="ij")
    with open(tmp_path / "t.dat", "w") as fh:
        io.write_tecplot_zone(fh, ["X", "Y", "T"], [x, y, x * y + 123456789.0])
    z = io.read_tecplot(tmp_path / "t.dat")[0]
    assert np.allclose(z["T"], x * y + 123456789.0)


@pytest.mark.parametrize("kind", ["ring", "elliptic", "eccentric", "square_hole"])
def test_ready_made_shapes_give_valid_mesh(kind):
    from ringflow import shapes, quality
    xo, yo, xi, yi = shapes.make(kind)
    fp = (np.zeros(51), np.zeros(51))
    o = boundary.redistribute(xo, yo, 51, _fp=fp)
    i = boundary.redistribute(xi, yi, 51, _fp=fp, closed_fix_first_to_last=True)
    res = mesh.generate(o.x, o.y, i.x, i.y, mesh.MeshConfig(max_iter=300), verbose=False)
    _, stats = quality.analyze(res.best["Q1"]["x"], res.best["Q1"]["y"])
    assert stats["Q1"]["bad"] == 0 and stats["Q1"]["min"] > 0.5


def test_shape_rejects_hole_outside():
    from ringflow import shapes
    with pytest.raises(ValueError):
        shapes.make("elliptic", b_in=3.0)
