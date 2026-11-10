"""Tridiagonal solvers ("progonka") used throughout the original code.

INDEXING CONVENTION (used in the whole package)
-----------------------------------------------
All internal arrays are allocated one element larger in every direction and
index 0 is never used, so ``a[i, j]`` here is exactly ``a(i, j)`` in the
Fortran source.  This makes the port auditable line by line.  Public functions
return ordinary 0-based arrays (``a[1:, 1:]``).

Both solvers solve, for node k,

    -A[k] * u[k-1] + C[k] * u[k] - B[k] * u[k+1] = F[k],   C[k] = 1 + A[k] + B[k]

which is the implicit pseudo-time step ``u - tau * L(u) = rhs`` of the
original ADI schemes.
"""

import numpy as np

from ._jit import njit


@njit
def cyclic_sweep(A, B, F, n1, out):
    """Periodic progonka along a closed grid line (node n1 == node 1).

    A, B, F are 1-based arrays with valid entries 1..n1-1.  The result is
    written to out[1..n1].  Direct port of the alf/bt/gam + p/q "cyclic
    progonka" block in Setka_ED2_II.for (drov1) and Dvizh_psi_om_Kolc.for.

    Note (behaviour of the original, kept on purpose): the algorithm solves
    a ring of n1 unknowns in which the seam node n1 is treated as a separate
    node carrying node 1's coefficients (left neighbour n1-1, right
    neighbour 1).  Therefore out[n1] is close to, but not exactly equal to,
    out[1]; the calling code later overwrites node n1 with node 1.
    """
    n12 = n1 - 1
    alf = np.zeros(n1 + 2)
    bt = np.zeros(n1 + 2)
    gam = np.zeros(n1 + 2)
    p = np.zeros(n1 + 2)
    q = np.zeros(n1 + 2)

    a1 = A[1]
    b1 = B[1]
    c1 = 1.0 + a1 + b1
    f1 = F[1]
    alf[2] = b1 / c1
    bt[2] = f1 / c1
    gam[2] = a1 / c1
    for i in range(2, n12 + 1):
        a = A[i]
        b = B[i]
        c = 1.0 + a + b
        den = c - a * alf[i]
        alf[i + 1] = b / den
        bt[i + 1] = (F[i] + a * bt[i]) / den
        gam[i + 1] = a * gam[i] / den

    den = c1 - a1 * alf[n1]
    aaa = b1 / den
    bbb = (f1 + a1 * bt[n1]) / den
    ggg = a1 * gam[n1] / den

    p[n12] = bt[n1]
    q[n12] = alf[n1] + gam[n1]
    for i in range(2, n12 + 1):
        l = n1 - i
        p[l] = alf[l + 1] * p[l + 1] + bt[l + 1]
        q[l] = alf[l + 1] * q[l + 1] + gam[l + 1]

    out[n1] = (bbb + aaa * p[1]) / (1.0 - aaa * q[1] - ggg)
    for i in range(1, n12 + 1):
        out[i] = p[i] + out[n1] * q[i]


@njit
def dirichlet_sweep(A, B, F, n2, left, right, out):
    """Ordinary progonka on 1..n2 with fixed end values.

    Interior equations use A, B, F at 2..n2-1; out[1] = left, out[n2] = right.
    Direct port of drov2 in Setka_ED2_II.for.
    """
    alf = np.zeros(n2 + 2)
    bt = np.zeros(n2 + 2)
    alf[2] = 0.0
    bt[2] = left
    for j in range(2, n2):
        a = A[j]
        b = B[j]
        c = 1.0 + a + b
        den = c - a * alf[j]
        alf[j + 1] = b / den
        bt[j + 1] = (F[j] + a * bt[j]) / den
    out[n2] = right
    for j in range(1, n2):
        l = n2 - j
        out[l] = alf[l + 1] * out[l + 1] + bt[l + 1]
