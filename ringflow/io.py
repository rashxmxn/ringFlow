"""File input/output, compatible with the files the Fortran programs used.

* Tecplot ASCII "POINT" files (rezsurf*.dat, surf*.dat, metr_koef.dat)
* Grid coordinate files xxx.dat / yyy.dat / zzz.dat (Fortran format 51(f15.10))
* Two-column point files (bottom_border.dat, rez_param.dat ...)
* Extra (new): NumPy .npz and VTK legacy files for ParaView
"""

from pathlib import Path
import re

import numpy as np


# ---------------------------------------------------------------------------
# Plain point lists
# ---------------------------------------------------------------------------

def read_points(path):
    """Read a whitespace separated 2-column (x, y) file."""
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0].copy(), data[:, 1].copy()


def write_points(path, x, y, fmt="{:15.8f}{:15.8f}"):
    """Fortran ``format(f15.8,f15.8)`` style output."""
    with open(path, "w") as fh:
        for a, b in zip(x, y):
            fh.write(fmt.format(a, b) + "\n")


# ---------------------------------------------------------------------------
# xxx.dat / yyy.dat  (Fortran: write(54,105) ((x(i,j),j=1,n2),i=1,n1),
#                     105 format(51(f15.10)))
# ---------------------------------------------------------------------------

def write_grid_array(path, a, per_line=51):
    """Write a (n1, n2) array in the order j fastest, then i, 51 per line."""
    flat = np.asarray(a, float).reshape(-1)  # C order: last index (j) fastest
    with open(path, "w") as fh:
        for k in range(0, flat.size, per_line):
            fh.write("".join(f"{v:15.10f}" for v in flat[k:k + per_line]) + "\n")


def read_grid_array(path, n1, n2):
    """Inverse of :func:`write_grid_array` (Fortran list-directed read)."""
    vals = np.array(Path(path).read_text().split(), dtype=float)
    if vals.size < n1 * n2:
        raise ValueError(f"{path}: expected {n1*n2} numbers, found {vals.size}")
    return vals[: n1 * n2].reshape(n1, n2)


# ---------------------------------------------------------------------------
# Tecplot
# ---------------------------------------------------------------------------

def write_tecplot_zone(fh, variables, arrays, title=True, fmts=None):
    """Append one Tecplot POINT zone.

    arrays are (n1, n2) arrays; points are written with i fastest, like the
    Fortran ``do j=1,n2 / do i=1,n1`` loops.
    """
    n1, n2 = np.shape(arrays[0])
    if title:
        fh.write(" TITLE = Test\n")
    fh.write(" VARIABLES = " + ",".join(variables) + "\n")
    fh.write(f" ZONE T=Test,I={n1},J={n2},F=POINT\n")
    cols = np.column_stack([np.asarray(a, float).T.reshape(-1) for a in arrays])
    if fmts is None:
        fmts = ["%15.7f"] * len(arrays)
    # same column widths as the Fortran formats, but always one separating
    # space so that large numbers never get glued together
    safe = [re.sub(r"%(\d+)\.", lambda m: " %" + str(int(m.group(1)) - 1) + ".", f) for f in fmts]
    np.savetxt(fh, cols, fmt="".join(safe))


def _to_float(tok):
    tok = tok.strip()
    if not tok or "*" in tok:          # Fortran prints ***** on overflow
        return float("nan")
    return float(tok.replace("D", "E").replace("d", "e"))


def _parse_data_line(line, nvars):
    """Split a data line; fall back to Fortran fixed-width fields when big
    numbers are glued together (e.g. '0.32831011490.9882609')."""
    toks = line.split()
    glued = "*" in line or any(t.count(".") > 1 for t in toks)
    if not glued:
        try:
            return [_to_float(t) for t in toks]
        except ValueError:
            pass
    body = line.rstrip()
    for widths in ([8, 8] + [15] * (nvars - 2), [15] * nvars, [15] * 4, [15] * 3):
        if sum(widths) == len(body):
            out, pos = [], 0
            for w in widths:
                out.append(_to_float(body[pos:pos + w]))
                pos += w
            return out
    raise ValueError(f"cannot parse Tecplot data line: {line!r}")


def read_tecplot(path):
    """Read every zone of a Tecplot ASCII POINT file.

    Returns a list of dicts ``{variable_name: array(n1, n2)}``.  Works on the
    original 2015 output files too (also the ones where a record was split
    over two lines by Fortran format reversion).
    """
    zones = []
    variables = None
    current = None
    numbers = []

    def flush():
        if current is None:
            return
        nv = len(variables)
        n1, n2 = current
        vals = np.array(numbers[: n1 * n2 * nv], dtype=float)
        if vals.size < n1 * n2 * nv:  # truncated file (e.g. run aborted)
            return
        vals = vals.reshape(n2, n1, nv)
        zones.append({v: vals[:, :, k].T.copy() for k, v in enumerate(variables)})

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            u = s.upper()
            if u.startswith("TITLE"):
                continue
            if u.startswith("VARIABLES"):
                variables = [v.strip().strip('"') for v in s.split("=", 1)[1].split(",")]
                continue
            if u.startswith("ZONE"):
                flush()
                numbers = []
                mi = re.search(r"\bI\s*=\s*(\d+)", s, re.I)
                mj = re.search(r"\bJ\s*=\s*(\d+)", s, re.I)
                current = (int(mi.group(1)), int(mj.group(1)) if mj else 1)
                continue
            numbers.extend(_parse_data_line(line.rstrip("\r\n"), len(variables)))
    flush()
    return zones


# ---------------------------------------------------------------------------
# New output formats
# ---------------------------------------------------------------------------

def write_vtk(path, x, y, fields):
    """Write a legacy-VTK structured grid (open it in ParaView / VisIt).

    fields: dict name -> (n1, n2) array.
    """
    x = np.asarray(x, float)
    n1, n2 = x.shape
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nringflow output\nASCII\n")
        fh.write(f"DATASET STRUCTURED_GRID\nDIMENSIONS {n1} {n2} 1\n")
        fh.write(f"POINTS {n1*n2} double\n")
        pts = np.column_stack([x.T.reshape(-1), np.asarray(y).T.reshape(-1), np.zeros(n1 * n2)])
        np.savetxt(fh, pts, fmt="%.10g")
        fh.write(f"POINT_DATA {n1*n2}\n")
        for name, arr in fields.items():
            fh.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(fh, np.asarray(arr, float).T.reshape(-1), fmt="%.10g")
