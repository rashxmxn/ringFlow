"""Mesh quality criteria: analysis and pictures.

The criteria are computed exactly as in subroutine KRIT_KA of
Setka_ED2_II.for (see mesh.quality).  Every cell with vertices 1, 2, 3, 4 is
split by its diagonals into the triangles 123, 134, 124, 234 (Fig. 1 of the
thesis, p. 72); all criteria lie in [0, 1], 1 = ideal cell, <= 0 = degenerate
(non-convex / inverted) cell.

  Q1  non-degeneracy (convexity):   2*min(S123, S134, S124, S234) / S_cell
  Q2  orthogonality:                min sin(angle) over the 4 cell angles
  Q3  non-elongation (aspect ratio): min edge / max edge
  Q4  local uniformity:             min(S_cell/S_avg, S_avg/S_cell)

Q2, Q3, Q4 are the three characteristics listed in the thesis
("ортогональность, невытянутость ячеек, локальная равномерность");
Q1 is the triangle-area check of Fig. 1.
"""

import numpy as np

from .mesh import grid_quality

CRITERIA = {
    "Q1": dict(ru="Невырожденность (по площадям треугольников)", short="Невырожденность",
               en="non-degeneracy / convexity",
               formula=r"$Q_1 = 2\,\min(S_{123},S_{134},S_{124},S_{234})\,/\,S_{1234}$"),
    "Q2": dict(ru="Ортогональность", short="Ортогональность",
               en="orthogonality",
               formula=r"$Q_2 = \min_k \sin\varphi_k$  (углы ячейки)"),
    "Q3": dict(ru="Невытянутость ячеек", short="Невытянутость ячеек",
               en="aspect ratio",
               formula=r"$Q_3 = \min_k l_k \,/\, \max_k l_k$  (длины сторон)"),
    "Q4": dict(ru="Локальная равномерность", short="Локальная равномерность",
               en="area uniformity",
               formula=r"$Q_4 = \min(S/S_{ср},\ S_{ср}/S)$"),
}


def analyze(x, y):
    """Criteria per node + statistics.  Returns (fields, stats).

    fields: {'Q1': (n1, n2) array, ...}
    stats : {'Q1': dict(min, mean, max, bad), ...}  (bad = cells with Q <= 0)
    Statistics are taken over real cells only (the duplicated last row /
    column are copies).
    """
    qs = grid_quality(np.asarray(x, float), np.asarray(y, float))
    fields, stats = {}, {}
    for k, q in zip(CRITERIA, qs):
        fields[k] = q
        cells = q[:-1, :-1]
        stats[k] = dict(min=float(cells.min()), mean=float(cells.mean()),
                        max=float(cells.max()), bad=int((cells <= 0).sum()))
    return fields, stats


def ring_mesh(n1=51, n2=26, r_in=0.5, r_out=2.0):
    """Circular ring mesh as in Fig. 6 of the thesis (j=1 outer, j=n2 inner)."""
    phi = np.linspace(0, 2 * np.pi, n1)[:, None]
    r = np.linspace(r_out, r_in, n2)[None, :]
    return r * np.cos(phi), r * np.sin(phi)


# ---------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------

from .plotting import new_fig  # noqa: E402


def plot_3d(x, y, q, key, path=None, title=None, ax=None, zlim=(0, 1)):
    """3-D surface W = Q(x, y) over the mesh -- same style as Fig. 9-10."""
    own = ax is None
    if own:
        fig = new_fig((8, 6))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, np.clip(q, *zlim), cmap="RdYlGn", vmin=0, vmax=1,
                    edgecolor="k", linewidth=0.15, antialiased=True)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("W")
    ax.view_init(elev=32, azim=-55)
    ax.set_title(title or f"{key}: {CRITERIA[key]['ru']}", fontsize=10)
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=140)
    return ax


def plot_2d(x, y, q, key, path=None, title=None, ax=None):
    """Mesh coloured by the criterion (green = good, red = bad)."""
    own = ax is None
    if own:
        fig = new_fig((7, 5.5)); ax = fig.add_subplot()
    m = ax.pcolormesh(x, y, q, cmap="RdYlGn", vmin=0, vmax=1, shading="gouraud")
    for i in range(x.shape[0]):
        ax.plot(x[i, :], y[i, :], "k", lw=0.25)
    for j in range(x.shape[1]):
        ax.plot(x[:, j], y[:, j], "k", lw=0.25)
    ax.set_aspect("equal")
    ax.set_title(title or f"{key}: {CRITERIA[key]['ru']}", fontsize=10)
    ax.figure.colorbar(m, ax=ax, shrink=0.8)
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=140)
    return ax


def plot_panel(x, y, fields, stats, path, suptitle="", kind="2d"):
    """All four criteria on one figure."""
    fig = new_fig((13, 10))
    for n, k in enumerate(CRITERIA, start=1):
        s = stats[k]
        t = f"{k}: {CRITERIA[k]['ru']}\nmin={s['min']:.3f}  mean={s['mean']:.3f}"
        if kind == "3d":
            ax = fig.add_subplot(2, 2, n, projection="3d")
            plot_3d(x, y, fields[k], k, ax=ax, title=t)
        else:
            ax = fig.add_subplot(2, 2, n)
            plot_2d(x, y, fields[k], k, ax=ax, title=t)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_history(history, best, path):
    """min Q_k of the mesh vs iteration of the mesh generator (+ best points)."""
    h = np.asarray(history)
    fig = new_fig((9, 4.5)); ax = fig.add_subplot()
    for c, k in enumerate(CRITERIA, start=2):
        line, = ax.plot(h[:, 0], h[:, c], label=f"{k}: {CRITERIA[k]['ru']}")
        b = best[k]
        ax.plot([b["iteration"]], [b["value"]], "o", color=line.get_color())
    ax.set_ylim(-0.6, 1.0)
    low = h[:, 2:6].min()
    if low < -0.6:
        ax.text(0.01, 0.03, f"в начале min Q доходит до {low:.1f} (вывернутые ячейки), ось обрезана",
                transform=ax.transAxes, fontsize=8, color="0.3")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("итерация генератора сетки")
    ax.set_ylabel("min Q по сетке")
    ax.set_title("Качество сетки в процессе построения (точки - лучшая сетка)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def stats_table(stats):
    rows = ["criterion,name,min,mean,max,degenerate_cells"]
    for k, s in stats.items():
        rows.append(f"{k},{CRITERIA[k]['ru']},{s['min']:.5f},{s['mean']:.5f},"
                    f"{s['max']:.5f},{s['bad']}")
    return "\n".join(rows) + "\n"
