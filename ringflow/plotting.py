"""Visualisation (new -- the original relied on Tecplot for all pictures).

* plot_mesh      : grid lines, optionally coloured by a quality criterion
* plot_fields    : temperature + stream-function isolines + velocity vectors
* plot_history   : convergence history
* animate        : GIF/MP4 from the snapshots in a rezsurf.dat file
                   (replaces the .avi files made in Tecplot for the talk)
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def new_fig(figsize=(8, 6)):
    """Off-screen figure (no pyplot): safe in threads and inside the GUI."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def _grid_lines(ax, x, y, step=1, color="k", lw=0.4):
    for i in range(0, x.shape[0], step):
        ax.plot(x[i, :], y[i, :], color=color, lw=lw)
    for j in range(0, x.shape[1], step):
        ax.plot(x[:, j], y[:, j], color=color, lw=lw)


def plot_mesh(x, y, q=None, title="Mesh", path=None, qlabel="Q1"):
    fig = new_fig((8, 6)); ax = fig.add_subplot()
    if q is not None:
        m = ax.pcolormesh(x, y, q, shading="gouraud", cmap="RdYlGn", vmin=0, vmax=1)
        fig.colorbar(m, ax=ax, label=qlabel)
    _grid_lines(ax, x, y)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_fields(x, y, T=None, psi=None, u=None, v=None, title="", path=None,
                quiver_step=3, ax=None):
    own = ax is None
    if own:
        fig = new_fig((8, 7)); ax = fig.add_subplot()
    else:
        fig = ax.figure
    if T is not None:
        m = ax.contourf(x, y, T, levels=30, cmap="inferno")
        if own:
            fig.colorbar(m, ax=ax, label="T")
    if psi is not None and np.ptp(psi) > 0:
        ax.contour(x, y, psi, levels=20, colors="white", linewidths=0.7)
    if u is not None and v is not None and (np.any(u) or np.any(v)):
        s = quiver_step
        ax.quiver(x[::s, ::s], y[::s, ::s], u[::s, ::s], v[::s, ::s], color="cyan",
                  scale_units="xy", angles="xy", width=0.002)
    ax.plot(x[:, 0], y[:, 0], "k", lw=1)
    ax.plot(x[:, -1], y[:, -1], "k", lw=1)
    ax.set_aspect("equal")
    ax.set_title(title)
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
    return fig


def plot_history(history, labels, path=None, title="Convergence"):
    history = np.asarray(history)
    fig = new_fig((7, 4)); ax = fig.add_subplot()
    for k, lab in labels.items():
        ax.semilogy(history[:, 0], np.abs(history[:, k]) + 1e-300, label=lab)
    ax.set_xlabel("iteration")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def animate(zones, path, var="T", fps=10, quiver=True, max_frames=None):
    """Animate the Tecplot zones of a rezsurf.dat file (list from io.read_tecplot)."""
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    if max_frames and len(zones) > max_frames:
        idx = np.linspace(0, len(zones) - 1, max_frames).astype(int)
        zones = [zones[i] for i in idx]
    z0 = zones[0]
    vals = np.concatenate([z[var].ravel() for z in zones])
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        vmax = vmin + 1e-12

    fig = new_fig((7, 6)); ax = fig.add_subplot()

    def draw(k):
        ax.clear()
        z = zones[k]
        ax.contourf(z["X"], z["Y"], z[var], levels=np.linspace(vmin, vmax, 31), cmap="inferno")
        if quiver and "U" in z and (np.any(z["U"]) or np.any(z["V"])):
            s = 3
            ax.quiver(z["X"][::s, ::s], z["Y"][::s, ::s], z["U"][::s, ::s], z["V"][::s, ::s],
                      color="cyan", width=0.002)
        ax.set_aspect("equal")
        ax.set_title(f"{var}, frame {k + 1}/{len(zones)}")

    anim = FuncAnimation(fig, draw, frames=len(zones))
    if str(path).lower().endswith(".mp4"):
        anim.save(path, writer=FFMpegWriter(fps=fps))
    else:
        anim.save(path, writer=PillowWriter(fps=fps))
