"""Command line interface.

    python -m ringflow boundary  --bottom bottom_border.dat --top top_border.dat
    python -m ringflow mesh      --boundary rez_param.dat
    python -m ringflow convect   --preset ring | ring_adi | curved
    python -m ringflow pipeline  --bottom ... --top ...     (all three in a row)
    python -m ringflow plot      rezsurf.dat  [--zone -1] [--var T]
    python -m ringflow animate   rezsurf.dat  --out anim.gif
    python -m ringflow to-vtk    rezsurf.dat  [--zone -1]
    python -m ringflow quality   --ring | --grid xxx.dat yyy.dat   (criteria Q1-Q4)
    python -m ringflow report    (runs everything -> one report.html)

Every command writes into --out (default: current folder).  Run any command
with --help for all options.
"""

import argparse
from pathlib import Path

import numpy as np


def _add_flow_args(p):
    p.add_argument("--preset", default="ring", choices=["ring", "ring_adi", "curved"])
    p.add_argument("--mesh-x", help="grid x file (curved preset), default xxx.dat")
    p.add_argument("--mesh-y", help="grid y file (curved preset), default yyy.dat")
    p.add_argument("--max-steps", type=int, help="time steps limit (Fortran m)")
    p.add_argument("--max-psi-iter", type=int, help="psi iterations per step (Fortran mp)")
    p.add_argument("--tau", type=float, help="time step (Fortran t)")
    p.add_argument("--beta", type=float, help="buoyancy number (Fortran bt)")
    p.add_argument("--viscosity", type=float, help="diffusion coefficient (Fortran tmx)")
    p.add_argument("--eps", type=float, help="convergence tolerance")
    p.add_argument("--snapshot-every", type=int,
                   help="write a Tecplot zone every N steps (0 = never). "
                        "Default 5 as in the original -> very big files on long runs")
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--fix-metric-stencil", action="store_true",
                   help="use the corrected d/dxi stencil (see README, 'known issues')")
    p.add_argument("--plot", action="store_true", help="save PNG pictures of the result")
    p.add_argument("--vtk", action="store_true", help="also write result.vtk for ParaView")


def _flow_cfg(a):
    from .convection import preset
    over = {}
    for name, attr in [("mesh_x", "mesh_x"), ("mesh_y", "mesh_y"), ("max_steps", "max_steps"),
                       ("max_psi_iter", "max_psi_iter"), ("tau", "tau"), ("beta", "beta"),
                       ("viscosity", "viscosity"), ("eps", "eps"),
                       ("snapshot_every", "snapshot_every")]:
        val = getattr(a, name, None)
        if val is not None:
            over[attr] = val
    over["print_every"] = a.print_every
    if a.fix_metric_stencil:
        over["fix_metric_stencil"] = True
    return preset(a.preset, **over)


def _flow_outputs(res, out, a):
    from . import io
    if a.plot:
        from .plotting import plot_fields, plot_history
        plot_fields(res.x, res.y, res.T, res.psi, res.u, res.v,
                    title=f"{res.config.preset}: T (colour), psi (lines), velocity",
                    path=out / "fields.png")
        if len(res.history):
            plot_history(res.history, {2: "max|dw|", 3: "max|dT|", 4: "psi residual"},
                         path=out / "convergence.png")
        print(f"pictures: {out/'fields.png'}, {out/'convergence.png'}")
    if a.vtk:
        io.write_vtk(out / "result.vtk", res.x, res.y,
                     dict(T=res.T, psi=res.psi, w=res.w, u=res.u, v=res.v))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ringflow", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("boundary", help="place n1 points on the two contours (interp.for)")
    p.add_argument("--bottom", required=True, help="outer contour, e.g. bottom_border.dat")
    p.add_argument("--top", required=True, help="inner contour, e.g. top_border.dat")
    p.add_argument("--n1", type=int, default=51)
    p.add_argument("--out", default=".")
    p.add_argument("--plot", action="store_true")

    p = sub.add_parser("mesh", help="generate the curvilinear mesh (Setka_ED2_II.for)")
    p.add_argument("--boundary", default="rez_param.dat")
    p.add_argument("--n1", type=int, default=51)
    p.add_argument("--n2", type=int, default=26)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=1e-5)
    p.add_argument("--fix-metric-stencil", action="store_true")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--out", default=".")
    p.add_argument("--plot", action="store_true")

    p = sub.add_parser("convect", help="flow + heat transfer solver (Dvizh_psi_om_Kolc*.for)")
    _add_flow_args(p)
    p.add_argument("--out", default=".")

    p = sub.add_parser("pipeline", help="boundary -> mesh -> convect (curved domain)")
    p.add_argument("--bottom", required=True)
    p.add_argument("--top", required=True)
    p.add_argument("--mesh-iter", type=int, default=5000)
    _add_flow_args(p)
    p.add_argument("--out", default="pipeline_out")

    p = sub.add_parser("plot", help="picture of one zone of a Tecplot file")
    p.add_argument("file")
    p.add_argument("--zone", type=int, default=-1)
    p.add_argument("--var", default=None, help="variable for the colour (default T or last)")
    p.add_argument("--out", default=None)

    p = sub.add_parser("animate", help="GIF/MP4 from the zones of a Tecplot file")
    p.add_argument("file")
    p.add_argument("--var", default="T")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--out", default="animation.gif")

    p = sub.add_parser("quality", help="mesh quality criteria Q1-Q4: table + 2D/3D pictures")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ring", action="store_true", help="circular ring mesh 0.5 <= r <= 2")
    src.add_argument("--grid", nargs=2, metavar=("XXX", "YYY"), help="xxx.dat yyy.dat")
    src.add_argument("--tecplot", help="Tecplot file with X, Y (last zone is used)")
    p.add_argument("--n1", type=int, default=51)
    p.add_argument("--n2", type=int, default=26)
    p.add_argument("--out", default="quality")

    p = sub.add_parser("report", help="run everything and build report.html (for showing)")
    p.add_argument("--bottom", default="../Dvizh_zhid/physical_area/bottom_border.dat")
    p.add_argument("--top", default="../Dvizh_zhid/physical_area/top_border.dat")
    p.add_argument("--mesh-iter", type=int, default=5000)
    p.add_argument("--ring-steps", type=int, default=1500)
    p.add_argument("--curved-steps", type=int, default=600)
    p.add_argument("--out", default="results/report")

    p = sub.add_parser("to-vtk", help="convert a Tecplot zone to VTK (ParaView)")
    p.add_argument("file")
    p.add_argument("--zone", type=int, default=-1)
    p.add_argument("--out", default=None)

    a = ap.parse_args(argv)

    if a.cmd == "boundary":
        from . import boundary
        out = Path(a.out)
        outer, inner = boundary.run(a.bottom, a.top, out, n1=a.n1)
        print(f"outer: {outer.iterations} it., inner: {inner.iterations} it. "
              f"-> {out/'rez_param.dat'}")
        if a.plot:
            from . import io
            from .plotting import new_fig
            fig = new_fig((7, 6)); ax = fig.add_subplot()
            for f in (a.bottom, a.top):
                x, y = io.read_points(f)
                ax.plot(x, y, "-", color="0.6", lw=1)
            for r in (outer, inner):
                ax.plot(r.x, r.y, "o", ms=3)
            ax.set_aspect("equal")
            fig.savefig(out / "boundary.png", dpi=150)

    elif a.cmd == "mesh":
        from . import mesh
        cfg = mesh.MeshConfig(n1=a.n1, n2=a.n2, max_iter=a.max_iter, eps=a.eps, tau=a.tau,
                              fix_metric_stencil=a.fix_metric_stencil)
        out = Path(a.out)
        res = mesh.run(a.boundary, out, cfg, print_every=a.print_every)
        if a.plot:
            from .plotting import plot_mesh
            for k in ("Q1", "Q2", "Q3", "Q4"):
                b = res.best[k]
                plot_mesh(b["x"], b["y"], b["q"], qlabel=k,
                          title=f"best mesh by {k} (iteration {b['iteration']}, min {k}={b['value']:.4f})",
                          path=out / f"mesh_{k}.png")

    elif a.cmd == "convect":
        from .convection import solve
        out = Path(a.out)
        cfg = _flow_cfg(a)
        res = solve(cfg, out)
        _flow_outputs(res, out, a)

    elif a.cmd == "pipeline":
        from . import boundary, mesh
        from .convection import solve
        out = Path(a.out)
        print("== 1/3 boundary points ==")
        boundary.run(a.bottom, a.top, out / "1_boundary", n1=51, verbose=False)
        print("== 2/3 mesh ==")
        mres = mesh.run(out / "1_boundary" / "rez_param.dat", out / "2_mesh",
                        mesh.MeshConfig(max_iter=a.mesh_iter,
                                        fix_metric_stencil=a.fix_metric_stencil),
                        print_every=500)
        if a.plot:
            from .plotting import plot_mesh
            b = mres.best["Q1"]
            plot_mesh(b["x"], b["y"], b["q"], title="mesh used for the flow (best Q1)",
                      path=out / "2_mesh" / "mesh_Q1.png")
        print("== 3/3 flow ==")
        a.preset = "curved"
        cfg = _flow_cfg(a)
        cfg.mesh_x = str(out / "2_mesh" / "xxx.dat")
        cfg.mesh_y = str(out / "2_mesh" / "yyy.dat")
        res = solve(cfg, out / "3_flow")
        _flow_outputs(res, out / "3_flow", a)

    elif a.cmd in ("plot", "to-vtk"):
        from . import io
        zones = io.read_tecplot(a.file)
        if not zones:
            raise SystemExit("no complete zone found in " + a.file)
        z = zones[a.zone]
        print(f"{len(zones)} zone(s) in file, using zone {a.zone}; variables: {list(z)}")
        stem = Path(a.file).with_suffix("")
        if a.cmd == "plot":
            from .plotting import plot_fields, plot_mesh
            var = a.var or ("T" if "T" in z else list(z)[-1])
            path = a.out or f"{stem}_{var}.png"
            if "U" in z:
                plot_fields(z["X"], z["Y"], z[var], z.get("P"), z.get("U"), z.get("V"),
                            title=f"{Path(a.file).name}: {var}", path=path)
            else:
                plot_mesh(z["X"], z["Y"], z[var], qlabel=var, title=Path(a.file).name, path=path)
            print("saved", path)
        else:
            path = a.out or f"{stem}.vtk"
            io.write_vtk(path, z["X"], z["Y"], {k: v for k, v in z.items() if k not in ("X", "Y")})
            print("saved", path)

    elif a.cmd == "quality":
        from . import io, quality
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        if a.ring:
            x, y = quality.ring_mesh(a.n1, a.n2)
        elif a.grid:
            x = io.read_grid_array(a.grid[0], a.n1, a.n2)
            y = io.read_grid_array(a.grid[1], a.n1, a.n2)
        else:
            z = io.read_tecplot(a.tecplot)[-1]
            x, y = z["X"], z["Y"]
        fields, stats = quality.analyze(x, y)
        table = quality.stats_table(stats)
        (out / "quality.csv").write_text(table, encoding="utf-8")
        print(table)
        for k in fields:
            quality.plot_2d(x, y, fields[k], k, out / f"{k}_2d.png")
            quality.plot_3d(x, y, fields[k], k, out / f"{k}_3d.png")
        quality.plot_panel(x, y, fields, stats, out / "all_2d.png")
        quality.plot_panel(x, y, fields, stats, out / "all_3d.png", kind="3d")
        print("pictures in", out)

    elif a.cmd == "report":
        from .report import make_report
        make_report(a.out, a.bottom, a.top, a.mesh_iter, a.ring_steps, a.curved_steps)

    elif a.cmd == "animate":
        from . import io
        from .plotting import animate
        zones = io.read_tecplot(a.file)
        animate(zones, a.out, var=a.var, fps=a.fps, max_frames=a.max_frames)
        print(f"saved {a.out} ({min(len(zones), a.max_frames)} frames)")


if __name__ == "__main__":
    main()
