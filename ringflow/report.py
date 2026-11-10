"""One-click demonstration report (single self-contained HTML file).

    python -m ringflow report --out results/report

Runs every part of the project on the original data and collects the
pictures into  report.html  (images embedded, opens in any browser, can be
sent by e-mail / printed to PDF).
"""

import base64
import html
import time
from pathlib import Path

import numpy as np

from . import boundary, mesh, convection, io, quality
from .quality import CRITERIA


def _img(path, width="100%"):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    mime = "image/gif" if str(path).endswith(".gif") else "image/png"
    return f'<img src="data:{mime};base64,{data}" style="max-width:{width}">'


def _stats_html(stats):
    rows = "".join(
        f"<tr><td>{k}</td><td>{html.escape(CRITERIA[k]['ru'])}</td>"
        f"<td>{s['min']:.4f}</td><td>{s['mean']:.4f}</td><td>{s['max']:.4f}</td>"
        f"<td>{s['bad']}</td></tr>" for k, s in stats.items())
    return ("<table><tr><th></th><th>Критерий</th><th>min</th><th>среднее</th>"
            f"<th>max</th><th>вырожд. ячеек</th></tr>{rows}</table>")


CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#222;line-height:1.5}
h1{border-bottom:3px solid #2a6;padding-bottom:6px} h2{margin-top:40px;color:#264}
table{border-collapse:collapse;margin:10px 0} td,th{border:1px solid #bbb;padding:4px 10px;text-align:right}
td:nth-child(2),th:nth-child(2){text-align:left} .note{background:#f4f7f4;border-left:4px solid #2a6;padding:8px 14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px} code{background:#eee;padding:1px 4px}
"""


def make_report(out_dir, bottom, top, mesh_iter=5000, ring_steps=1500, curved_steps=600,
                verbose=True, log=None):
    out = Path(out_dir)
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    say = log or (print if verbose else (lambda *a, **k: None))
    t0 = time.time()
    parts = []

    # ------------------------------------------------------------------ 1
    say("[1/5] circular ring mesh and its quality criteria")
    xr, yr = quality.ring_mesh(51, 26)
    fr, sr = quality.analyze(xr, yr)
    from .plotting import plot_mesh
    plot_mesh(xr, yr, None, title="Расчетная сетка (кольцо), 51 x 26", path=img / "ring_mesh.png")
    quality.plot_panel(xr, yr, fr, sr, img / "ring_q_3d.png",
                       "Критерии качества сетки в кольце (поверхности W = Q(x, y))", kind="3d")
    parts.append(f"""
<h2>1. Критерии качества сетки (кольцо)</h2>
<p>Каждая ячейка сетки с вершинами 1-2-3-4 разбивается диагоналями на треугольники
123, 134, 124, 234 (рис. 1 диссертации), и по их площадям, углам и длинам сторон
вычисляются четыре критерия. Все критерии лежат в [0, 1]: 1 &mdash; идеальная
ячейка, &le; 0 &mdash; вырожденная (невыпуклая) ячейка.</p>
<ul>{''.join(f"<li><b>{k}</b> &mdash; {html.escape(v['ru'])} ({v['en']})</li>" for k, v in CRITERIA.items())}</ul>
<p class="note">Q2, Q3, Q4 &mdash; три характеристики из подглавы об оценке качества сеток:
ортогональность, невытянутость ячеек, локальная равномерность.
Q1 &mdash; проверка невырожденности по площадям треугольников.</p>
<div class="grid2">{_img(img / 'ring_mesh.png')}<div>{_stats_html(sr)}
<p>В кольцевой сетке все ячейки почти ортогональны (Q2 &asymp; 1), а размер ячеек растет
с радиусом, поэтому Q3 и Q4 меняются от внутренней границы к внешней
(ср. рис. 9&ndash;10).</p></div></div>
{_img(img / 'ring_q_3d.png')}""")

    # ------------------------------------------------------------------ 2
    say("[2/5] boundary points of the curved domain")
    bdir = out / "data" / "1_boundary"
    outer, inner = boundary.run(bottom, top, bdir, n1=51, verbose=False)
    from .plotting import new_fig
    fig = new_fig((7, 5)); ax = fig.add_subplot()
    for f, lab in ((bottom, "внешний контур (bottom_border.dat)"),
                   (top, "внутренний контур (top_border.dat)")):
        x, y = io.read_points(f)
        ax.plot(x, y, "-", lw=1, label=lab)
    for r in (outer, inner):
        ax.plot(r.x, r.y, "ko", ms=3)
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("Шаг 1: 51 точка на каждом контуре (interp.for)")
    fig.tight_layout(); fig.savefig(img / "boundary.png", dpi=130)

    # ------------------------------------------------------------------ 3
    say(f"[3/5] mesh generation in the curved domain ({mesh_iter} iterations)")
    mdir = out / "data" / "2_mesh"
    mres = mesh.run(bdir / "rez_param.dat", mdir, mesh.MeshConfig(max_iter=mesh_iter),
                    verbose=False)
    x0, y0 = mesh.initial_grid(*mesh.read_boundary(bdir / "rez_param.dat", 51), 26)
    f0, s0 = quality.analyze(x0, y0)
    b1 = mres.best["Q1"]
    fb, sb = quality.analyze(b1["x"], b1["y"])
    quality.plot_2d(x0, y0, f0["Q1"], "Q1", img / "mesh_initial.png",
                    title=f"Начальная сетка (прямые линии): min Q1 = {s0['Q1']['min']:.3f}")
    quality.plot_2d(b1["x"], b1["y"], fb["Q1"], "Q1", img / "mesh_best.png",
                    title=f"Построенная сетка (итерация {b1['iteration']}): min Q1 = {sb['Q1']['min']:.3f}")
    quality.plot_history(mres.history, mres.best, img / "mesh_history.png")
    quality.plot_panel(b1["x"], b1["y"], fb, sb, img / "mesh_q_2d.png",
                       "Критерии качества построенной сетки (2D)")
    quality.plot_panel(b1["x"], b1["y"], fb, sb, img / "mesh_q_3d.png",
                       "Критерии качества построенной сетки (W = Q(x, y))", kind="3d")
    for k in CRITERIA:
        b = mres.best[k]
        quality.plot_3d(b["x"], b["y"], b["q"], k, img / f"best_by_{k}.png",
                        title=f"Полученная сетка по критерию {k[1]} ({CRITERIA[k]['ru']}),"
                              f" итерация {b['iteration']}, min = {b['value']:.3f}")
    best_rows = "".join(
        f"<tr><td>{k}</td><td>{html.escape(CRITERIA[k]['ru'])}</td>"
        f"<td>{mres.best[k]['iteration']}</td><td>{mres.best[k]['value']:.4f}</td></tr>"
        for k in CRITERIA)
    parts.append(f"""
<h2>2. Криволинейная двусвязная область: построение сетки</h2>
<p><b>Шаг 1</b> (<code>interp.for</code>): на внешний и внутренний контуры
равномерно расставляется по 51 точке.</p>
{_img(img / 'boundary.png', '70%')}
<p><b>Шаг 2</b> (<code>Setka_ED2_II.for</code>): начальная сетка из отрезков
улучшается итерациями решения эллиптических уравнений (метод переменных
направлений). На каждой итерации вычисляются критерии, и для каждого критерия
запоминается сетка, у которой худшая ячейка лучше всего (max-min).</p>
<div class="grid2">{_img(img / 'mesh_initial.png')}{_img(img / 'mesh_best.png')}</div>
{_img(img / 'mesh_history.png')}
<p class="note">Итерации генератора не сходятся к одной сетке: примерно каждые 270 итераций
сетка на короткое время выворачивается (min Q1 &lt; 0), затем снова улучшается.
Поэтому программа сохраняет лучшую сетку по каждому критерию, а не последнюю.</p>
<table><tr><th></th><th>Критерий</th><th>лучшая итерация</th><th>min Q</th></tr>{best_rows}</table>
<h3>Критерии качества итоговой сетки</h3>
{_stats_html(sb)}
{_img(img / 'mesh_q_2d.png')}
{_img(img / 'mesh_q_3d.png')}
<h3>Лучшая сетка по каждому критерию</h3>
<div class="grid2">{''.join(_img(img / f'best_by_{k}.png') for k in CRITERIA)}</div>""")

    # ------------------------------------------------------------------ 4
    from .plotting import plot_fields, animate
    say(f"[4/5] free convection in the circular ring ({ring_steps} steps)")
    rcfg = convection.preset("ring", max_steps=ring_steps,
                             snapshot_every=max(1, ring_steps // 60), print_every=10 ** 9)
    rres = convection.solve(rcfg, out / "data" / "3_ring", verbose=False)
    plot_fields(rres.x, rres.y, rres.T, rres.psi, rres.u, rres.v,
                title=f"Кольцо: температура, линии тока, скорость (шаг {rres.steps})",
                path=img / "ring_flow.png")
    animate(io.read_tecplot(out / "data" / "3_ring" / "rezsurf.dat"), img / "ring_flow.gif",
            fps=8, max_frames=36)

    say(f"[5/5] free convection in the curved domain ({curved_steps} steps)")
    ccfg = convection.preset("curved", max_steps=curved_steps, viscosity=1e-3,
                             snapshot_every=max(1, curved_steps // 60), print_every=10 ** 9,
                             mesh_x=str(mdir / "xxx.dat"), mesh_y=str(mdir / "yyy.dat"))
    cres = convection.solve(ccfg, out / "data" / "4_curved", verbose=False)
    plot_fields(cres.x, cres.y, cres.T, cres.psi, cres.u, cres.v,
                title=f"Криволинейная область: T, линии тока, скорость (шаг {cres.steps})",
                path=img / "curved_flow.png", quiver_step=2)
    animate(io.read_tecplot(out / "data" / "4_curved" / "rezsurf.dat"), img / "curved_flow.gif",
            fps=8, max_frames=36)
    parts.append(f"""
<h2>3. Свободная конвекция вязкой жидкости</h2>
<p><b>Шаг 3</b> (<code>Dvizh_psi_om_Kolc_jav.for</code>): уравнения для функции тока
&psi;, вихря &omega; и температуры T в криволинейных координатах. На каждом шаге по
времени: явный шаг для вихря, решение уравнения Пуассона для &psi;, вычисление
скоростей и вихря на стенках, явный шаг для температуры. Цвет &mdash;
температура, белые линии &mdash; линии тока, стрелки &mdash; скорость.</p>
<div class="grid2">{_img(img / 'ring_flow.png')}{_img(img / 'ring_flow.gif')}</div>
<div class="grid2">{_img(img / 'curved_flow.png')}{_img(img / 'curved_flow.gif')}</div>
<p class="note">Для криволинейной области задан коэффициент вязкости 0.001: в исходной
программе строка <code>tmx=0.001</code> закомментирована (вязкость равна 0).</p>""")

    page = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>Комплекс программ: сетки, критерии качества, конвекция</title><style>{CSS}</style></head><body>
<h1>Комплекс программ для численного моделирования течений в двусвязных областях</h1>
<p>Отчет построен автоматически Python-версией программ (<code>python -m ringflow report</code>)
на исходных данных. Результаты Python-версии совпадают с исходными Fortran-программами.</p>
{''.join(parts)}
<p style="color:#888;margin-top:40px">Время расчета: {time.time() - t0:.0f} c.</p>
</body></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")
    say(f"report: {out / 'report.html'}  ({time.time() - t0:.0f} s)")
    return out / "report.html"
