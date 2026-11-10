"""Графический интерфейс (окно) для комплекса программ.

Запуск:  python -m ringflow.gui      (или RingFlow.exe)

Вкладки:
  1. Сетка и критерии качества  -- построение сетки в двусвязной области,
                                   критерии Q1..Q4 (2D / 3D), история итераций
  2. Течение и теплообмен       -- расчет свободной конвекции, параметры,
                                   картинки, анимация
  3. Отчет                      -- HTML-отчет со всеми результатами
"""

import os
import sys
import queue
import threading
import traceback
from pathlib import Path


def _prepare_environment():
    """Writable folders for the .exe (numba cache, results)."""
    home = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "RingFlow"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(home / "numba_cache"))
    try:
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


_prepare_environment()

import tkinter as tk                                     # noqa: E402
from tkinter import ttk, filedialog, messagebox          # noqa: E402

import numpy as np                                       # noqa: E402
import matplotlib                                        # noqa: E402
matplotlib.rcParams.update({"axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
                            "axes.edgecolor": "#9aa1a9", "axes.labelcolor": "#4b5563"})
from matplotlib.figure import Figure                     # noqa: E402
from matplotlib.backends.backend_tkagg import (          # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from . import boundary, mesh, convection, io, quality    # noqa: E402
from .quality import CRITERIA                             # noqa: E402

DATA = Path(__file__).parent / "data"
FROZEN = getattr(sys, "frozen", False)


def open_path(p):
    p = str(p)
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # noqa: S606
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", p])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", p])
    except Exception as e:  # pragma: no cover
        messagebox.showerror("Ошибка", f"Не удалось открыть {p}\n{e}")


def default_results_dir():
    if FROZEN:
        docs = Path.home() / "Documents"
        return (docs if docs.exists() else Path.home()) / "RingFlow_результаты"
    return Path.cwd() / "results"


class Stop(Exception):
    pass


# ===========================================================================
# Design tokens (one place for colours, fonts, spacing)
ACCENT = "#1f7a4d"        # primary actions, selected states
ACCENT_HOVER = "#17603c"
ACCENT_SOFT = "#e7f2ec"
BG = "#f3f4f6"            # window background
PANEL = "#ffffff"         # sidebar / cards
TEXT = "#1f2328"
MUTED = "#4b5563"         # secondary text (contrast >= 7:1 on white)
BORDER = "#d7dbe0"
ERROR = "#b42318"
PAD = 8                   # base spacing unit


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RingFlow - сетки, критерии качества, свободная конвекция")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1320, sw - 40), min(900, sh - 80)
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(min(1000, w), min(620, h))
        if sys.platform.startswith("win") and sh < 900:
            self.state("zoomed")
        self.configure(bg=BG)
        self.q = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()
        self.mesh_res = None
        self.flow_res = None
        self._run_buttons = []
        self._busy = False
        self._setup_style()

        # ---- header: product name + results folder ---------------------
        head = ttk.Frame(self, style="Bg.TFrame", padding=(2 * PAD, PAD, 2 * PAD, 0))
        head.pack(fill="x")
        ttk.Label(head, text="RingFlow", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="  сетки в двусвязных областях · критерии качества · свободная конвекция",
                  style="Sub.TLabel").pack(side="left", pady=(4, 0))
        ttk.Button(head, text="Описание", command=self._help).pack(side="right")

        folder = ttk.Frame(self, style="Bg.TFrame", padding=(2 * PAD, PAD, 2 * PAD, PAD))
        folder.pack(fill="x")
        ttk.Label(folder, text="Результаты сохраняются в", style="BgMuted.TLabel").pack(side="left")
        self.outdir = tk.StringVar(value=str(default_results_dir()))
        ttk.Entry(folder, textvariable=self.outdir).pack(side="left", fill="x", expand=True, padx=PAD)
        ttk.Button(folder, text="Изменить…", command=self._choose_outdir).pack(side="left")
        ttk.Button(folder, text="Открыть папку", command=self._open_outdir).pack(side="left", padx=(PAD, 0))

        logf = ttk.Frame(self, style="Bg.TFrame", padding=(2 * PAD, 0, 2 * PAD, PAD))
        logf.pack(side="bottom", fill="x")
        self.log = tk.Text(logf, height=3, font=self.mono, relief="flat", bg=PANEL, fg=MUTED,
                           highlightthickness=1, highlightbackground=BORDER, padx=PAD, pady=4)
        self.log.pack(fill="x")

        # ---- status bar: progress, message, the one Stop button ---------
        bar = ttk.Frame(self, style="Bg.TFrame", padding=(2 * PAD, PAD))
        bar.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(bar, length=260, mode="determinate",
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.pack(side="left")
        self.status = tk.StringVar(value="Готово к работе.")
        self.status_label = ttk.Label(bar, textvariable=self.status, style="Status.TLabel")
        self.status_label.pack(side="left", padx=2 * PAD)
        self.stop_btn = ttk.Button(bar, text="Стоп", style="Stop.TButton",
                                   command=self._request_stop, state="disabled")
        self.stop_btn.pack(side="right")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=2 * PAD)
        self._build_mesh_tab()
        self._build_flow_tab()
        self._build_report_tab()

        self.after(100, self._poll)
        self.after(300, self._show_ring_mesh)

    # ------------------------------------------------------------------ style
    def _setup_style(self):
        import tkinter.font as tkfont
        families = set(tkfont.families(self))
        base = next((f for f in ("Segoe UI", "DejaVu Sans", "Arial") if f in families), "TkDefaultFont")
        mono = next((f for f in ("Consolas", "DejaVu Sans Mono", "Courier New") if f in families), "TkFixedFont")
        self.mono = (mono, 9)
        self.base = base
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=base, size=10)

        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=PANEL, foreground=TEXT, bordercolor=BORDER,
                    focuscolor=ACCENT, font=(base, 10))
        s.configure("Bg.TFrame", background=BG)
        s.configure("Side.TFrame", background=PANEL)
        s.configure("Title.TLabel", background=BG, foreground=TEXT, font=(base, 16, "bold"))
        s.configure("Sub.TLabel", background=BG, foreground=MUTED, font=(base, 10))
        s.configure("BgMuted.TLabel", background=BG, foreground=MUTED)
        s.configure("Section.TLabel", background=PANEL, foreground=ACCENT, font=(base, 10, "bold"))
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=(base, 9))
        s.configure("Status.TLabel", background=BG, foreground=TEXT)
        s.configure("Ok.TLabel", background=BG, foreground=ACCENT, font=(base, 10, "bold"))
        s.configure("Err.TLabel", background=BG, foreground=ERROR, font=(base, 10, "bold"))
        s.configure("TButton", padding=(10, 4), background="#eef0f3", relief="flat")
        s.map("TButton", background=[("active", "#e2e5ea"), ("disabled", "#f3f4f6")],
              foreground=[("disabled", "#9aa1a9")])
        s.configure("Accent.TButton", background=ACCENT, foreground="white",
                    font=(base, 11, "bold"), padding=(12, 9), borderwidth=0)
        s.map("Accent.TButton", background=[("disabled", "#a9cbb8"), ("active", ACCENT_HOVER)],
              foreground=[("disabled", "white")])
        s.configure("Stop.TButton", foreground=ERROR, padding=(14, 4))
        s.map("Stop.TButton", foreground=[("disabled", "#c4c8cd")])
        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        s.configure("TNotebook.Tab", padding=(18, 8), background="#e4e7eb", foreground=MUTED,
                    font=(base, 10))
        s.map("TNotebook.Tab", background=[("selected", PANEL)],
              foreground=[("selected", ACCENT)], font=[("selected", (base, 10, "bold"))])
        s.configure("TRadiobutton", background=PANEL, indicatorcolor=PANEL)
        s.map("TRadiobutton", indicatorcolor=[("selected", ACCENT)])
        s.configure("TCheckbutton", background=PANEL, indicatorcolor=PANEL)
        s.map("TCheckbutton", indicatorcolor=[("selected", ACCENT)])
        s.configure("TEntry", fieldbackground="white", padding=(4, 2))
        s.configure("Treeview", rowheight=24, fieldbackground=PANEL, borderwidth=0)
        s.configure("Treeview.Heading", font=(base, 9, "bold"), background="#eef0f3",
                    foreground=MUTED, relief="flat")
        s.configure("Accent.Horizontal.TProgressbar", troughcolor="#e4e7eb", background=ACCENT,
                    bordercolor="#e4e7eb", lightcolor=ACCENT, darkcolor=ACCENT)

    # ---------------------------------------------------- layout helpers
    def _sidebar(self, tab):
        """Left control panel (scrolls on small screens) + main area."""
        outer = ttk.Frame(tab, style="Side.TFrame")
        outer.pack(side="left", fill="y")
        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        side = ttk.Frame(canvas, style="Side.TFrame", padding=(2 * PAD, 2 * PAD, 2 * PAD, 2 * PAD))
        canvas.create_window((0, 0), window=side, anchor="nw")
        canvas.pack(side="left", fill="y")

        def refit(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"), width=side.winfo_reqwidth())
            if side.winfo_reqheight() > canvas.winfo_height() > 1:
                if not sb.winfo_ismapped():
                    sb.pack(side="left", fill="y")
            elif sb.winfo_ismapped():
                sb.pack_forget()
        side.bind("<Configure>", refit)
        canvas.bind("<Configure>", refit)

        def wheel(ev):
            if sb.winfo_ismapped():
                step = -1 if (getattr(ev, "num", 0) == 4 or ev.delta > 0) else 1
                canvas.yview_scroll(step, "units")
        for w in (canvas, side):
            w.bind("<Enter>", lambda e: (canvas.bind_all("<MouseWheel>", wheel),
                                         canvas.bind_all("<Button-4>", wheel),
                                         canvas.bind_all("<Button-5>", wheel)))
            w.bind("<Leave>", lambda e: (canvas.unbind_all("<MouseWheel>"),
                                         canvas.unbind_all("<Button-4>"),
                                         canvas.unbind_all("<Button-5>")))
        sep = tk.Frame(tab, width=1, bg=BORDER)
        sep.pack(side="left", fill="y")
        main = ttk.Frame(tab, style="Side.TFrame", padding=(PAD, PAD))
        main.pack(side="left", fill="both", expand=True)
        return side, main

    def _section(self, parent, title, first=False):
        ttk.Label(parent, text=title, style="Section.TLabel").pack(
            anchor="w", pady=(0 if first else 2 * PAD, PAD // 2))
        f = ttk.Frame(parent, style="Side.TFrame")
        f.pack(fill="x")
        return f

    def _params(self, parent, rows):
        """Aligned label / entry grid.  rows: (label, StringVar) -> entries list."""
        parent.columnconfigure(0, weight=1)
        entries = []
        for r, (lab, var) in enumerate(rows):
            ttk.Label(parent, text=lab).grid(row=r, column=0, sticky="w", pady=1)
            e = ttk.Entry(parent, textvariable=var, width=9, justify="right")
            e.grid(row=r, column=1, sticky="e", pady=1, padx=(PAD, 0))
            entries.append(e)
        return entries

    def _primary(self, parent, text, command):
        b = ttk.Button(parent, text=text, style="Accent.TButton", command=command)
        b.pack(fill="x", pady=(2 * PAD, 0))
        self._run_buttons.append(b)
        return b

    def _secondary(self, parent, text, command):
        b = ttk.Button(parent, text=text, command=command)
        b.pack(fill="x", pady=(PAD // 2, 0))
        self._run_buttons.append(b)
        return b

    def _figure(self, parent):
        fig = Figure(figsize=(8, 6), dpi=100, facecolor=PANEL)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        bar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        bar.update()
        bar.configure(background=PANEL)
        for child in bar.winfo_children():
            try:
                child.configure(background=PANEL)
            except tk.TclError:
                pass
        bar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas

    # ------------------------------------------------------------------ utils
    def out(self, sub=""):
        p = Path(self.outdir.get()) / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _choose_outdir(self):
        d = filedialog.askdirectory(initialdir=self.outdir.get())
        if d:
            self.outdir.set(d)

    def _open_outdir(self):
        open_path(self.out())

    def write(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _num(self, var, kind=float, name=""):
        try:
            return kind(var.get().replace(",", ".").strip())
        except ValueError:
            raise ValueError(f"Неверное значение параметра «{name}»: {var.get()!r}")

    def _set_status(self, text, kind="normal"):
        self.status.set(text)
        self.status_label.configure(style={"ok": "Ok.TLabel", "error": "Err.TLabel"}.get(kind, "Status.TLabel"))

    def _set_busy(self, busy):
        self._busy = busy
        for b in self._run_buttons:
            b.configure(state="disabled" if busy else "normal")
        self.stop_btn.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.configure(mode="indeterminate", value=0)
            self.progress.start(12)          # loading state until real progress arrives
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")

    def _request_stop(self):
        self.stop_flag.set()
        self._set_status("Останавливаю…")

    def _flash_done(self, text):
        """Success feedback: green check, fades back to normal text."""
        self.progress.configure(value=100)
        self._set_status("✓ " + text, "ok")

        def fade():
            if self.status.get() == "✓ " + text:
                self._set_status(text)
        self.after(4000, fade)

    def run_bg(self, func, done):
        """Run func() in a thread; done(result) is called in the GUI thread."""
        if self.worker and self.worker.is_alive():
            return
        self.stop_flag.clear()
        self._set_busy(True)

        def target():
            try:
                res = func()
                self.q.put(("done", done, res))
            except Stop:
                self.q.put(("stopped",))
            except Exception as e:
                self.q.put(("error", f"{e}\n\n{traceback.format_exc()}"))
        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def _poll(self):
        """Handle messages from the worker thread.

        Only what is already queued is handled (never loop while the worker
        keeps producing), and of several live frames only the newest is
        drawn, so the window stays responsive during long runs.
        """
        items = []
        try:
            for _ in range(self.q.qsize() + 1):
                items.append(self.q.get_nowait())
        except queue.Empty:
            pass
        last_call = max((i for i, it in enumerate(items) if it[0] == "call"), default=-1)
        for idx, item in enumerate(items):
            kind = item[0]
            if kind == "msg":
                self.write(item[1]); self._set_status(item[1])
            elif kind == "status":
                self._set_status(item[1])
            elif kind == "progress":
                if str(self.progress["mode"]) == "indeterminate":
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                self.progress["value"] = item[1]
            elif kind == "call":
                if idx == last_call:
                    item[1](*item[2:])
            elif kind == "done":
                self._set_busy(False)
                item[1](item[2])
            elif kind == "stopped":
                self._set_busy(False)
                self.write("Расчет остановлен.")
                self._set_status("Расчет остановлен.")
            elif kind == "error":
                self._set_busy(False)
                first = item[1].split("\n\n")[0]
                self._set_status("Ошибка: " + first, "error")
                self.write(item[1])
                messagebox.showerror("Ошибка", first)
        self.after(100, self._poll)

    def _help(self):
        for cand in (Path(sys.executable).parent / "Описание.html",
                     Path(__file__).resolve().parent.parent / "README.md"):
            if cand.exists():
                open_path(cand)
                return
        messagebox.showinfo("Описание", __doc__)

    # ================================================================== TAB 1
    def _build_mesh_tab(self):
        tab = ttk.Frame(self.nb, style="Side.TFrame")
        self.nb.add(tab, text="1. Сетка и критерии качества")
        side, main = self._sidebar(tab)

        g = self._section(side, "Граница области", first=True)
        self.f_outer = tk.StringVar(value=str(DATA / "bottom_border.dat"))
        self.f_inner = tk.StringVar(value=str(DATA / "top_border.dat"))
        for lab, var in (("Внешний контур", self.f_outer), ("Внутренний контур", self.f_inner)):
            row = ttk.Frame(g, style="Side.TFrame"); row.pack(fill="x", pady=2)
            ttk.Label(row, text=lab, width=17).pack(side="left")
            name = tk.StringVar(value=Path(var.get()).name)
            var.trace_add("write", lambda *a, v=var, n=name: n.set(Path(v.get()).name))
            ttk.Label(row, textvariable=name, style="Muted.TLabel", width=17).pack(side="left")
            ttk.Button(row, text="Файл…", command=lambda v=var: self._pick_file(v)).pack(side="right")

        p = self._section(side, "Параметры построения сетки")
        self.m_n1 = tk.StringVar(value="51")
        self.m_n2 = tk.StringVar(value="26")
        self.m_it = tk.StringVar(value="5000")
        self.m_tau = tk.StringVar(value="1e-5")
        self.m_eps = tk.StringVar(value="1e-4")
        self._params(p, (("Узлов вдоль контура, n1", self.m_n1),
                         ("Узлов поперек, n2", self.m_n2),
                         ("Максимум итераций", self.m_it),
                         ("Шаг итераций, τ", self.m_tau),
                         ("Точность, ε", self.m_eps)))
        self.m_fix = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Исправленный шаблон производной", variable=self.m_fix).pack(
            anchor="w", pady=(PAD // 2, 0))

        self._primary(side, "Построить сетку", self.build_mesh)
        row = ttk.Frame(side, style="Side.TFrame"); row.pack(fill="x", pady=(PAD // 2, 0))
        row.columnconfigure((0, 1), weight=1, uniform="b")
        for c, (t, cmd) in enumerate((("Кольцо (рис. 6)", self._show_ring_mesh),
                                      ("Открыть сетку…", self._load_grid))):
            b = ttk.Button(row, text=t, command=cmd)
            b.grid(row=0, column=c, sticky="ew", padx=(0 if c == 0 else PAD // 2, 0))
            self._run_buttons.append(b)

        v = self._section(side, "Что показать")
        self.crit = tk.StringVar(value="Q1")
        short = {"Q1": "Невырожденность", "Q2": "Ортогональность",
                 "Q3": "Невытянутость ячеек", "Q4": "Локальная равномерность"}
        for k in CRITERIA:
            ttk.Radiobutton(v, text=f"{k}   {short[k]}", value=k, variable=self.crit,
                            command=self.redraw_mesh).pack(anchor="w")
        ttk.Radiobutton(v, text="Сетка без раскраски", value="mesh", variable=self.crit,
                        command=self.redraw_mesh).pack(anchor="w")
        ttk.Radiobutton(v, text="История итераций", value="hist", variable=self.crit,
                        command=self.redraw_mesh).pack(anchor="w")
        self.view3d = tk.BooleanVar(value=False)
        ttk.Checkbutton(v, text="Объемный вид W = Q(x, y), как рис. 9-10", variable=self.view3d,
                        command=self.redraw_mesh).pack(anchor="w", pady=(PAD // 2, PAD // 2))
        row = ttk.Frame(v, style="Side.TFrame"); row.pack(fill="x")
        ttk.Label(row, text="Сетка").pack(side="left")
        self.which = tk.StringVar()
        self.which_box = ttk.Combobox(row, textvariable=self.which, state="readonly", width=22,
                                      values=[f"лучшая по {k}" for k in CRITERIA] + ["последняя итерация"])
        self.which_box.current(0)
        self.which_box.pack(side="right")
        self.which_box.bind("<<ComboboxSelected>>", lambda e: self.redraw_mesh())

        cols = ("name", "min", "mean", "max", "bad")
        self.table = ttk.Treeview(main, columns=cols, height=4, show="headings")
        for c, t, w, anchor in zip(cols, ("Критерий", "min", "среднее", "max", "вырожденных ячеек"),
                                   (260, 90, 90, 90, 170), ("w", "e", "e", "e", "e")):
            self.table.heading(c, text=t, anchor=anchor)
            self.table.column(c, width=w, anchor=anchor, stretch=(c == "name"))
        self.table.pack(side="bottom", fill="x", pady=(PAD, 0))
        self.mesh_fig, self.mesh_canvas = self._figure(main)

        self.grid_xy = None          # currently displayed grid (x, y)
        self.grid_title = ""

    def _pick_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Данные", "*.dat *.txt"), ("Все", "*.*")])
        if f:
            var.set(f)

    def _show_ring_mesh(self):
        self.mesh_res = None
        self.grid_xy = quality.ring_mesh(51, 26)
        self.grid_title = "Круговое кольцо 51 x 26 (рис. 6)"
        self.redraw_mesh()

    def _load_grid(self):
        fx = filedialog.askopenfilename(title="Файл X (xxx.dat)", filetypes=[("dat", "*.dat"), ("Все", "*.*")])
        if not fx:
            return
        fy = filedialog.askopenfilename(title="Файл Y (yyy.dat)", filetypes=[("dat", "*.dat"), ("Все", "*.*")])
        if not fy:
            return
        try:
            n1, n2 = int(self.m_n1.get()), int(self.m_n2.get())
            self.grid_xy = (io.read_grid_array(fx, n1, n2), io.read_grid_array(fy, n1, n2))
        except Exception as e:
            messagebox.showerror("Ошибка", str(e)); return
        self.mesh_res = None
        self.grid_title = Path(fx).name
        self.redraw_mesh()

    def build_mesh(self):
        try:
            cfg = mesh.MeshConfig(n1=self._num(self.m_n1, int, "n1"), n2=self._num(self.m_n2, int, "n2"),
                                  max_iter=self._num(self.m_it, int, "итерации"),
                                  tau=self._num(self.m_tau, float, "τ"),
                                  eps=self._num(self.m_eps, float, "ε"),
                                  fix_metric_stencil=self.m_fix.get())
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e)); return
        outer, inner = self.f_outer.get(), self.f_inner.get()
        bdir, mdir = self.out("1_граница"), self.out("2_сетка")
        self.write(f"Построение сетки {cfg.n1} x {cfg.n2}, до {cfg.max_iter} итераций ...")
        self.progress["value"] = 0

        def job():
            self.q.put(("status", "Расстановка точек на контурах (interp)..."))
            boundary.run(outer, inner, bdir, n1=cfg.n1, verbose=False)
            self.q.put(("status", "Подготовка вычислительного ядра..."))

            def cb(n, hor, mins):
                if self.stop_flag.is_set():
                    return True
                if n % 50 == 0 or n == 1:
                    self.q.put(("progress", 100.0 * n / cfg.max_iter))
                    self.q.put(("status", f"Итерация {n} из {cfg.max_iter}:  сдвиг узлов {hor:.2e},"
                                          f"  min Q1 = {mins[0]:.3f}"))
                return False
            return mesh.run(bdir / "rez_param.dat", mdir, cfg, verbose=False, callback=cb)

        def done(res):
            self.mesh_res = res
            self.grid_title = "Построенная сетка"
            b = res.best
            self.write("Готово. Лучшие сетки: " + ", ".join(
                f"{k}: итерация {b[k]['iteration']}, min={b[k]['value']:.3f}" for k in b))
            self.write(f"Файлы сетки сохранены в {mdir} (xxx.dat, yyy.dat, surf1..4.dat)")
            self._flash_done("Сетка построена")
            self.redraw_mesh()
        self.run_bg(job, done)

    def _current_grid(self):
        if self.mesh_res is None:
            return self.grid_xy
        w = self.which_box.current()
        if w < 4:
            b = self.mesh_res.best[f"Q{w + 1}"]
            return b["x"], b["y"]
        return self.mesh_res.x, self.mesh_res.y

    def redraw_mesh(self):
        fig = self.mesh_fig
        fig.clear()
        sel = self.crit.get()
        if sel == "hist":
            ax = fig.add_subplot(111)
            if self.mesh_res is None:
                ax.text(0.5, 0.5, "Сначала постройте сетку", ha="center")
            else:
                h = self.mesh_res.history
                for c, k in enumerate(CRITERIA, start=2):
                    line, = ax.plot(h[:, 0], h[:, c], label=f"{k}: {CRITERIA[k]['ru']}")
                    b = self.mesh_res.best[k]
                    ax.plot([b["iteration"]], [b["value"]], "o", color=line.get_color())
                ax.set_ylim(-0.6, 1); ax.axhline(0, color="k", lw=0.6)
                ax.set_xlabel("итерация"); ax.set_ylabel("min Q по сетке")
                ax.set_title("Качество сетки по итерациям (точки - сохраненная лучшая сетка)")
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
            self.mesh_canvas.draw()
            return
        if self.grid_xy is None and self.mesh_res is None:
            return
        x, y = self._current_grid()
        fields, stats = quality.analyze(x, y)
        self.table.delete(*self.table.get_children())
        for k, s in stats.items():
            self.table.insert("", "end", values=(f"{k}   {CRITERIA[k]['short']}", f"{s['min']:.4f}",
                                                        f"{s['mean']:.4f}", f"{s['max']:.4f}", s["bad"]))
        title = self.grid_title
        if self.mesh_res is not None:
            title += f" ({self.which_box.get()})"
        if sel == "mesh":
            ax = fig.add_subplot(111)
            for i in range(x.shape[0]):
                ax.plot(x[i, :], y[i, :], "k", lw=0.4)
            for j in range(x.shape[1]):
                ax.plot(x[:, j], y[:, j], "k", lw=0.4)
            ax.set_aspect("equal"); ax.set_title(title)
        elif self.view3d.get():
            ax = fig.add_subplot(111, projection="3d")
            quality.plot_3d(x, y, fields[sel], sel, ax=ax,
                            title=f"{title}\n{sel}: {CRITERIA[sel]['ru']}  (min = {stats[sel]['min']:.3f})")
            ax.set_box_aspect(None, zoom=1.0)
            fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.88)
            self.mesh_canvas.draw()
            return
        else:
            ax = fig.add_subplot(111)
            quality.plot_2d(x, y, fields[sel], sel, ax=ax,
                            title=f"{title}\n{sel}: {CRITERIA[sel]['ru']}  (min = {stats[sel]['min']:.3f})")
        fig.tight_layout()
        self.mesh_canvas.draw()

    # ================================================================== TAB 2
    def _build_flow_tab(self):
        tab = ttk.Frame(self.nb, style="Side.TFrame")
        self.nb.add(tab, text="2. Течение и теплообмен")
        side, main = self._sidebar(tab)

        d = self._section(side, "Расчетная область", first=True)
        self.domain = tk.StringVar(value="ring")
        ttk.Radiobutton(d, text="Круговое кольцо, 0.5 ≤ r ≤ 2", value="ring",
                        variable=self.domain, command=self._domain_changed).pack(anchor="w", pady=1)
        ttk.Radiobutton(d, text="Криволинейная область (сетка вкладки 1)", value="curved",
                        variable=self.domain, command=self._domain_changed).pack(anchor="w", pady=1)
        ttk.Label(d, text="Внутренняя стенка горячая (T = 1), внешняя холодная (T = 0).",
                  style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=(2, 0))

        p = self._section(side, "Физические параметры")
        self.fp = {}
        spec_phys = [("beta", "Число плавучести, β", "10"),
                     ("nu", "Вязкость, теплопроводность, ν", "0.001")]
        spec_num = [("steps", "Число шагов по времени", "1500"),
                    ("tau", "Шаг по времени, τ", "0.001"),
                    ("eps", "Точность остановки", "1e-6"),
                    ("mp", "Итераций для ψ на шаге", "1000"),
                    ("n1", "Узлов по окружности", "101"),
                    ("n2", "Узлов по радиусу", "51"),
                    ("every", "Кадр анимации каждые N шагов", "25")]
        for key, _, val in spec_phys + spec_num:
            self.fp[key] = tk.StringVar(value=val)
        self._params(p, [(lab, self.fp[k]) for k, lab, _ in spec_phys])
        n = self._section(side, "Численные параметры")
        entries = self._params(n, [(lab, self.fp[k]) for k, lab, _ in spec_num])
        self.entry_n1, self.entry_n2 = entries[4], entries[5]

        self._primary(side, "Рассчитать", self.run_flow)
        row = ttk.Frame(side, style="Side.TFrame"); row.pack(fill="x", pady=(PAD // 2, 0))
        row.columnconfigure((0, 1), weight=1, uniform="b")
        for c, (t, cmd) in enumerate((("Анимация GIF", self.make_gif),
                                      ("Открыть Tecplot…", self.open_tecplot))):
            b = ttk.Button(row, text=t, command=cmd)
            b.grid(row=0, column=c, sticky="ew", padx=(0 if c == 0 else PAD // 2, 0))
            self._run_buttons.append(b)

        s = self._section(side, "Что показать")
        self.show = tk.StringVar(value="T")
        for val, lab in (("T", "Температура и линии тока"), ("psi", "Функция тока ψ"),
                         ("w", "Вихрь ω"), ("speed", "Модуль скорости |V|")):
            ttk.Radiobutton(s, text=lab, value=val, variable=self.show,
                            command=self.redraw_flow).pack(anchor="w")
        self.arrows = tk.BooleanVar(value=True)
        ttk.Checkbutton(s, text="Стрелки скорости", variable=self.arrows,
                        command=self.redraw_flow).pack(anchor="w", pady=(PAD // 2, 0))

        self.flow_fig, self.flow_canvas = self._figure(main)
        self.flow_fields = None
        self._domain_changed()
        self.after(200, self.redraw_flow)

    def _domain_changed(self):
        st = "normal" if self.domain.get() == "ring" else "disabled"
        for k in ("n1", "n2"):
            getattr(self, f"entry_{k}").configure(state=st)

    def run_flow(self):
        try:
            v = {k: self.fp[k] for k in self.fp}
            steps = self._num(v["steps"], int, "число шагов")
            every = self._num(v["every"], int, "кадр каждые N шагов")
            common = dict(max_steps=steps, tau=self._num(v["tau"], float, "τ"),
                          beta=self._num(v["beta"], float, "β"),
                          viscosity=self._num(v["nu"], float, "ν"),
                          eps=self._num(v["eps"], float, "точность"),
                          max_psi_iter=self._num(v["mp"], int, "итерации ψ"),
                          snapshot_every=every, print_every=10 ** 9)
            if self.domain.get() == "ring":
                cfg = convection.preset("ring", n1=self._num(v["n1"], int, "n1"),
                                        n2=self._num(v["n2"], int, "n2"), **common)
                out = self.out("3_течение_кольцо")
            else:
                mdir = Path(self.outdir.get()) / "2_сетка"
                if (mdir / "xxx.dat").exists() and (mdir / "mesh_result.npz").exists():
                    mx, my = mdir / "xxx.dat", mdir / "yyy.dat"
                    gn1, gn2 = np.load(mdir / "mesh_result.npz")["x"].shape
                    self.write(f"Используется сетка {gn1} x {gn2} из {mdir}")
                else:
                    mx, my = DATA / "default_xxx.dat", DATA / "default_yyy.dat"
                    gn1, gn2 = 51, 26
                    self.write("Сетка еще не построена - используется исходная сетка диссертации.")
                cfg = convection.preset("curved", mesh_x=str(mx), mesh_y=str(my),
                                        n1=gn1, n2=gn2, **common)
                out = self.out("3_течение_кривол")
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e)); return

        self.progress["value"] = 0
        self.write(f"Расчет: {cfg.preset}, {cfg.n1} x {cfg.n2}, до {steps} шагов, τ={cfg.tau},"
                   f" β={cfg.beta}, ν={cfg.viscosity}")
        self._set_status("Подготовка вычислительного ядра (при первом запуске до 1 мин)…")
        live_every = max(1, steps // 40)
        geo_holder = {}

        def job():
            def cb(n, f):
                if self.stop_flag.is_set():
                    return True
                if n % 10 == 0 or n == 1:
                    self.q.put(("progress", 100.0 * n / steps))
                    self.q.put(("status", f"Шаг {n} из {steps}:  max|ΔT| = {f['hortt']:.2e},"
                                          f"  итераций ψ = {f['k']}"))
                if n % live_every == 0 and "g" in geo_holder:
                    snap = {k: np.array(f[k]) for k in ("u", "v", "w", "psi", "T")}
                    self.q.put(("call", self._set_flow, geo_holder["g"], snap, f"шаг {n}"))
                return False
            if cfg.geometry == "ring":
                g = convection.ring_geometry(cfg.n1, cfg.n2)
            else:
                g = convection.file_geometry(io.read_grid_array(cfg.mesh_x, cfg.n1, cfg.n2),
                                             io.read_grid_array(cfg.mesh_y, cfg.n1, cfg.n2))
            geo_holder["g"] = g
            res = convection.solve(cfg, out, verbose=False, geometry=g, callback=cb)
            return res, out

        def done(r):
            res, out = r
            self.flow_res = res
            self._set_flow(res.geometry, dict(u=res.u, v=res.v, w=res.w, psi=res.psi, T=res.T),
                           f"шаг {res.steps}")
            msg = "сошелся" if res.converged else "достигнуто заданное число шагов"
            self.write(f"Готово: {res.steps} шагов ({msg}). Результаты: {out}")
            self._flash_done(f"Расчет завершен: {res.steps} шагов")
            try:
                io.write_vtk(out / "result.vtk", res.x, res.y,
                             dict(T=res.T, psi=res.psi, w=res.w, u=res.u, v=res.v))
            except Exception:
                pass
        self.run_bg(job, done)

    def _set_flow(self, g, f, label):
        self.flow_fields = (g.x[1:, 1:], g.y[1:, 1:], f, label)
        self.redraw_flow()

    def redraw_flow(self):
        fig = self.flow_fig
        fig.clear()
        if self.flow_fields is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Задайте параметры и нажмите «Рассчитать»", ha="center", va="center")
            ax.axis("off")
            self.flow_canvas.draw()
            return
        x, y, f, label = self.flow_fields
        ax = fig.add_subplot(111)
        sel = self.show.get()
        if sel == "T":
            m = ax.contourf(x, y, f["T"], levels=30, cmap="inferno")
            if np.ptp(f["psi"]) > 0:
                ax.contour(x, y, f["psi"], levels=20, colors="white", linewidths=0.6)
            name = "Температура T, белые линии - линии тока"
        else:
            data = {"psi": f["psi"], "w": f["w"], "speed": np.hypot(f["u"], f["v"])}[sel]
            m = ax.contourf(x, y, data, levels=30, cmap="viridis" if sel != "w" else "RdBu_r")
            name = {"psi": "Функция тока ψ", "w": "Вихрь ω", "speed": "Модуль скорости |V|"}[sel]
        fig.colorbar(m, ax=ax)
        if self.arrows.get() and (np.any(f["u"]) or np.any(f["v"])):
            s = 3 if x.shape[0] > 60 else 2
            ax.quiver(x[::s, ::s], y[::s, ::s], f["u"][::s, ::s], f["v"][::s, ::s],
                      color="cyan", width=0.002)
        ax.plot(x[:, 0], y[:, 0], "k", lw=1)
        ax.plot(x[:, -1], y[:, -1], "k", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"{name} ({label})")
        fig.tight_layout()
        self.flow_canvas.draw()

    def make_gif(self):
        sub = "3_течение_кольцо" if self.domain.get() == "ring" else "3_течение_кривол"
        src = Path(self.outdir.get()) / sub / "rezsurf.dat"
        if not src.exists():
            messagebox.showinfo("Анимация", "Сначала выполните расчет на этой вкладке."); return
        dst = src.parent / "анимация_T.gif"
        self.write(f"Создание анимации из {src} ...")

        def job():
            from .plotting import animate
            animate(io.read_tecplot(src), dst, var="T", fps=8, max_frames=80)
            return dst

        def done(p):
            self.write(f"Анимация сохранена: {p}")
            self._flash_done("Анимация готова")
            open_path(p)
        self.run_bg(job, done)

    def open_tecplot(self):
        f = filedialog.askopenfilename(filetypes=[("Tecplot", "*.dat"), ("Все", "*.*")])
        if not f:
            return
        try:
            zones = io.read_tecplot(f)
            z = zones[-1]
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}"); return
        if not {"X", "Y"} <= set(z):
            messagebox.showerror("Ошибка", "В файле нет переменных X, Y"); return
        if {"U", "V", "T"} <= set(z):
            class G:  # minimal geometry holder
                pass
            g = G(); g.x = np.pad(z["X"], ((1, 0), (1, 0))); g.y = np.pad(z["Y"], ((1, 0), (1, 0)))
            self._set_flow(g, dict(u=z["U"], v=z["V"], w=z.get("W", z["T"] * 0),
                                   psi=z.get("P", z["T"] * 0), T=z["T"]),
                           f"{Path(f).name}, последний из {len(zones)} кадров")
            self.nb.select(1)
        else:
            self.mesh_res = None
            self.grid_xy = (z["X"], z["Y"])
            self.grid_title = Path(f).name
            self.redraw_mesh()
            self.nb.select(0)
        self.write(f"Открыт файл {f}: {len(zones)} кадр(ов)")

    # ================================================================== TAB 3
    def _build_report_tab(self):
        tab = ttk.Frame(self.nb, style="Side.TFrame", padding=(4 * PAD, 3 * PAD))
        self.nb.add(tab, text="3. Отчет")
        ttk.Label(tab, text="Отчет со всеми результатами в одном файле",
                  font=(self.base, 14, "bold"), background=PANEL).pack(anchor="w")
        ttk.Label(tab, style="Muted.TLabel", font=(self.base, 10), wraplength=760, justify="left", text=(
            "Программа выполнит все этапы на исходных данных диссертации и соберет рисунки в файл "
            "report.html. Он открывается в любом браузере, его можно отправить по почте "
            "или сохранить в PDF (Ctrl+P).")).pack(anchor="w", pady=(PAD, 2 * PAD))
        items = ("Критерии качества сетки в круговом кольце - объемные рисунки, как рис. 9-10",
                 "Построение сетки в криволинейной области: начальная и итоговая сетка, "
                 "критерии Q1-Q4, история итераций, лучшая сетка по каждому критерию",
                 "Свободная конвекция в кольце и в криволинейной области: рисунки и анимации")
        for t in items:
            row = ttk.Frame(tab, style="Side.TFrame"); row.pack(anchor="w", fill="x", pady=3)
            ttk.Label(row, text="●", foreground=ACCENT, background=PANEL).pack(side="left", anchor="n")
            ttk.Label(row, text=t, wraplength=740, justify="left", background=PANEL).pack(
                side="left", padx=PAD)
        box = ttk.Frame(tab, style="Side.TFrame"); box.pack(anchor="w", pady=(3 * PAD, 0))
        b = ttk.Button(box, text="Создать отчет", style="Accent.TButton", command=self.make_report)
        b.pack(side="left")
        self._run_buttons.append(b)
        ttk.Label(box, text="около 2 минут", style="Muted.TLabel").pack(side="left", padx=2 * PAD)

    def make_report(self):
        out = self.out("отчет")
        self.write("Создание отчета ...")
        self.progress["value"] = 0

        def job():
            from .report import make_report
            return make_report(out, DATA / "bottom_border.dat", DATA / "top_border.dat",
                               log=lambda m: self.q.put(("msg", m)))

        def done(p):
            self.write(f"Отчет: {p}")
            self._flash_done("Отчет готов")
            open_path(p)
        self.run_bg(job, done)


def selftest(path):
    """Quick check of the packaged program: mesh + flow, writes a report file.
    Used by the build server:  RingFlow.exe --selftest result.txt"""
    import time
    import tempfile
    lines = []
    t0 = time.time()
    try:
        tmp = Path(tempfile.mkdtemp())
        boundary.run(DATA / "bottom_border.dat", DATA / "top_border.dat", tmp, n1=51, verbose=False)
        r = mesh.run(tmp / "rez_param.dat", tmp, mesh.MeshConfig(max_iter=200), verbose=False)
        lines.append(f"mesh ok: min Q1 = {r.best['Q1']['value']:.4f}")
        f = convection.solve(convection.preset("ring", max_steps=20, snapshot_every=0,
                                               write_initial=False), None, verbose=False)
        lines.append(f"flow ok: max T = {f.T.max():.4f}, min psi = {f.psi.min():.6f}")
        from ._jit import HAVE_NUMBA
        lines.append(f"numba: {HAVE_NUMBA}")
        lines.append(f"time: {time.time() - t0:.1f} s")
        lines.append("SELFTEST OK")
        code = 0
    except Exception:
        lines.append(traceback.format_exc())
        lines.append("SELFTEST FAILED")
        code = 1
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return code


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
