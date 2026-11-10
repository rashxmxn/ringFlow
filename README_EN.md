# ringflow — Python port of the 2015 Fortran project

This folder is a complete Python 3 rewrite of the Fortran programs in
`!проект док`. Every program has been run against the original Fortran
code on the original input files, and the results match.

## What the original project is

It is a set of programs for **numerical modelling of natural (free)
convection of a viscous fluid in a ring-shaped domain**. The domain is
"doubly-connected": a region with a hole, like a washer. The flow is
driven by a temperature difference between the inner and the outer wall.
The 2015 talk is `Prezent_09_04_2015/Комплекс программ для численного
моделирования.ppt`, and the .avi files there are animations of its results.

The work flow has three steps. Each original program does one step:

| Step | Original Fortran file | Python module | What it does |
|---|---|---|---|
| 1 | `Dvizh_zhid/physical_area/interp.for` | `ringflow/boundary.py` | Takes the outer contour (`bottom_border.dat`) and the inner contour (`top_border.dat`) and places 51 evenly spaced points on each. Writes `rez_param.dat`. |
| 2 | `Dvusv_set_dvish/Setka_ED2_II.for` | `ringflow/mesh.py` | Builds a curvilinear O-type mesh (51×26) between the two contours. It solves elliptic grid equations with an ADI iteration and scores the mesh with 4 quality criteria (cell skewness, cell angles, aspect ratio, area uniformity). It keeps the best mesh and writes `xxx.dat`, `yyy.dat` and `surf1..4.dat`. |
| 3 | `Dvizh_zhid/Dvizh_psi_om_Kolc_jav.for` (+ 2 variants) | `ringflow/convection.py` | The flow solver, using stream function ψ, vorticity ω and temperature T in curvilinear coordinates with Boussinesq buoyancy. Each time step does: an explicit vorticity step, then the Poisson equation for ψ, then velocities and wall vorticity, then an explicit temperature step. Writes Tecplot files `rezsurf.dat` and `metr_koef.dat`. |

The flow solver came in three versions. Here they are **presets**:

| preset | original | domain |
|---|---|---|
| `ring` | `Dvizh_zhid/Dvizh_psi_om_Kolc_jav.for` | circular annulus 0.5 ≤ r ≤ 2, 101×51 grid, hot inner wall, Gauss–Seidel ψ solver |
| `curved` | `Dvizh_zhid/Dv_kv_set/Dvizh_psi_om_Kolc_jav.for` | the curved domain, using the mesh from step 2 (`xxx.dat` / `yyy.dat`) |
| `ring_adi` | `Dvizh_zhid/Dvizh_psi_om_Kolc.for` | annulus 101×101 with an ADI ψ solver. This was an unfinished experiment by the original author (see *Known issues*). |

`Dvusv_set_dvish/dmin.for` only prints the min and max of 4 numbers, so it is not ported.

## Windows program (no Python needed)

`RingFlow.exe` is a window program with Russian labels (`ringflow/gui.py`,
start from source with `python run_gui.py`). It has three tabs: mesh +
quality criteria (2-D/3-D, iteration history, statistics table), flow and
heat transfer (all parameters editable, live picture, Stop, GIF), and the
HTML report. The .exe is built automatically by GitHub Actions
(`.github/workflows/build-windows.yml`): tests, PyInstaller build, a
self-test of the built program (`RingFlow.exe --selftest out.txt`), then
`RingFlow_Windows.zip` is published on the Releases page. The user-facing
description in Russian is `README.md`.

## Installation

You need Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

Numba is optional but strongly recommended. With Numba the solvers run at
the same speed as the compiled Fortran. Without it the code still works but
is roughly 100–200× slower.

## Usage

Run these from this folder:

```bash
# step 1: boundary points
python -m ringflow boundary --bottom ../Dvizh_zhid/physical_area/bottom_border.dat --top ../Dvizh_zhid/physical_area/top_border.dat --out results/1_boundary --plot

# step 2: mesh (--plot saves a picture coloured by each quality criterion)
python -m ringflow mesh --boundary results/1_boundary/rez_param.dat --out results/2_mesh --plot

# step 3: flow in the curved domain
python -m ringflow convect --preset curved --mesh-x results/2_mesh/xxx.dat --mesh-y results/2_mesh/yyy.dat --out results/3_flow --plot

# or all three steps in one command
python -m ringflow pipeline --bottom ../Dvizh_zhid/physical_area/bottom_border.dat --top ../Dvizh_zhid/physical_area/top_border.dat --out results/pipeline --plot --vtk

# the circular-ring case
python -m ringflow convect --preset ring --out results/ring --plot
```

Useful options for `convect` (run `python -m ringflow convect --help` to see all):

* `--max-steps`, `--tau`, `--beta` (buoyancy), `--viscosity`, `--eps` override the values that were hard-coded in the Fortran
* `--snapshot-every N` writes a Tecplot zone every N steps. The original wrote one every 5 steps, which is how the 1.9 GB `rezsurf.dat` files came about. Use 0 to turn it off.
* `--plot` saves `fields.png` (temperature, streamlines and velocity vectors) and `convergence.png`
* `--vtk` writes `result.vtk`, which opens in the free ParaView

## Showing the project: `report` and `quality`

```bash
python -m ringflow report                      # ~1.5 min -> results/report/report.html
python -m ringflow quality --ring --out q_ring # criteria of the circular mesh (Fig. 6, 9, 10)
python -m ringflow quality --grid results/2_mesh/xxx.dat results/2_mesh/yyy.dat --out q_curved
```

`report` runs every step on the original data and writes **one HTML file**
(all pictures embedded, in Russian). It opens in any browser and can be
emailed or printed to PDF. It contains:

* the circular mesh and its quality criteria as 3-D surfaces W = Q(x, y), like Fig. 9–10 of the thesis
* the curved domain: boundary points, initial vs final mesh, how the criteria change during the mesh iterations, and the best mesh for each criterion (2-D and 3-D)
* the flow and temperature results for both domains, as pictures and animations

The mesh quality criteria (`ringflow/quality.py`, computed exactly like
`KRIT_KA` in the Fortran) are:

| | criterion (thesis wording) | formula |
|---|---|---|
| Q1 | невырожденность (triangle areas, Fig. 1 p. 72) | 2·min(S123, S134, S124, S234) / S_cell |
| Q2 | ортогональность | min sin of the 4 cell angles |
| Q3 | невытянутость ячеек | min edge / max edge |
| Q4 | локальная равномерность | min(S/S_avg, S_avg/S) |

All are in [0, 1], where 1 is a perfect cell; a value ≤ 0 means an inverted cell.
`quality` writes `quality.csv` (min/mean/max for each criterion) and one 2-D and one 3-D picture per criterion.

Note: the mesh iterations never converge to a single mesh. About every 270
iterations the mesh briefly gets inverted cells (min Q1 < 0) and then
recovers. This is why the program keeps the best mesh for each criterion
rather than the last one.

## New features (not in the original)

* A **command line interface**. All parameters are options instead of being edited into the source.
* **Pictures** made with matplotlib. The original needed Tecplot, a paid program.
* **Animations**: `python -m ringflow animate results/ring/rezsurf.dat --out ring.gif` (or `.mp4` if ffmpeg is installed). This replaces the Tecplot .avi files.
* **Viewing the old 2015 results**: `python -m ringflow plot "../Dvusv_set_dvish/surf1.dat"`. This reads the original Tecplot files, including the `*****` overflow fields.
* **Export** to NumPy (`result.npz`, `mesh_result.npz`) and VTK/ParaView (`to-vtk`).
* **Automated tests** (`python -m pytest`). They check the Python code against reference output from the original Fortran.
* Use from Python:
  ```python
  from ringflow import convection
  res = convection.solve(convection.preset("ring", max_steps=2000, snapshot_every=0))
  print(res.T.max(), res.psi.min())
  ```

## How the port was checked

The original `.for` files were compiled with gfortran 15 in double precision
and run on the original input data. The Python output was then compared
file by file:

| program | compared | result |
|---|---|---|
| interp (n1 = 21 and 51) | `rez_param.dat`, `rez_kord.dat` | identical |
| Setka_ED2_II (full run, 5000 iterations) | `xxx/yyy/zzz.dat`, `surf1-4.dat`, `rezsurf1.dat`, log | identical; same best iterations L1–L4 = 4969 / 3380 / 2 / 595 |
| ring (`..._jav.for`), 20 and 300 steps | `rezsurf.dat`, `metr_koef.dat`, log | identical |
| curved (`Dv_kv_set`), 20 steps | `rezsurf.dat`, `metr_koef.dat`, log | identical |
| ring_adi (`Dvizh_psi_om_Kolc.for`) | `rezsurf.dat`, log | identical |

The gfortran reference also reproduces the 2015 data files on disk, for
example `Dvusv_set_dvish/xxx.dat`, to within 2·10⁻⁵. That difference comes
only from single vs double precision.

Two differences from the original are intentional:

* The original mixes `REAL*4` (single precision) and `double precision`. The port uses double precision everywhere, so new results can differ from the 2015 files in the 5th–6th digit.
* In output files the columns always have at least one space between them, so very large numbers never run together.

The internal arrays are 1-based (index 0 is unused), so `a[i, j]` in the
Python kernels is the same element as `a(i,j)` in Fortran. This lets you
compare the code side by side, line by line.

## Known issues inherited from the original (the port reproduces them on purpose)

These were found while porting. By default they are **reproduced exactly**,
so the results stay comparable with the original. Some of them have a
switch to fix them. Please check them with whoever owns the science.

1. **Missing factor ½ in the ∂/∂ξ metric terms.** In `Setka_ED2_II.for` and `Dv_kv_set/...jav.for`, the interior nodes use `(x(i+1)-x(i-1))/h1` without the 0.5. The periodic node `i=1` does have the 0.5. Fix: `--fix-metric-stencil`.
2. **The `curved` version has zero viscosity and zero heat conduction.** The line `tmx(i,j)=0.001` is commented out, so `tmx` is an uninitialised array (zero with Compaq Visual Fortran). Fix: `--viscosity 0.001`.
3. **The `ring_adi` version diverges.** It is a 3-step test with no wall-vorticity condition, and ψ grows to about 5·10⁴. The Fortran output shows the same `*****` fields.
4. **The `ring` version uses analytic metric terms with respect to the angle φ and radius r**, not the unit computational coordinates ξ, η that the grid steps `h1`, `h2` assume. The missing factors are 2π and 1.5. The `curved` version uses finite differences in ξ, η, so the two versions scale the equations differently.
5. **The periodic tridiagonal solver** treats the seam node n1 as a separate node instead of the same point as node 1. Its value is overwritten afterwards, so the effect is small. See `tridiag.cyclic_sweep`.
6. `pi = 3.141592655` (the original value) is kept.

## Files

```
ringflow/
  boundary.py    step 1 (interp.for)
  mesh.py        step 2 (Setka_ED2_II.for, KRIT_KA)
  convection.py  step 3 (Dvizh_psi_om_Kolc*.for), presets ring / curved / ring_adi
  tridiag.py     tridiagonal "progonka" solvers
  io.py          Tecplot, xxx.dat, VTK reading and writing
  plotting.py    pictures and animations
  quality.py     mesh quality criteria: tables, 2-D / 3-D pictures
  report.py      one-click HTML demonstration report
  cli.py         command line
tests/           comparison with Fortran reference output
run_examples.bat runs everything on Windows (double-click)
```

The older, partial `python_rewrite` folder in `!проект док` was left as it
was. It was never checked against the Fortran, and this package replaces it.
