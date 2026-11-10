@echo off
rem Runs the whole chain on the original input data. Results go to .\results
rem Creates its own Python environment in .\.venv (the old ..\.venv was made
rem on another computer and does not work here).
chcp 65001 >nul
cd /d "%~dp0"
set VIRTUAL_ENV=
set PYTHONHOME=

if not exist ".venv\Scripts\python.exe" (
    echo === creating Python environment in .venv ===
    py -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Could not create a virtual environment. Is Python 3.9+ installed?
    echo Get it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
    pause
    exit /b 1
)
set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt || (echo pip install failed & pause & exit /b 1)

echo === tests (Python vs original Fortran) ===
%PY% -m pytest -q

echo === demonstration report (results\report\report.html) ===
%PY% -m ringflow report --out results\report
start "" "results\report\report.html"

echo === curved domain: boundary -^> mesh -^> flow ===
%PY% -m ringflow pipeline --bottom ..\Dvizh_zhid\physical_area\bottom_border.dat --top ..\Dvizh_zhid\physical_area\top_border.dat --out results\pipeline --viscosity 0.001 --max-steps 3000 --snapshot-every 30 --print-every 100 --plot --vtk
%PY% -m ringflow animate results\pipeline\3_flow\rezsurf.dat --out results\pipeline\3_flow\temperature.gif

echo === circular ring (stops at convergence or 20000 steps, ~10 min) ===
%PY% -m ringflow convect --preset ring --snapshot-every 100 --print-every 100 --out results\ring --plot --vtk
%PY% -m ringflow animate results\ring\rezsurf.dat --out results\ring\temperature.gif

echo Done. Open the results folder.
pause
