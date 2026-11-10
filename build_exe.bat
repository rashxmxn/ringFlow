@echo off
rem Builds RingFlow.exe on this computer (same steps as the GitHub build).
rem Result: dist\RingFlow\RingFlow.exe  and  RingFlow_Windows.zip
chcp 65001 >nul
cd /d "%~dp0"
set VIRTUAL_ENV=
set PYTHONHOME=

if not exist ".venv\Scripts\python.exe" (
    echo === creating Python environment in .venv ===
    py -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Python 3.9+ is required: https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
    pause
    exit /b 1
)
set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt pyinstaller markdown || (echo pip install failed & pause & exit /b 1)

echo === tests (Python vs original Fortran) ===
%PY% -m pytest -q || (echo TESTS FAILED & pause & exit /b 1)

echo === building RingFlow.exe (a few minutes) ===
%PY% -m PyInstaller --noconfirm --windowed --name RingFlow --add-data "ringflow/data;ringflow/data" --hidden-import PIL._tkinter_finder run_gui.py || (echo BUILD FAILED & pause & exit /b 1)

echo === self-test of the built program ===
start "" /wait "dist\RingFlow\RingFlow.exe" --selftest selftest.txt
type selftest.txt
findstr /c:"SELFTEST OK" selftest.txt >nul || (echo SELF-TEST FAILED & pause & exit /b 1)

echo === packaging ===
%PY% tools\make_description_html.py README.md "dist\RingFlow\Описание.html"
copy /y README.md "dist\RingFlow\README.md" >nul
if exist RingFlow_Windows.zip del RingFlow_Windows.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\RingFlow' -DestinationPath 'RingFlow_Windows.zip'"

echo.
echo Done:  %CD%\RingFlow_Windows.zip
echo The program itself:  %CD%\dist\RingFlow\RingFlow.exe
explorer .
pause
