@echo off
cd /d "%~dp0"
set "PROJECT_PY=%~dp0..\.venv\Scripts\python.exe"
if exist "%PROJECT_PY%" (
  "%PROJECT_PY%" -m streamlit run "%~dp0Project1_EDA.py"
) else (
  python -m streamlit run "%~dp0Project1_EDA.py"
)
pause
