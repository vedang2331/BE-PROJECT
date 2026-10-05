@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=python

rem Check if current active python has torch
"%PYTHON_EXE%" -c "import torch" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto RUN

rem Check Conda environment be_project in user profile
if exist "%USERPROFILE%\anaconda3\envs\be_project\python.exe" (
    set PYTHON_EXE=%USERPROFILE%\anaconda3\envs\be_project\python.exe
    goto RUN
)
if exist "%USERPROFILE%\miniconda3\envs\be_project\python.exe" (
    set PYTHON_EXE=%USERPROFILE%\miniconda3\envs\be_project\python.exe
    goto RUN
)
if exist "%USERPROFILE%\anaconda3\envs\moment_detr\python.exe" (
    set PYTHON_EXE=%USERPROFILE%\anaconda3\envs\moment_detr\python.exe
    goto RUN
)

:RUN
"%PYTHON_EXE%" run_on_video\run_custom.py %*
endlocal
