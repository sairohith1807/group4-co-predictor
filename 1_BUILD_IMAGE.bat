@echo off
title Group 4 - Build Docker image
cd /d "%~dp0"
echo ============================================================
echo  Group 4 - CO(GT) predictor : building the Docker image
echo  This takes 5-15 minutes the first time (installs packages
echo  and trains the model). Keep Docker Desktop open.
echo ============================================================
docker build -t group4-co-predictor:1.0 .
if errorlevel 1 (
  echo.
  echo BUILD FAILED - is Docker Desktop running ^(Engine running^)?
  pause
  exit /b 1
)
echo.
echo DONE. Image "group4-co-predictor:1.0" is ready.
echo Next: double-click 2_RUN_PROJECT.bat
pause
