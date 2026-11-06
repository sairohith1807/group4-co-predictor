@echo off
title Group 4 - Running
cd /d "%~dp0"
docker rm -f group4-co >nul 2>&1
echo Starting the project...
docker run -d --name group4-co -p 8501:8501 -p 8000:8000 -p 5000:5000 group4-co-predictor:1.0
if errorlevel 1 (
  echo Could not start. Did you run 1_BUILD_IMAGE.bat first?
  pause
  exit /b 1
)
echo Waiting 15 seconds for the services to start...
timeout /t 15 /nobreak >nul
start http://localhost:8501
echo.
echo  App ......... http://localhost:8501
echo  API docs .... http://localhost:8000/docs
echo  MLflow ...... http://localhost:5000
echo.
echo To stop: double-click 4_STOP_PROJECT.bat
pause
