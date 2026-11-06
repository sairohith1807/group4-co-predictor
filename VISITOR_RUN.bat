@echo off
title Group 4 - CO(GT) predictor
cd /d "%~dp0"
echo Loading the Group 4 image (first time only, about 1 minute)...
docker image inspect group4-co-predictor:1.0 >nul 2>&1 || docker load -i group4-co-predictor.tar
if errorlevel 1 (
  echo FAILED - install Docker Desktop and start it first.
  pause
  exit /b 1
)
docker rm -f group4-co >nul 2>&1
docker run -d --name group4-co -p 8501:8501 -p 8000:8000 -p 5000:5000 group4-co-predictor:1.0
echo Waiting 15 seconds...
timeout /t 15 /nobreak >nul
start http://localhost:8501
echo  App ......... http://localhost:8501
echo  API docs .... http://localhost:8000/docs
echo  MLflow ...... http://localhost:5000
echo To stop later:  docker rm -f group4-co
pause
