@echo off
title Group 4 - Save image for visitors
cd /d "%~dp0"
if not exist share mkdir share
echo Saving the image to share\group4-co-predictor.tar (about 1-2 GB, a few minutes)...
docker save group4-co-predictor:1.0 -o share\group4-co-predictor.tar
if errorlevel 1 (
  echo FAILED - did you run 1_BUILD_IMAGE.bat first?
  pause
  exit /b 1
)
copy /y VISITOR_RUN.bat share\ >nul
copy /y VISITOR_README.txt share\ >nul
echo.
echo DONE. Give visitors the whole "share" folder (USB key, Google Drive, Teams):
dir share
pause
