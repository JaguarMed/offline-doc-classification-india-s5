@echo off
echo Starting Document Classification API...
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m api.main
pause
