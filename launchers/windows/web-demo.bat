@echo off
rem Windows-двойник файла "launchers/macos/web-demo.command" для macOS.
setlocal
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" -E -s -m scripts.run_web_demo %*
if errorlevel 1 pause
endlocal
