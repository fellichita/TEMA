@echo off
rem Открывает панель управления веб-сервера (работает, пока открыто окно "web-demo").
rem Другой порт панели: передайте его первым аргументом.
set "PANEL_PORT=%~1"
if "%PANEL_PORT%"=="" set "PANEL_PORT=8502"
start "" "http://127.0.0.1:%PANEL_PORT%/"
