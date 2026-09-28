@echo off
rem Один раз сохраняет токен fxTunnel в системном хранилище Windows.
rem Публичная ссылка сайта (web-demo.bat) без этого не откроется.
cd /d "%~dp0..\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\fxtunnel_login.ps1"
