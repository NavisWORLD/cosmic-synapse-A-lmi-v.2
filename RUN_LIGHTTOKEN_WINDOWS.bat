@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\lighttoken\run.ps1" %*
exit /b %ERRORLEVEL%
