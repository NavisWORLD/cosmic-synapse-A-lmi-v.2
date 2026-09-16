@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\update.ps1" %*
exit /b %ERRORLEVEL%
