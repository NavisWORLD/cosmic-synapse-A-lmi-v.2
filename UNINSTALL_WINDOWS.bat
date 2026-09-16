@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\uninstall.ps1" %*
exit /b %ERRORLEVEL%
