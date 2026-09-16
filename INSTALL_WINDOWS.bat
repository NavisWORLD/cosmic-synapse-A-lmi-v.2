@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\install.ps1" %*
exit /b %ERRORLEVEL%
