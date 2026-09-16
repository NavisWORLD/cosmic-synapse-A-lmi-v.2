@echo off
setlocal
powershell.exe -NoLogo -NoProfile -File "%~dp0scripts\windows\uninstall.ps1" %*
exit /b %ERRORLEVEL%
