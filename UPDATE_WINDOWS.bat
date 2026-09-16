@echo off
setlocal
powershell.exe -NoLogo -NoProfile -File "%~dp0scripts\windows\update.ps1" %*
exit /b %ERRORLEVEL%
