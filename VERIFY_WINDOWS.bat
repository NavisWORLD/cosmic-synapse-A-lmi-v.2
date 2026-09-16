@echo off
setlocal
powershell.exe -NoLogo -NoProfile -File "%~dp0scripts\windows\verify.ps1" %*
exit /b %ERRORLEVEL%
