@echo off
rem Lets "stm <command>" work in cmd (the SSH default shell) from this folder; it runs stm.ps1 beside it.
pwsh -NoProfile -File "%~dp0stm.ps1" %*
