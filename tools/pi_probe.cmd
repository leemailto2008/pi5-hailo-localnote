@echo off
setlocal EnableDelayedExpansion
set KEY=C:\Users\Andrew\.ssh\id_rsa_pi5
set OUT=%~dp0pi_probe_out
set SSHOPT=-i %KEY% -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=6
del /q "%OUT%\DONE.flag" 2>nul
for %%H in (192.168.50.228 192.168.50.229) do (
  echo [%TIME%] probing %%H ...
  ssh %SSHOPT% pi@%%H bash -s < "%~dp0pi_probe_remote.sh" > "%OUT%\%%H.txt" 2>&1
  echo exit=!ERRORLEVEL! >> "%OUT%\%%H.txt"
)
echo done > "%OUT%\DONE.flag"
