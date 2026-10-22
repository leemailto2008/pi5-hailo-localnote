@echo off
setlocal EnableDelayedExpansion
set KEY=C:\Users\Andrew\.ssh\id_rsa_pi5
set OUT=%~dp0pi_probe_out
del /q "%OUT%\DONE.flag" 2>nul
scp -i %KEY% -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=6 "%~2" pi@%~1:"%~3" > "%OUT%\%~1_push.txt" 2>&1
echo exit=!ERRORLEVEL! >> "%OUT%\%~1_push.txt"
if not "%~4"=="" (
  ssh -i %KEY% -o BatchMode=yes -o ConnectTimeout=6 pi@%~1 bash -s < "%~dp0%~4" > "%OUT%\%~1_%~n4.txt" 2>&1
  echo exit=!ERRORLEVEL! >> "%OUT%\%~1_%~n4.txt"
)
echo done > "%OUT%\DONE.flag"
