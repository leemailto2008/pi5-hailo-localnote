@echo off
setlocal
cd /d F:\13_prj_raspi5_localnote
set OUT=%~dp0pi_probe_out
del /q "%OUT%\DONE.flag" 2>nul
(
git add code_dispatcher/voice_pipeline.py tools/git_sync.cmd
git commit -F tools/commit_msg.txt
git log --oneline -1
git push gitlab main
git push origin main
git status -sb
) > "%OUT%\git_sync.txt" 2>&1
echo done > "%OUT%\DONE.flag"
