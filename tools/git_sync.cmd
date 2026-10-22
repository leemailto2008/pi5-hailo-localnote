@echo off
setlocal
cd /d F:\13_prj_raspi5_localnote
set OUT=%~dp0pi_probe_out
del /q "%OUT%\DONE.flag" 2>nul
(
git add .gitignore .claudeignore .claude/CLAUDE.md code_dispatcher/bluetooth_manager.py code_dispatcher/main.py tools/pi_run.cmd tools/pi_push.cmd tools/pi_probe.cmd tools/pi_probe_remote.sh tools/git_sync.cmd
git commit -F tools/commit_msg.txt
git log --oneline -1
git push gitlab main
git push origin main
git status -sb
) > "%OUT%\git_sync.txt" 2>&1
echo done > "%OUT%\DONE.flag"
