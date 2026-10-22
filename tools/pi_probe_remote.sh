echo "===HOST==="; hostname; whoami; uptime; head -2 /etc/os-release; uname -r
echo "===THERMAL==="; vcgencmd measure_temp 2>/dev/null; cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null
echo "===MEM==="; free -h
echo "===DISK==="; df -h / /dev/shm; ls -la /dev/shm/audio_cache 2>/dev/null | tail -5
echo "===SERVICES==="; systemctl list-units --type=service --state=running --no-pager | grep -iE "pi5|dispatch|hailo|ollama|bluetooth|pipewire|wireplumber|telegram|hotspot|gpio"
echo "===DISPATCHER==="; systemctl status pi5-dispatcher-api.service --no-pager -n 0 2>&1 | head -12
echo "===API==="; curl -s -m 3 http://127.0.0.1:8000/api/status 2>/dev/null | head -c 800; echo
echo "===HAILO==="; lspci 2>/dev/null | grep -i hailo; ls /dev/hailo* 2>/dev/null; which hailortcli && timeout 10 hailortcli fw-control identify 2>&1 | head -15
echo "===OLLAMA==="; which ollama && timeout 10 ollama list 2>&1; curl -s -m 3 http://127.0.0.1:11434/api/tags 2>/dev/null | head -c 600; echo
echo "===GIT==="; for d in /home/pi/code_dispatcher /home/pi/pi5-hailo-localnote /home/pi/pi5-llm-localnote; do [ -d "$d" ] && { echo "-- $d"; git -C "$d" log --oneline -3 2>/dev/null; git -C "$d" status -sb 2>/dev/null | head -8; }; done
echo "===BT==="; timeout 5 bluetoothctl devices 2>/dev/null
echo "===NET==="; ip -4 -br addr
echo "===JOURNAL==="; journalctl -u pi5-dispatcher-api.service -n 25 --no-pager -o short-iso 2>/dev/null
echo "===ERRORS_1H==="; journalctl -p err -S -1h --no-pager -o short-iso 2>/dev/null | tail -20
