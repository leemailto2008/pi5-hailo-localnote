# code_dispatcher/last_meeting.py
"""
最後一場會議的保存 (Last Meeting Snapshot)：服務重啟後還看得到上一場的逐字稿、紀要與錄音指標，也方便除錯。

- 只保留一份：每場完成 (含重新整理紀要) 就整份覆蓋，不累積歷史。
- 只存文字 JSON (幾 KB)，不存音訊；零磁碟寫入原則針對的是音訊。
- 先寫暫存檔再 os.replace，斷電時不會留下半份檔案。
- 2026-10-30 使用者決定：10-29 測試人員那場會議，在部署重啟服務時連同紀要一起遺失 (結果只存在記憶體)。
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "last_meeting.json")


def last_meeting_path() -> str:
    return os.environ.get("PI5_LAST_MEETING_PATH", DEFAULT_PATH)


def save_last_meeting(result: Dict[str, Any], status: Dict[str, Any], path: Optional[str] = None) -> bool:
    """整份覆蓋寫入；失敗只印警告並回傳 False，不影響會議流程。"""
    path = path or last_meeting_path()
    payload = {"saved_at": datetime.now().isoformat(timespec="seconds"), "result": result, "status": status}
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError) as e:
        print(f"[LastMeeting] 無法保存最後一場會議: {e}")
        return False


def load_last_meeting(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """讀回 {"saved_at", "result", "status"}；沒有檔案或內容壞掉時回傳 None。"""
    path = path or last_meeting_path()
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        print(f"[LastMeeting] 最後一場會議檔案無法讀取，略過: {e}")
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict) or not payload["result"].get("job"):
        print("[LastMeeting] 最後一場會議檔案格式不符，略過")
        return None
    return payload
