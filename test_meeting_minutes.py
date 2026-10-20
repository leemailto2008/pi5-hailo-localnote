# test_meeting_minutes.py
"""
Unit tests for LocalNote Meeting Minutes parser and state machine.
"""

import unittest
from code_dispatcher.meeting_minutes import parse_markdown_minutes, JobStatus, MeetingResult


class TestMeetingMinutesParser(unittest.TestCase):

    def test_standard_markdown_parsing(self):
        sample_md = """## 摘要
- 客戶希望在 11 月前交付 200 台樣機，首波試產
- 針對單價提出 8% 的降價要求
- 雙方同意先行簽署 50 台試產訂單

## 待辦
| 事項 | 負責人 | 時間 |
|---|---|---|
| 發送修訂版報價單 | 張三 | 本週五 |
| 確認樣機產能排程 | 李四 | 下週一 |
| 安排客戶參訪產線 | 待定 | 下週三 |

## 決議
- 先行簽署 50 台試產訂單，價格以新報價為準
- 11 月前交付第一批產品
"""
        parsed = parse_markdown_minutes(sample_md)
        self.assertEqual(len(parsed["summary"]), 3)
        self.assertIn("客戶希望在 11 月前交付 200 台樣機", parsed["summary"][0])

        self.assertEqual(len(parsed["todos"]), 3)
        self.assertEqual(parsed["todos"][0]["item"], "發送修訂版報價單")
        self.assertEqual(parsed["todos"][0]["owner"], "張三")
        self.assertEqual(parsed["todos"][0]["due"], "本週五")

        self.assertEqual(len(parsed["decisions"]), 2)
        self.assertIn("先行簽署 50 台試產訂單", parsed["decisions"][0])

    def test_missing_table_graceful_degradation(self):
        sample_freeform = """## 摘要
- 這是一段簡短的測試摘要

這段逐字稿沒有明確待辦事項。
"""
        parsed = parse_markdown_minutes(sample_freeform)
        self.assertEqual(len(parsed["summary"]), 2)
        self.assertEqual(len(parsed["todos"]), 0)
        self.assertEqual(len(parsed["decisions"]), 0)


    def test_job_status_and_result_models(self):
        status = JobStatus()
        self.assertEqual(status.state, "idle")
        self.assertEqual(status.progress, 0)

        result = MeetingResult(
            job="test_job_1",
            started_at="2026-10-12T11:00:00",
            duration=90,
            summary=["重點一"],
            todos=[{"item": "工作", "owner": "工程師", "due": "今天"}],
            decisions=["定案"],
            transcript="錄音逐字稿",
            markdown="# 標題"
        )
        self.assertEqual(result.job, "test_job_1")
        self.assertEqual(len(result.summary), 1)
        self.assertEqual(result.todos[0]["owner"], "工程師")


if __name__ == "__main__":
    unittest.main()
