# f:\12_prj_raspi5\code_dispatcher\tasks.py
"""
Raspberry Pi 5 Code Inference Worker Task Module
Executes asynchronous code analysis and review tasks via local Ollama.
"""

import time
import requests
from datetime import datetime
from typing import Dict, Any


def analyze_code_task(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ Worker Job: Calls local Ollama to perform code analysis/review.
    Strictly executed sequentially (Concurrency = 1) to safeguard CPU resources.
    """
    model = task_payload.get("model", "qwen2.5-coder:7b")
    code = task_payload.get("code", "")
    prompt = task_payload.get("prompt", "Perform a thorough code review, identifying potential bugs, security vulnerabilities, edge cases, and performance optimizations.")
    language = task_payload.get("language", "python")
    filename = task_payload.get("filename", "snippet")
    temperature = float(task_payload.get("temperature", 0.2))
    max_tokens = int(task_payload.get("max_tokens", 1500))

    if code and code.strip():
        constructed_prompt = f"""You are an elite Senior Staff Software Engineer and Security Architect.
Please review the provided code carefully and provide concise, actionable, and highly technical feedback.

### Instructions:
{prompt}

### Target File: `{filename}` (Language: {language})
```{language}
{code}
```

### Review Format:
1. **Summary & Architecture Assessment**: Overall code quality, readability, and design pattern evaluation.
2. **Critical Bugs & Security Flaws**: Vulnerabilities, edge cases, null pointer/exception risks, or race conditions.
3. **Performance & Memory Optimizations**: Time/space complexity improvements and I/O efficiency.
4. **Refactored Code (if applicable)**: Clean, production-ready code with type annotations and docstrings.
"""
    else:
        constructed_prompt = f"""You are an elite Senior Staff Software Engineer and AI Assistant.
Please provide a clear, accurate, and structured answer to the following technical question.

### Question:
{prompt}
"""

    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model,
        "prompt": constructed_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        t1 = time.perf_counter()

        data = response.json()
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns > 0 else (t1 - t0)
        tps = eval_count / eval_duration_s if eval_duration_s > 0 else 0.0

        return {
            "status": "completed",
            "model": model,
            "filename": filename,
            "language": language,
            "result": data.get("response", ""),
            "total_tokens": eval_count,
            "eval_duration_s": round(eval_duration_s, 2),
            "eval_tps": round(tps, 2),
            "total_elapsed_s": round(t1 - t0, 2),
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        t1 = time.perf_counter()
        return {
            "status": "failed",
            "model": model,
            "filename": filename,
            "error": str(e),
            "total_elapsed_s": round(t1 - t0, 2),
            "failed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
