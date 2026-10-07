# f:\12_prj_raspi5\code_dispatcher\worker.py
"""
Raspberry Pi 5 Single-Concurrency Worker Daemon
Listens on the Redis queue and executes inference tasks sequentially.
"""

import os
import sys
from redis import Redis
from rq import Worker, Queue

# Ensure current directory is on python path for tasks module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QUEUE_NAME = "code_review_queue"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379

def main():
    redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    queue = Queue(QUEUE_NAME, connection=redis_conn)
    print(f"[*] Starting Pi 5 Worker on queue: '{QUEUE_NAME}' (Concurrency = 1)...")
    worker = Worker([queue], connection=redis_conn)
    worker.work()

if __name__ == "__main__":
    main()
