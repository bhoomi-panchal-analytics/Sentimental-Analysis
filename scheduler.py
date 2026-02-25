# scheduler.py

import time

def auto_refresh(interval_seconds=60):
    while True:
        time.sleep(interval_seconds)
