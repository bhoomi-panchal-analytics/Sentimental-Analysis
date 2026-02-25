# utils.py

import logging

logging.basicConfig(level=logging.INFO)

def safe_execution(func):
    try:
        return func()
    except Exception as e:
        logging.error(f"Error: {e}")
        return None
