"""
Logging utility for the desktop assistant with timestamping and structured formatting.
"""

import logging
import sys
import os

# Ensure standard streams use UTF-8 with character replacement on Windows
if hasattr(sys.stdout, "reconfigure") and sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure") and sys.stderr:
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def setup_logger(name: str = "MultimodalAssistant", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S"
        )

        # Console handler (if stdout available)
        if sys.stdout is not None:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(level)
            stdout_handler.setFormatter(formatter)
            log.addHandler(stdout_handler)

        # File handler
        try:
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assistant.log")
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)
        except Exception:
            pass

        log.propagate = False
    return log

logger = setup_logger()
