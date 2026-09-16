"""
Logging utility for the desktop assistant with timestamping and structured formatting.
"""

import logging
import sys

def setup_logger(name: str = "MultimodalAssistant", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.propagate = False
    return log

logger = setup_logger()
