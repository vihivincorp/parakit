"""
Response Manager to validate, format, prioritize, and deduplicate assistant outputs.
"""

from dataclasses import dataclass
import time
from typing import Optional, Callable
from utils.logger import logger

@dataclass
class UIResponse:
    question: str
    answer: str
    timestamp: float
    is_duplicate: bool = False

class ResponseManager:
    """Controls the flow of generated answers into the presentation overlay."""

    def __init__(self, debounce_interval_sec: float = 1.5):
        self.debounce_interval_sec = debounce_interval_sec
        self.last_question: str = ""
        self.last_answer: str = ""
        self.last_emit_time: float = 0.0

    def process_and_dispatch(
        self,
        question: str,
        raw_answer: str,
        on_response_ready: Optional[Callable[[UIResponse], None]] = None
    ) -> Optional[UIResponse]:
        """
        Validates, formats, and debounces response delivery.
        """
        if not raw_answer or not raw_answer.strip():
            logger.debug("Suppressed empty answer.")
            return None

        # Clean response while preserving newlines, bullet points, and code indentation
        import re
        clean_answer = re.sub(r'\n{3,}', '\n\n', raw_answer.strip())
        clean_question = question.strip()
        now = time.time()

        # Duplicate suppression: identical answer or identical question within debounce window
        is_duplicate = (
            (clean_answer == self.last_answer or clean_question == self.last_question)
            and (now - self.last_emit_time < self.debounce_interval_sec * 4)
        )

        if is_duplicate:
            logger.info("Suppressed duplicate response to maintain stable UI.")
            return None

        # Format and construct response
        ui_resp = UIResponse(
            question=clean_question,
            answer=clean_answer,
            timestamp=now,
            is_duplicate=False
        )

        self.last_question = clean_question
        self.last_answer = clean_answer
        self.last_emit_time = now

        if on_response_ready:
            on_response_ready(ui_resp)

        return ui_resp
