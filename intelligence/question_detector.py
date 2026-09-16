"""
Lightweight Question and Intent Detector to gate expensive LLM reasoning calls.
"""

from enum import Enum
from dataclasses import dataclass
import re
from typing import Tuple

from utils.timing import time_it

class IntentCategory(Enum):
    QUESTION = "QUESTION"
    STATEMENT = "STATEMENT"
    COMMAND = "COMMAND"
    IRRELEVANT = "IRRELEVANT"

@dataclass
class DetectedIntent:
    category: IntentCategory
    confidence: float
    trigger_llm: bool
    text: str

class QuestionDetector:
    """Classifies utterances to prevent unnecessary LLM API calls and latency."""

    QUESTION_WORDS = {
        "what", "why", "how", "when", "where", "who", "which",
        "can", "could", "would", "should", "is", "are", "do", "does", "did",
        "tell", "give", "define"
    }

    COMMAND_WORDS = {
        "explain", "summarize", "describe", "show", "calculate", "find", "compare",
        "search", "solve", "code", "write", "implement", "analyze", "debug", "check", "help", "answer"
    }

    FILLER_WORDS = {
        "uh", "um", "yeah", "ok", "okay", "alright", "hmm", "mhm", "cool", "sure", "fine", "bye", "good"
    }

    @time_it("question_detection")
    def detect(self, utterance: str) -> DetectedIntent:
        """
        Evaluates an utterance and determines whether it should invoke the reasoning layer.
        """
        clean = utterance.strip()
        lower = clean.lower()

        # 1. Check for empty or short filler utterances
        words = re.findall(r"\b\w+\b", lower)
        if not words or (len(words) <= 4 and all(w in self.FILLER_WORDS or w in ("i", "am", "it", "that") for w in words)):
            return DetectedIntent(
                category=IntentCategory.IRRELEVANT,
                confidence=0.9,
                trigger_llm=False,
                text=clean
            )

        # 2. Check for explicit question mark
        if "?" in clean:
            return DetectedIntent(
                category=IntentCategory.QUESTION,
                confidence=0.95,
                trigger_llm=True,
                text=clean
            )

        # 3. Check for question words anywhere in the first 4 words
        first_few = words[:4]
        if any(w in self.QUESTION_WORDS for w in first_few):
            return DetectedIntent(
                category=IntentCategory.QUESTION,
                confidence=0.88,
                trigger_llm=True,
                text=clean
            )

        # 4. Check for imperative / command words in the first 4 words
        if any(w in self.COMMAND_WORDS for w in first_few):
            return DetectedIntent(
                category=IntentCategory.COMMAND,
                confidence=0.85,
                trigger_llm=True,
                text=clean
            )

        # 5. Technical keywords indicator (interview / coding / meeting)
        tech_indicators = {"difference", "complexity", "algorithm", "approach", "solution", "leetcode", "runtime", "memory", "function"}
        if any(w in words for w in tech_indicators):
            return DetectedIntent(
                category=IntentCategory.QUESTION,
                confidence=0.80,
                trigger_llm=True,
                text=clean
            )

        # 6. If utterance is substantial (>= 4 words) and not filler, trigger reasoning
        if len(words) >= 4:
            return DetectedIntent(
                category=IntentCategory.QUESTION,
                confidence=0.75,
                trigger_llm=True,
                text=clean
            )

        return DetectedIntent(
            category=IntentCategory.STATEMENT,
            confidence=0.70,
            trigger_llm=False,
            text=clean
        )
