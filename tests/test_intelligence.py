"""
Unit tests for Context Manager, Question Detector, LLM Client, and Response Manager.
"""

import time
from audio.speech_to_text import TranscriptSegment
from intelligence.context_manager import MultimodalContextManager
from intelligence.question_detector import QuestionDetector, IntentCategory
from intelligence.llm_client import LLMClient
from intelligence.response_manager import ResponseManager

def test_context_manager():
    cm = MultimodalContextManager(speech_window_sec=2.0)
    cm.update_screen("Model: Random Forest, Accuracy: 91%")

    seg1 = TranscriptSegment(text="Why Random Forest?", start_time=1.0, end_time=2.0, confidence=0.9, timestamp=time.time())
    cm.add_transcript(seg1)

    payload = cm.format_context_for_llm("Why Random Forest?")
    assert "Random Forest" in payload["screen_context"]
    assert "Why Random Forest?" in payload["speech_context"]

    # Test rolling window expiration
    old_seg = TranscriptSegment(text="Old speech", start_time=0.0, end_time=1.0, confidence=0.9, timestamp=time.time() - 10.0)
    cm.add_transcript(old_seg)
    cm._prune_expired_transcripts()
    assert "Old speech" not in [t.text for t in cm._transcript_history]

def test_question_detector():
    qd = QuestionDetector()

    # Question with question mark
    q1 = qd.detect("Can you explain the metric?")
    assert q1.category == IntentCategory.QUESTION
    assert q1.trigger_llm is True

    # Question word without question mark
    q2 = qd.detect("Why did you choose this model")
    assert q2.category == IntentCategory.QUESTION
    assert q2.trigger_llm is True

    # Command
    c1 = qd.detect("Explain this slide")
    assert c1.category == IntentCategory.COMMAND
    assert c1.trigger_llm is True

    # Filler noise
    f1 = qd.detect("um yeah ok")
    assert f1.category == IntentCategory.IRRELEVANT
    assert f1.trigger_llm is False

def test_llm_client_mock_mode():
    client = LLMClient(mock_mode=True)
    ans = client.generate_response(
        screen_context="Random Forest Accuracy = 91.4%",
        audio_context="Why did you choose Random Forest?",
        conversation_history="",
        current_question="Why did you choose Random Forest?"
    )
    assert "Random Forest" in ans
    assert "91.4%" in ans

def test_llm_client_offline_mock_provider():
    client = LLMClient(provider="mock")
    assert client.mock_mode is True
    ans = client.generate_response(
        screen_context="which of the following Option B: O(log N)",
        audio_context="Which option is correct?",
        conversation_history="",
        current_question="which of the following option is correct"
    )
    assert "Option B" in ans

def test_response_manager_deduplication():
    rm = ResponseManager(debounce_interval_sec=1.0)
    received = []

    # First emission
    r1 = rm.process_and_dispatch("Question 1", "Answer 1", on_response_ready=received.append)
    assert r1 is not None
    assert len(received) == 1

    # Immediate duplicate emission
    r2 = rm.process_and_dispatch("Question 1", "Answer 1", on_response_ready=received.append)
    assert r2 is None
    assert len(received) == 1  # Not duplicated!
