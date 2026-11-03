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

    # LeetCode domain / URL noise suppression
    u1 = qd.detect("www.leetcode.com")
    assert u1.category == IntentCategory.IRRELEVANT
    assert u1.trigger_llm is False

    u2 = qd.detect("https://leetcode.com/problems/two-sum")
    assert u2.category == IntentCategory.IRRELEVANT
    assert u2.trigger_llm is False

def test_llm_client_mock_mode():
    client = LLMClient(mock_mode=True)
    ans = client.generate_response(
        screen_context="Screen context",
        audio_context="",
        conversation_history="",
        current_question="Test query"
    )
    assert "Mock Analysis" in ans
    assert "Test query" in ans

def test_llm_client_offline_mock_provider():
    client = LLMClient(provider="mock")
    assert client.mock_mode is True
    ans = client.generate_response(
        screen_context="Screen context",
        audio_context="",
        conversation_history="",
        current_question="which of the following option is correct"
    )
    assert "Mock Analysis" in ans

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


def test_key_manager_gemini_support(tmp_path):
    from intelligence.key_manager import KeyManager
    km = KeyManager(storage_path=str(tmp_path / "test_gemini_keys.json"))

    # 1. Test auto-detection
    assert km.detect_provider("AIzaSyB1234567890ABCDEF12345") == "gemini"
    assert km.detect_provider("AIzaCustomGoogleKey12345") == "gemini"
    assert km.DEFAULT_MODELS["gemini"] == "gemini-flash-lite-latest"

    # 2. Add Gemini key
    entry = km.add_key("AIzaSyB1234567890ABCDEF12345", provider="gemini", label="My Gemini Key")
    assert entry.provider == "gemini"
    assert entry.model == "gemini-flash-lite-latest"
    assert entry.label == "My Gemini Key"
    assert entry.masked_key == "AIzaSy...2345"

    # 3. Failover rotation to Gemini
    km.keys.clear()
    k1 = km.add_key("sk-primary-key-11111111111", provider="openai")
    k2 = km.add_key("AIzaSyBackupGeminiKey22222", provider="gemini")

    active = km.get_active_key()
    assert active.key == k1.key

    # Report 429 on primary
    next_k = km.report_failure(k1.key, "429 Rate Limit Exceeded", status_code=429)
    assert next_k is not None
    assert next_k.key == k2.key
    assert next_k.provider == "gemini"


def test_llm_client_gemini_client_configuration(tmp_path):
    from intelligence.key_manager import KeyManager
    km = KeyManager(storage_path=str(tmp_path / "test_gemini_client.json"))
    km.keys.clear()

    entry = km.add_key("AIzaSyTestGeminiKey123456789", provider="gemini")
    client = LLMClient(mgr=km)

    # Verify client setup for Gemini routes to Google's official OpenAI-compatible endpoint
    openai_client = client._get_or_create_client(entry)
    assert openai_client is not None
    assert "generativelanguage.googleapis.com" in str(openai_client.base_url)
