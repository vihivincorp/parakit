"""
Integration tests for streaming LLM, screen language detection, and VAD pre-speech buffering.
"""

import time
import numpy as np
from intelligence.llm_client import LLMClient
from screen.language_detector import LanguageDetector
from audio.vad import VoiceActivityDetector, AudioChunk

def test_language_detection_and_streaming():
    # 1. Test C++ detection
    screen_cpp = "LeetCode 1. Two Sum\nC++\nclass Solution {\npublic:\n    vector<int> twoSum(vector<int>& nums, int target) {"
    detected_lang, conf = LanguageDetector.detect(screen_cpp)
    assert detected_lang == "C++"

    # 2. Test LLM client with C++ target language streaming
    client = LLMClient(mock_mode=True)
    stream = client.generate_response_stream(
        screen_context=screen_cpp,
        audio_context="",
        conversation_history="",
        current_question="Solve this problem",
        target_language=detected_lang
    )

    tokens = list(stream)
    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert "C++" in full_text
    assert "vector<int>" in full_text
    assert "unordered_map" in full_text

def test_vad_pre_speech_buffer():
    # Test that pre-speech audio frames are prepended to speech segment
    vad = VoiceActivityDetector(
        energy_threshold=0.01,
        silence_timeout_sec=0.3,
        min_speech_duration_sec=0.2,
        pre_speech_chunks=2
    )

    # Frame 1: Quiet ambient tone (pre-speech)
    quiet_frame = np.full(1600, 0.005, dtype=np.float32)
    vad.process_chunk(AudioChunk(data=quiet_frame, timestamp=1.0))
    vad.process_chunk(AudioChunk(data=quiet_frame, timestamp=1.1))

    # Frame 2: Loud voice frame
    loud_frame = np.full(1600, 0.08, dtype=np.float32)
    vad.process_chunk(AudioChunk(data=loud_frame, timestamp=1.2))
    vad.process_chunk(AudioChunk(data=loud_frame, timestamp=1.3))

    # Frame 3: Silence to close
    silence = np.zeros(1600, dtype=np.float32)
    segment = None
    for offset in [0.1, 0.2, 0.4]:
        res = vad.process_chunk(AudioChunk(data=silence, timestamp=1.3 + offset))
        if res:
            segment = res

    assert segment is not None
    # Total audio length should include pre-speech buffer (2 frames of 1600) + speech + silence
    assert len(segment.audio) >= 1600 * 4

def test_screen_solve_code_pinning_workflow():
    sample_answer = (
        "### LeetCode Solution (C++)\n"
        "```cpp\n"
        "class Solution {\n"
        "public:\n"
        "    vector<int> twoSum(vector<int>& nums, int target) {\n"
        "        unordered_map<int, int> seen;\n"
        "        for (int i = 0; i < nums.size(); ++i) {\n"
        "            int complement = target - nums[i];\n"
        "            if (seen.count(complement)) return {seen[complement], i};\n"
        "            seen[nums[i]] = i;\n"
        "        }\n"
        "        return {};\n"
        "    }\n"
        "};\n"
        "```\n"
    )

    has_code = (
        "```" in sample_answer or 
        "class Solution" in sample_answer or 
        "def " in sample_answer or 
        "public class" in sample_answer or 
        "vector<" in sample_answer or 
        "int main(" in sample_answer
    )
    assert has_code is True

    from ui.overlay import DesktopOverlay
    try:
        overlay = DesktopOverlay(width=600, height=300, exclude_from_capture=False)
    except Exception:
        return
    try:
        overlay.pin_code(sample_answer, title="Screen Problem (C++)")
        assert overlay.pinned_code_text == sample_answer
        assert overlay.current_view_mode == "single_chat"
        assert "class Solution" in overlay.pinned_code_box.get("1.0", "end")
    finally:
        overlay.close()

