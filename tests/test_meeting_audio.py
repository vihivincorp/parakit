"""
Unit tests for Meeting Audio Focus, WASAPI Loopback switching, rapid VAD cutoff, and Cheat-Sheet LLM Prompting.
"""

import time
import pytest
import numpy as np
from audio.capture import AudioCapture
from audio.vad import VoiceActivityDetector, AudioChunk, SpeechSegment
from intelligence.llm_client import LLMClient
from ui.overlay import DesktopOverlay
from audio.speech_to_text import SpeechToText

def test_meeting_audio_source_switching():
    capture = AudioCapture(audio_source="loopback")
    assert capture.audio_source == "loopback"
    assert capture.use_wasapi_loopback is True

    capture.set_audio_source("dual")
    assert capture.audio_source == "dual"
    assert capture.use_wasapi_loopback is True

    capture.set_audio_source("mic")
    assert capture.audio_source == "mic"
    assert capture.use_wasapi_loopback is False

    capture.set_audio_source("loopback")
    assert capture.audio_source == "loopback"
    assert capture.use_wasapi_loopback is True

def test_meeting_vad_rapid_silence_detection():
    # Rapid VAD test with 0.65s silence timeout
    vad = VoiceActivityDetector(
        energy_threshold=0.005,
        silence_timeout_sec=0.65,
        min_speech_duration_sec=0.3
    )

    t = np.linspace(0, 0.1, 1600, endpoint=False)
    voice_frame = (np.sin(2 * np.pi * 350 * t) * 0.3).astype(np.float32)
    silence_frame = np.zeros(1600, dtype=np.float32)

    # 1. 400ms of voice
    vad.process_chunk(AudioChunk(data=voice_frame, timestamp=1.0))
    vad.process_chunk(AudioChunk(data=voice_frame, timestamp=1.1))
    vad.process_chunk(AudioChunk(data=voice_frame, timestamp=1.2))
    vad.process_chunk(AudioChunk(data=voice_frame, timestamp=1.3))

    # 2. Feed silence at 100ms steps
    seg = None
    timestamps = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for ts in timestamps:
        res = vad.process_chunk(AudioChunk(data=silence_frame, timestamp=ts))
        if res:
            seg = res
            break

    assert seg is not None, "VAD must produce SpeechSegment once silence timeout expires"
    assert isinstance(seg, SpeechSegment)
    assert seg.duration_sec >= 0.3

def test_llm_meeting_cheat_sheet_prompt():
    prompt = LLMClient.SYSTEM_PROMPT.lower()
    assert "meeting" in prompt
    assert "bullet" in prompt
    assert "bold" in prompt
    assert "interview" in prompt

    client = LLMClient(mock_mode=True)
    assert client.max_tokens >= 400

def test_overlay_audio_mode_toggle():
    try:
        overlay = DesktopOverlay(width=400, height=300, opacity=0.50, exclude_from_capture=False)
    except Exception as e:
        pytest.skip(f"Tkinter not available or window display conflict: {e}")
        return
    try:
        modes_received = []
        overlay.on_audio_source_change = lambda m: modes_received.append(m)

        # Initially at index 0 ('loopback')
        assert overlay.audio_modes[overlay.current_audio_mode_idx][0] == "loopback"

        # Cycle 1: -> dual
        overlay._cycle_audio_mode()
        assert overlay.audio_modes[overlay.current_audio_mode_idx][0] == "dual"

        # Cycle 2: -> mic
        overlay._cycle_audio_mode()
        assert overlay.audio_modes[overlay.current_audio_mode_idx][0] == "mic"

        # Cycle 3: -> loopback
        overlay._cycle_audio_mode()
        assert overlay.audio_modes[overlay.current_audio_mode_idx][0] == "loopback"

        assert modes_received == ["dual", "mic", "loopback"]
    finally:
        overlay.close()

def test_speaker_attribution_and_universal_screen():
    # 1. Test speaker tracking in VAD
    vad = VoiceActivityDetector(energy_threshold=0.01, silence_timeout_sec=0.2, min_speech_duration_sec=0.2)
    silence = np.zeros(1600, dtype=np.float32)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    loud_tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Feed chunks with candidate speaker
    vad.process_chunk(AudioChunk(data=loud_tone, timestamp=1.0, speaker="Candidate"))
    vad.process_chunk(AudioChunk(data=loud_tone, timestamp=1.1, speaker="Candidate"))
    vad.process_chunk(AudioChunk(data=loud_tone, timestamp=1.2, speaker="Candidate"))
    seg = vad.process_chunk(AudioChunk(data=silence, timestamp=1.45, speaker="Candidate"))

    assert seg is not None
    assert seg.speaker == "Candidate"

    # 2. Test universal screen mock reasoning (MCQ, Terminal, Coding)
    client = LLMClient(mock_mode=True)
    mcq_ans = client._mock_reasoning("What is the answer?", "Which of the following sorting algorithms runs in O(N log N)?\nOption A: Bubble Sort\nOption B: Quick Sort", "")
    assert "Option B" in mcq_ans or "Correct Answer" in mcq_ans

    err_ans = client._mock_reasoning("Solve this error", "Traceback (most recent call last):\nModuleNotFoundError: No module named 'foo'", "")
    assert "Fix" in err_ans or "Root Cause" in err_ans

def test_stt_hallucination_and_repetition_suppression():
    stt = SpeechToText()
    
    # 1. Repeating Whisper hallucination loops must be suppressed
    assert stt._clean_transcript("Thank you very much. Thank you very much. Thank you very much") is None
    assert stt._clean_transcript("Thanks for watching. Please subscribe.") is None
    assert stt._clean_transcript("Thank you. Thank you.") is None

    # 2. Genuine speech questions must be preserved
    legit_q = "What is the time complexity of quicksort?"
    assert stt._clean_transcript(legit_q) == legit_q

    # 3. Repeated non-hallucination sentences get deduplicated
    assert stt._clean_transcript("Good job. Good job.") == "Good job"


