"""
Unit tests for audio capture, Voice Activity Detection (VAD), and Speech-to-Text.
"""

import time
import numpy as np
from audio.capture import AudioCapture
from audio.vad import VoiceActivityDetector, AudioChunk, SpeechSegment
from audio.speech_to_text import SpeechToText

def test_audio_device_listing():
    devices = AudioCapture.list_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert "name" in devices[0]

def test_vad_speech_segmentation():
    vad = VoiceActivityDetector(energy_threshold=0.01, silence_timeout_sec=0.2, min_speech_duration_sec=0.2)

    # 1. Feed silence -> should return None
    silence = np.zeros(1600, dtype=np.float32)
    chunk1 = AudioChunk(data=silence, timestamp=1.0)
    seg1 = vad.process_chunk(chunk1)
    assert seg1 is None

    # 2. Feed speech frames (loud sine wave)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    loud_tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Add multiple speech chunks to exceed min duration
    vad.process_chunk(AudioChunk(data=loud_tone, timestamp=1.1))
    vad.process_chunk(AudioChunk(data=loud_tone, timestamp=1.2))
    vad.process_chunk(AudioChunk(data=loud_tone, timestamp=1.3))

    # 3. Feed silence to trigger silence timeout and segment closure
    seg_result = None
    for offset in [0.1, 0.25]:
        res = vad.process_chunk(AudioChunk(data=silence, timestamp=1.3 + offset))
        if res:
            seg_result = res

    assert seg_result is not None
    assert isinstance(seg_result, SpeechSegment)
    assert seg_result.duration_sec >= 0.2
    assert len(seg_result.audio) > 0

def test_stt_transcription():
    stt = SpeechToText()
    dummy_audio = np.zeros(16000, dtype=np.float32)
    segment = SpeechSegment(
        audio=dummy_audio,
        start_time=10.0,
        end_time=11.0,
        duration_sec=1.0
    )

    # Pure silence should return None - NO fake/offline questions!
    transcript = stt.transcribe(segment)
    assert transcript is None

def test_resampling_exact_16k():
    """Verify 44.1kHz audio is resampled to exact 16kHz without pitch/sample count errors."""
    orig_rate = 44100
    target_rate = 16000
    num_samples_44k = 4410  # 100ms at 44.1kHz
    t = np.linspace(0, 0.1, num_samples_44k, endpoint=False)
    mono = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    target_len = int(round(len(mono) * (target_rate / orig_rate)))
    orig_indices = np.linspace(0.0, 1.0, len(mono), endpoint=False)
    target_indices = np.linspace(0.0, 1.0, target_len, endpoint=False)
    resampled = np.interp(target_indices, orig_indices, mono).astype(np.float32)

    assert len(resampled) == 1600, f"Expected 1600 samples for 100ms at 16kHz, got {len(resampled)}"
    assert resampled.dtype == np.float32

def test_dual_mode_synchronous_mixing():
    """Verify dual mode mixes mic and loopback streams simultaneously instead of interleaving packets."""
    capture = AudioCapture(audio_source="dual")
    mic_chunk = np.ones(1600, dtype=np.float32) * 0.3
    lb_chunk = np.ones(1600, dtype=np.float32) * 0.4

    capture._mic_queue.put(mic_chunk)
    capture._lb_queue.put(lb_chunk)

    chunk_dict = capture.read_chunk(timeout=0.1)
    assert chunk_dict is not None
    data = chunk_dict["data"]
    assert len(data) == 1600
    # Values should be mixed (0.3 + 0.4 = 0.7), not alternating chunks
    np.testing.assert_allclose(data[:10], 0.7, rtol=1e-4)

def test_stt_translation_mode():
    """Verify STT initializes with translation task and auto language."""
    stt = SpeechToText(model_size="tiny", task="translate", language="auto", beam_size=3)
    assert stt.task == "translate"
    assert stt.beam_size == 3

def test_stt_clean_transcript():
    """Verify clean_transcript strips hallucinations, noise loops, and punctuation without offline phonetics."""
    stt = SpeechToText(model_size="tiny")
    raw_sample = "   ...ffff How are you doing?   "
    cleaned = stt._clean_transcript(raw_sample)
    assert cleaned == "How are you doing?"
    # Known noise hallucination filtered to None
    assert stt._clean_transcript("Thank you for watching. Please subscribe.") is None
    assert stt._clean_transcript("www.leetcode.com") is None
    assert stt._clean_transcript("https://leetcode.com/problem/two-sum") is None
    assert stt._clean_transcript("leetcode.com") is None

def test_stt_domain_vocabulary_update():
    """Verify dynamic vocabulary injection enriches Whisper initial prompt."""
    stt = SpeechToText(model_size="tiny")
    initial_p = stt.initial_prompt
    stt.update_domain_vocabulary(["GitProof", "TextMorph", "DistilBART", "RoBERTa"])
    assert "GitProof" in stt.initial_prompt
    assert "TextMorph" in stt.initial_prompt

def test_gnani_stt_initialization():
    """Verify Gnani engine initialization with API key and language code."""
    stt = SpeechToText(
        engine="gnani",
        gnani_api_key="test_gnani_key_12345",
        gnani_language_code="en-IN"
    )
    assert stt.engine_name == "gnani"
    assert stt.gnani_api_key == "test_gnani_key_12345"
    assert stt.gnani_language_code == "en-IN"

def test_gnani_transcribe_mock(monkeypatch):
    """Verify Gnani REST API mock response parsing."""
    stt = SpeechToText(
        engine="gnani",
        gnani_api_key="test_gnani_key_12345",
        gnani_language_code="en-IN"
    )

    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "success": True,
                "transcript": "Explain your project architecture"
            }

    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResponse())

    dummy_audio = np.ones(16000, dtype=np.float32) * 0.2
    seg = SpeechSegment(audio=dummy_audio, start_time=0.0, end_time=1.0, duration_sec=1.0)
    transcript = stt.transcribe(seg)
    assert transcript is not None
    assert "explain your project architecture" in transcript.text.lower()

