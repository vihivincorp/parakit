"""
Configuration module for the Multimodal Real-Time AI Desktop Assistant.
Handles paths, model options, thresholds, audio/screen settings, and environment variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class ScreenConfig:
    monitor_index: int = 1  # 1 is primary monitor in mss
    capture_interval_sec: float = 2.0
    auto_screen_capture: bool = False  # False: Screen OCR runs strictly on-demand when clicked, not continuously
    similarity_threshold: float = 0.95  # Skip re-processing if screen text unchanged
    preprocess_grayscale: bool = True
    preprocess_contrast: float = 1.5
    preprocess_resize_factor: float = 1.0
    crop_region: tuple = None  # (left, top, right, bottom) or None for full screen
    tesseract_cmd: str = os.getenv("TESSERACT_PATH", "")

@dataclass
class AudioConfig:
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    channels: int = 1
    chunk_duration_sec: float = 0.1  # 100ms chunks for rapid 100ms VAD resolution
    device_index: int = None  # None selects system default input / microphone
    use_wasapi_loopback: bool = True  # Capture video call / meeting sound from speakers
    audio_source: str = os.getenv("AUDIO_SOURCE", "loopback")  # 'loopback' (meeting call focus), 'dual' (mic + call), 'mic' (mic only)
    vad_energy_threshold: float = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.005"))  # Sensitive energy threshold for clear speech capture
    vad_silence_duration_sec: float = float(os.getenv("VAD_SILENCE_SEC", "0.80"))  # 0.8s silence window to prevent cut-off sentences
    stt_model: str = os.getenv("STT_MODEL", "base")  # High-accuracy local int8 Whisper base model (cached locally)
    stt_engine: str = os.getenv("STT_ENGINE", "faster-whisper")  # 'faster-whisper' or 'google'
    stt_task: str = os.getenv("STT_TASK", "transcribe")  # 'transcribe'
    stt_language: str = os.getenv("STT_LANGUAGE", "en")  # 'en'
    stt_beam_size: int = int(os.getenv("STT_BEAM_SIZE", "3"))  # Beam size 3 for high-accuracy recognition
    initial_prompt: str = os.getenv(
        "STT_INITIAL_PROMPT",
        "Technical interview conversation between candidate and interviewer. Questions regarding projects, experience, resume, system design, coding problems, algorithms, bug fixes, architecture, HireMatrix, Python, Neo4j, FastAPI."
    )
    gnani_api_key: str = os.getenv("GNANI_API_KEY", "")
    gnani_language_code: str = os.getenv("GNANI_LANGUAGE_CODE", "en-IN")  # 'en-IN', 'indic-auto', 'hi-IN', etc.

@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "openai").lower()  # 'gemini', 'openai', 'groq', 'anthropic', 'openrouter'
    api_key: str = (
        os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if os.getenv("LLM_PROVIDER", "").lower() in ("gemini", "google")
        else (
            os.getenv("GROQ_API_KEY", "")
            if os.getenv("LLM_PROVIDER", "").lower() == "groq"
            else (
                os.getenv("OPENROUTER_API_KEY", "")
                if os.getenv("LLM_PROVIDER", "").lower() == "openrouter"
                else (os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", ""))
            )
        )
    )
    model_name: str = os.getenv(
        "ASSISTANT_MODEL",
        "gemini-1.5-flash" if os.getenv("LLM_PROVIDER", "").lower() in ("gemini", "google")
        else (
            "llama-3.3-70b-versatile" if os.getenv("LLM_PROVIDER", "").lower() == "groq"
            else (
                "openai/gpt-4o-mini" if os.getenv("LLM_PROVIDER", "").lower() == "openrouter"
                else ("gpt-4o-mini" if os.getenv("LLM_PROVIDER", "openai").lower() == "openai" else "claude-3-5-sonnet-latest")
            )
        )
    )
    max_tokens: int = int(os.getenv("MAX_TOKENS", "400"))
    temperature: float = 0.2
    mock_mode: bool = False  # Set True to test pipeline without API key

@dataclass
class ContextConfig:
    speech_window_sec: float = 20.0  # Keep last 20 seconds of transcript so stale speech does not linger
    max_history_turns: int = 2       # Keep only last 2 turns so assistant does not anchor to previous topics
    max_screen_chars: int = 2500

@dataclass
class UIConfig:
    width: int = 920
    height: int = 660
    min_height: int = 660
    max_height: int = 660
    opacity: float = float(os.getenv("OVERLAY_TRANSPARENCY", "0.92"))
    bg_color: str = "#12141A"
    card_color: str = "#1E222D"
    accent_color: str = "#4F46E5"
    text_color: str = "#F3F4F6"
    subtext_color: str = "#9CA3AF"
    exclude_from_capture: bool = os.getenv("EXCLUDE_FROM_CAPTURE", "true").lower() in ("true", "1", "yes")
    hotkey_toggle: str = os.getenv("HOTKEY_TOGGLE", "alt+m")

@dataclass
class AppConfig:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    debug: bool = True

# Global config singleton
cfg = AppConfig()
