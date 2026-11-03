"""
Speech-to-Text (STT) module transcribing real-time human speech from audio streams.
Optimized for ultra-fast response latency (~250ms), high accuracy, and zero hallucination.
"""

from dataclasses import dataclass
import re
import time
import threading
from typing import Optional
import numpy as np

from utils.logger import logger
from utils.timing import time_it
from .vad import SpeechSegment

@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    confidence: float
    timestamp: float
    detected_language: str = "en"
    task: str = "transcribe"
    speaker: str = "Interviewer"

# Known Whisper hallucination phrases on quiet background noise or breath
HALLUCINATION_PATTERNS = {
    "thank you", "thank you.", "thank you very much", "thank you very much.",
    "thanks for watching", "thanks for watching.", "subtitles by", "please subscribe",
    "thank you for watching", "thank you for watching.", "like and subscribe",
    "subscribe", "see you in the next video", "see you next time", "good luck",
    "bye", "bye.", "okay", "so", "f", "ff", "uh", "um", "you", "a", "the",
    "www.leetcode.com", "leetcode.com", "http://leetcode.com", "https://leetcode.com",
    "http://www.leetcode.com", "https://www.leetcode.com", "www.leetcode.com.", "leetcode.com.",
    "leetcode", "leet code", "lead code", "www.leetcode", "leetcode com"
}


class SpeechToText:
    """Transcribes speech audio segments into real timestamped text with hallucination suppression."""

    def __init__(
        self,
        model_size: str = "base",
        engine: str = "faster-whisper",
        task: str = "transcribe",
        language: str = "en",
        beam_size: int = 3,
        initial_prompt: Optional[str] = "Technical interview conversation between candidate and interviewer. Questions regarding projects, experience, resume, system design, explain your project, how it operates, coding problems, algorithms, bug fixes, architecture, HireMatrix, Python, Neo4j, FastAPI.",
        gnani_api_key: Optional[str] = None,
        gnani_language_code: str = "en-IN"
    ):
        import os
        self.model_size = model_size if model_size else "base"
        self.engine_name = engine.lower() if engine else "faster-whisper"
        self.task = task.lower() if task else "transcribe"
        self.language = "en" if (not language or language.lower() in ("auto", "none")) else language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.gnani_api_key = (gnani_api_key or os.getenv("GNANI_API_KEY", "")).strip()
        self.gnani_language_code = gnani_language_code or os.getenv("GNANI_LANGUAGE_CODE", "en-IN")
        self._sr_recognizer = None
        self._transcribe_lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        """Initializes STT engine: prioritizing gnani, faster-whisper, or google based on configuration."""
        # 1. Gnani.ai Prisma STT (Indian English & regional language specialist)
        if self.engine_name in ("gnani", "gnani-ai", "vachana") or (self.gnani_api_key and self.engine_name == "auto"):
            if self.gnani_api_key:
                self.engine_name = "gnani"
                logger.info(f"Real-Time Speech-to-Text active via Gnani.ai Prisma v3.0 (Lang={self.gnani_language_code}).")
                # Also load local faster-whisper as offline fallback in case of network interruption
                try:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(self.model_size if self.model_size not in ("tiny.en", "base.en") else "base", device="cpu", compute_type="int8")
                except Exception:
                    pass
                return
            else:
                logger.warning("Gnani engine requested but GNANI_API_KEY is not set. Falling back to local faster-whisper.")

        if self.engine_name in ("faster-whisper", "auto"):
            try:
                from faster_whisper import WhisperModel
                size = self.model_size if self.model_size else "base"
                # If tiny.en or base.en is passed, normalize to base which is locally cached
                if size in ("tiny.en", "base.en"):
                    size = "base"
                logger.info(f"Loading local high-accuracy faster-whisper model ({size})...")
                self._model = WhisperModel(size, device="cpu", compute_type="int8")
                self.engine_name = "faster-whisper"
                logger.info(f"Local faster-whisper ({size}) initialized - high-accuracy speech recognition active.")
                return
            except Exception as e:
                logger.warning(f"Could not load faster-whisper model '{self.model_size}': {e}. Falling back to SpeechRecognition.")

        # SpeechRecognition (Google STT)
        try:
            import speech_recognition as sr
            self._sr_recognizer = sr.Recognizer()
            self._sr_recognizer.energy_threshold = 200
            self._sr_recognizer.dynamic_energy_threshold = True
            self._sr_recognizer.pause_threshold = 0.8
            self._sr_recognizer.non_speaking_duration = 0.3
            self.engine_name = "speech_recognition"
            logger.info("Real-Time Speech-to-Text initialized via Google SpeechRecognition engine.")
            return
        except ImportError:
            logger.warning("speech_recognition library not available.")

        self.engine_name = "none"
        logger.warning("No live STT engine available. Install speech_recognition or faster-whisper.")

    @time_it("stt_transcription")
    def transcribe(self, segment: SpeechSegment) -> Optional[TranscriptSegment]:
        """
        Transcribes a detected SpeechSegment into real words.
        Returns None if audio is unclear, silent, or noise (never generates hallucinations).
        """
        if segment is None or len(segment.audio) == 0:
            return None

        # Ignore tiny audio clips (under 0.35s) to avoid clicks/keystroke triggers
        if segment.duration_sec < 0.35:
            return None

        # Energy gate: check peak amplitude to prevent transcribing microscopic room noise
        peak_amp = float(np.max(np.abs(segment.audio)))
        if peak_amp < 0.012:
            return None

        # Normalize audio volume so Whisper receives optimal dynamic range (~ -14 dB to -20 dBFS)
        # Low microphone gain (< 0.08 peak) causes Whisper to mishear phonemes.
        if peak_amp > 0.001:
            audio_norm = (segment.audio / peak_amp * 0.92).astype(np.float32)
        else:
            audio_norm = segment.audio

        text = ""
        confidence = 0.95

        try:
            if self.engine_name == "gnani" and self.gnani_api_key:
                text = self._transcribe_gnani(audio_norm)
                # If network fails or Gnani returns empty, fall back seamlessly to faster-whisper
                if not text and hasattr(self, "_model"):
                    logger.debug("Gnani API silent/timeout, attempting local faster-whisper fallback...")
                    segments, info = self._model.transcribe(
                        audio_norm,
                        beam_size=max(3, self.beam_size),
                        language=self.language,
                        task=self.task,
                        vad_filter=False,
                        initial_prompt=self.initial_prompt
                    )
                    text = " ".join([s.text for s in segments]).strip()

            elif self.engine_name == "faster-whisper" and hasattr(self, "_model"):
                effective_beam = max(3, self.beam_size)
                segments, info = self._model.transcribe(
                    audio_norm,
                    beam_size=effective_beam,
                    language=self.language,
                    task=self.task,
                    vad_filter=False,  # vad.py already filtered & segmented speech; double VAD drops opening/closing phonemes
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    no_speech_threshold=0.5,
                    compression_ratio_threshold=2.4,
                    initial_prompt=self.initial_prompt
                )
                text_parts = [s.text for s in segments]
                text = " ".join(text_parts).strip()
                if hasattr(info, 'language_probability') and info.language_probability is not None:
                    confidence = float(info.language_probability)

            elif (self.engine_name in ("speech_recognition", "google")) and self._sr_recognizer:
                import speech_recognition as sr
                audio_float = np.clip(audio_norm, -1.0, 1.0)
                pcm_bytes = (audio_float * 32767).astype(np.int16).tobytes()

                audio_data = sr.AudioData(
                    frame_data=pcm_bytes,
                    sample_rate=16000,
                    sample_width=2
                )
                try:
                    text = self._sr_recognizer.recognize_google(audio_data, language="en-US").strip()
                except sr.UnknownValueError:
                    return None
                except sr.RequestError as e:
                    logger.warning(f"Google STT service error: {e}")
                    return None

        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return None

        clean_text = self._clean_transcript(text)
        if not clean_text:
            return None

        logger.info(f"Live voice transcribed ({self.engine_name}): '{clean_text}'")

        return TranscriptSegment(
            text=clean_text,
            start_time=segment.start_time,
            end_time=segment.end_time,
            confidence=confidence,
            timestamp=time.time(),
            detected_language=self.language,
            task=self.task,
            speaker=getattr(segment, "speaker", "Interviewer")
        )

    def _transcribe_gnani(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribes audio using Gnani.ai Prisma v3.0 REST API (specialized for Indian English & regional accents)."""
        import io
        import wave
        import requests

        try:
            pcm16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())
            wav_buffer.seek(0)

            headers = {
                "X-API-Key-ID": self.gnani_api_key
            }
            files = {
                "audio_file": ("speech.wav", wav_buffer, "audio/wav")
            }
            data = {
                "model": "prisma-v3.0",
                "language_code": self.gnani_language_code or "en-IN"
            }
            with self._transcribe_lock:
                resp = requests.post(
                "https://api.vachana.ai/stt/v3",
                headers=headers,
                files=files,
                data=data,
                timeout=5.0
            )
            if resp.status_code == 200:
                res_json = resp.json()
                t = res_json.get("transcript") or res_json.get("text") or ""
                if isinstance(t, list):
                    t = " ".join(t)
                return str(t).strip()
            else:
                logger.warning(f"Gnani STT API returned status {resp.status_code}: {resp.text[:120]}")
                return None
        except Exception as e:
            logger.warning(f"Gnani STT transcription exception: {e}")
            return None

    def _clean_transcript(self, raw_text: str) -> Optional[str]:
        """Filters hallucinations, character repetitions (e.g. 'ffffff'), and repetition loops."""
        if not raw_text:
            return None

        # 1. Strip repetitive character loops (e.g. 'ffffffffffff' -> '')
        text = re.sub(r'([a-zA-Z])\1{3,}', '', raw_text).strip()

        # 2. Strip leading punctuation artifacts, preserve terminal sentence punctuation (? .)
        text = text.lstrip(" '\"`.,-!?:;").rstrip(" '\"`-,;:")
        # 2b. Discard pure URL tokens or website domain hallucinations (e.g. www.leetcode.com)
        if re.search(r'\bleetcode\.com\b', text, re.IGNORECASE) or re.match(r'^(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.(?:com|org|net|io|edu)(?:/[^\s]*)?$', text, re.IGNORECASE):
            return None
        if not text or len(text) < 2:
            return None

        # 3. Check for exact known Whisper noise hallucinations
        lower_clean = text.lower().strip(".!?, ")
        if lower_clean in HALLUCINATION_PATTERNS:
            return None

        # 4. Check if text is composed solely of hallucination phrases (e.g. 'Thanks for watching. Please subscribe.')
        sentences = [s.strip(" '\"`.,-!?:;") for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences and all(s.lower() in HALLUCINATION_PATTERNS for s in sentences):
            logger.debug(f"Filtered out compound hallucination: '{text}'")
            return None

        # 5. Check for hallucination repetition loops (e.g. 'Thank you very much. Thank you very much')
        lower_text = text.lower()
        for phrase in ("thank you very much", "thank you", "thanks for watching", "please subscribe", "subtitles by"):
            if phrase in lower_text:
                if lower_text.count(phrase) >= 2 or len(phrase) >= len(lower_text) * 0.7:
                    logger.debug(f"Filtered out hallucination phrase loop: '{text}'")
                    return None

        # 6. Deduplicate repeating sentences (e.g. 'Good job. Good job.')
        if len(sentences) >= 2:
            unique_sentences = list(dict.fromkeys(sentences))
            if len(unique_sentences) == 1:
                first = unique_sentences[0].lower().strip(".,?! ")
                if first in HALLUCINATION_PATTERNS:
                    return None
                text = unique_sentences[0]

        # 6. Deduplicate consecutive repeated word tokens
        tokens = text.split()
        if len(tokens) >= 4:
            mid = len(tokens) // 2
            if tokens[:mid] == tokens[mid:2*mid]:
                text = " ".join(tokens[mid:])



        return text.strip()

    def update_domain_vocabulary(self, keywords: list):
        """Dynamically enriches Whisper's initial_prompt with domain and resume vocabulary."""
        if not keywords:
            return
        cleaned = [str(w).strip() for w in keywords if w and len(str(w).strip()) > 1]
        if not cleaned:
            return
        base = self.initial_prompt or "Technical coding interview, algorithms, data structures."
        # Append new keywords without excessive duplication
        current_lower = base.lower()
        new_terms = [t for t in cleaned if t.lower() not in current_lower]
        if new_terms:
            combined = f"{base} Terms: {', '.join(new_terms[:20])}."
            if len(combined) > 400:
                combined = combined[:400]
            self.initial_prompt = combined
            logger.info(f"STT domain vocabulary updated with {len(new_terms)} terms.")

