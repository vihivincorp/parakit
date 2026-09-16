"""
Speech-to-Text (STT) module transcribing real-time human speech from audio streams.
Optimized for ultra-fast response latency (~250ms), high accuracy, and zero hallucination.
"""

from dataclasses import dataclass
import re
import time
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
    "bye", "bye.", "okay", "so", "f", "ff", "uh", "um", "you", "a", "the"
}

# Technical term phonetic & soundalike corrections
PHONETIC_CORRECTIONS = [
    (re.compile(r'\b(?:higher|hire)\s+(?:matrix|metrics)\b', re.IGNORECASE), "HireMatrix"),
    (re.compile(r'\bneo\s*(?:4|for)\s*j\b', re.IGNORECASE), "Neo4j"),
    (re.compile(r'\b(?:fast|first)\s+api\b', re.IGNORECASE), "FastAPI"),
    (re.compile(r'\b(?:lead|leet)\s+code\b', re.IGNORECASE), "LeetCode"),
    (re.compile(r'\b(?:cooper|kuber)\s*net(?:ies|ees|es)\b', re.IGNORECASE), "Kubernetes"),
    (re.compile(r'\bpie\s+torch\b', re.IGNORECASE), "PyTorch"),
    (re.compile(r'\b(?:get|git)\s+proof\b', re.IGNORECASE), "GitProof"),
    (re.compile(r'\btext\s+morph\b', re.IGNORECASE), "TextMorph"),
    (re.compile(r'\b(?:dyke|dike)\s*stra\b', re.IGNORECASE), "Dijkstra"),
    (re.compile(r'\b(?:too|to)\s+sum\b', re.IGNORECASE), "Two Sum"),
    (re.compile(r'\bbreath\s+first\s+search\b', re.IGNORECASE), "breadth first search"),
    (re.compile(r'\bdeath\s+first\s+search\b', re.IGNORECASE), "depth first search"),
    (re.compile(r'\bno\s*sequel\b', re.IGNORECASE), "NoSQL"),
    (re.compile(r'\bsci[\s-]?kit[\s-]?learn\b', re.IGNORECASE), "Scikit-learn"),
    (re.compile(r'\bdistil[\s-]?bart\b', re.IGNORECASE), "DistilBART"),
    (re.compile(r'\bro[\s-]?berta\b', re.IGNORECASE), "RoBERTa"),
]

class SpeechToText:
    """Transcribes speech audio segments into real timestamped text with hallucination suppression."""

    def __init__(
        self,
        model_size: str = "base",
        engine: str = "faster-whisper",
        task: str = "transcribe",
        language: str = "en",
        beam_size: int = 1,
        initial_prompt: Optional[str] = "Technical coding interview, HireMatrix, algorithms, data structures, Neo4j, FastAPI, LeetCode, programming questions."
    ):
        self.model_size = model_size if model_size else "base"
        self.engine_name = engine.lower() if engine else "faster-whisper"
        self.task = task.lower() if task else "transcribe"
        self.language = "en" if (not language or language.lower() in ("auto", "none")) else language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self._sr_recognizer = None
        self._init_engine()

    def _init_engine(self):
        """Initializes STT engine: prioritizing faster-whisper or google based on configuration."""
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
        if peak_amp < 0.020:
            return None

        text = ""
        confidence = 0.95

        try:
            if self.engine_name == "faster-whisper" and hasattr(self, "_model"):
                segments, info = self._model.transcribe(
                    segment.audio,
                    beam_size=self.beam_size,
                    language=self.language,
                    task=self.task,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=400, threshold=0.35),
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
                audio_float = np.clip(segment.audio, -1.0, 1.0)
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

    def _clean_transcript(self, raw_text: str) -> Optional[str]:
        """Filters hallucinations, character repetitions (e.g. 'ffffff'), and repetition loops."""
        if not raw_text:
            return None

        # 1. Strip repetitive character loops (e.g. 'ffffffffffff' -> '')
        text = re.sub(r'([a-zA-Z])\1{3,}', '', raw_text).strip()

        # 2. Strip leading punctuation artifacts, preserve terminal sentence punctuation (? .)
        text = text.lstrip(" '\"`.,-!?:;").rstrip(" '\"`-,;:")
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

        # 7. Apply phonetic soundalike normalization for tech terms & project names
        for pattern, replacement in PHONETIC_CORRECTIONS:
            text = pattern.sub(replacement, text)

        return text.strip()

    def update_domain_vocabulary(self, keywords: list):
        """Dynamically enriches Whisper's initial_prompt with domain and resume vocabulary."""
        if not keywords:
            return
        cleaned = [str(w).strip() for w in keywords if w and len(str(w).strip()) > 1]
        if not cleaned:
            return
        base = self.initial_prompt or "Technical coding interview, algorithms, LeetCode."
        # Append new keywords without excessive duplication
        current_lower = base.lower()
        new_terms = [t for t in cleaned if t.lower() not in current_lower]
        if new_terms:
            combined = f"{base} Terms: {', '.join(new_terms[:20])}."
            if len(combined) > 400:
                combined = combined[:400]
            self.initial_prompt = combined
            logger.info(f"STT domain vocabulary updated with {len(new_terms)} terms.")

