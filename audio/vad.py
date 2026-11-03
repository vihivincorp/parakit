"""
Voice Activity Detection (VAD) module for speech segmentation and silence suppression.
"""

from dataclasses import dataclass
import time
from typing import Optional, List
import numpy as np

from utils.logger import logger
from utils.timing import time_it

@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float
    speaker: str = "Interviewer"  # 'Interviewer' (meeting call / loopback), 'Candidate' (mic)

@dataclass
class SpeechSegment:
    audio: np.ndarray
    start_time: float
    end_time: float
    duration_sec: float
    speaker: str = "Interviewer"

class VoiceActivityDetector:
    """
    Detects voiced audio frames, strips silence, and segments continuous speech utterances.
    Uses adaptive RMS energy with dynamic noise floor tracking, pluggable with Silero/WebRTC.
    """

    def __init__(
        self,
        energy_threshold: float = 0.005,
        silence_timeout_sec: float = 0.80,
        min_speech_duration_sec: float = 0.35,
        pre_speech_chunks: int = 4
    ):
        self.energy_threshold = energy_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.pre_speech_chunks = pre_speech_chunks

        import collections
        self._pre_buffer = collections.deque(maxlen=pre_speech_chunks)
        self._in_speech = False
        self._speech_buffer: List[np.ndarray] = []
        self._speech_speakers: List[str] = []
        self._speech_start_time: float = 0.0
        self._last_speech_time: float = 0.0
        self._noise_floor: float = 0.0005

    @time_it("vad_processing")
    def process_chunk(self, chunk: AudioChunk) -> Optional[SpeechSegment]:
        """
        Processes an incoming audio chunk.
        
        Returns:
            SpeechSegment when an utterance completes, or None if still accumulating or silent.
        """
        audio_data = chunk.data
        if len(audio_data) == 0:
            return None

        # Compute Root Mean Square (RMS) energy
        rms = float(np.sqrt(np.mean(np.square(audio_data))))
        current_time = chunk.timestamp
        current_speaker = getattr(chunk, "speaker", "Interviewer")

        # Adaptive noise floor tracking during silence
        if not self._in_speech and rms < self.energy_threshold:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        dynamic_threshold = max(self.energy_threshold, self._noise_floor * 1.8)
        is_speech = rms >= dynamic_threshold

        if is_speech:
            if not self._in_speech:
                # Speech onset - prepend pre-speech buffer so opening syllables are preserved
                self._in_speech = True
                self._speech_start_time = current_time
                self._speech_buffer = list(self._pre_buffer) + [audio_data]
                self._speech_speakers = [current_speaker] * (len(self._pre_buffer) + 1)
                self._pre_buffer.clear()
            else:
                self._speech_buffer.append(audio_data)
                self._speech_speakers.append(current_speaker)
            self._last_speech_time = current_time
            return None

        elif self._in_speech:
            # In speech, currently quiet chunk
            self._speech_buffer.append(audio_data)
            self._speech_speakers.append(current_speaker)
            silence_duration = current_time - self._last_speech_time

            if silence_duration >= self.silence_timeout_sec:
                # Speech ended!
                self._in_speech = False
                total_duration = current_time - self._speech_start_time

                if total_duration >= self.min_speech_duration_sec:
                    combined_audio = np.concatenate(self._speech_buffer)
                    # Majority vote on speaker attribution across voiced frames
                    interviewer_count = sum(1 for s in self._speech_speakers if s == "Interviewer")
                    candidate_count = len(self._speech_speakers) - interviewer_count
                    dominant_speaker = "Interviewer" if interviewer_count >= candidate_count else "Candidate"

                    segment = SpeechSegment(
                        audio=combined_audio,
                        start_time=self._speech_start_time,
                        end_time=current_time,
                        duration_sec=total_duration,
                        speaker=dominant_speaker
                    )
                    self._speech_buffer.clear()
                    self._speech_speakers.clear()
                    return segment
                else:
                    self._speech_buffer.clear()
                    self._speech_speakers.clear()
                    return None
            return None

        else:
            # Silent frame, retain in rolling pre-speech buffer
            self._pre_buffer.append(audio_data)
            return None

    def reset(self):
        """Resets detector state."""
        self._in_speech = False
        self._speech_buffer.clear()
        self._speech_speakers.clear()
        self._pre_buffer.clear()
