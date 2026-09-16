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
        energy_threshold: float = 0.010,
        silence_timeout_sec: float = 0.80,
        min_speech_duration_sec: float = 0.40,
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
        self._noise_floor: float = 0.004

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
            self._noise_floor = 0.96 * self._noise_floor + 0.04 * rms

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

                if total_duration >= self.min_speech_duration_sec and self._speech_buffer:
                    concatenated_audio = np.concatenate(self._speech_buffer)
                    # Determine dominant speaker during this utterance
                    dominant_speaker = "Interviewer"
                    if self._speech_speakers:
                        candidate_count = self._speech_speakers.count("Candidate")
                        interviewer_count = self._speech_speakers.count("Interviewer")
                        dominant_speaker = "Candidate" if candidate_count > interviewer_count else "Interviewer"

                    segment = SpeechSegment(
                        audio=concatenated_audio,
                        start_time=self._speech_start_time,
                        end_time=self._last_speech_time,
                        duration_sec=round(total_duration, 2),
                        speaker=dominant_speaker
                    )
                    self._speech_buffer = []
                    self._speech_speakers = []
                    self._pre_buffer.clear()
                    return segment
                else:
                    # Too short, likely click or noise
                    self._speech_buffer = []
                    self._speech_speakers = []
                    self._pre_buffer.clear()
                    return None
        else:
            # Pure silence - maintain pre-speech rolling buffer
            self._pre_buffer.append(audio_data)

        return None

    def reset(self):
        self._in_speech = False
        self._speech_buffer = []
        self._speech_speakers = []
        self._pre_buffer.clear()
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
