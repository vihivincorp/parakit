"""
Audio capture module using sounddevice for Windows microphone and pyaudiowpatch for WASAPI loopback.
Enables dedicated capture of meeting/video call participants (speakers) or simultaneous dual capture.
"""

import queue
import time
from typing import Optional, List, Dict
import numpy as np
import sounddevice as sd

from utils.logger import logger
from utils.timing import time_it

class AudioCapture:
    """Captures real-time audio from meeting/video call speakers and/or microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_sec: float = 0.1,
        device_index: Optional[int] = None,
        use_wasapi_loopback: bool = True,
        audio_source: str = "dual"  # 'dual' (mic + call), 'loopback' (meeting call focus), 'mic' (mic only)
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = int(sample_rate * chunk_duration_sec)
        self.device_index = device_index
        self.use_wasapi_loopback = use_wasapi_loopback
        self.audio_source = audio_source

        self.audio_queue: queue.Queue = queue.Queue(maxsize=120)
        self._mic_queue: queue.Queue = queue.Queue(maxsize=30)
        self._lb_queue: queue.Queue = queue.Queue(maxsize=30)
        self._mic_stream: Optional[sd.InputStream] = None
        self._mic_rate: int = sample_rate
        self._pa = None
        self._lb_stream = None
        self._lb_rate = 48000
        self._lb_channels = 2
        self._is_running = False

    @staticmethod
    def list_devices() -> List[Dict]:
        """Lists all available audio input/output devices."""
        devices = sd.query_devices()
        results = []
        for i, dev in enumerate(devices):
            results.append({
                "index": i,
                "name": dev["name"],
                "hostapi": dev["hostapi"],
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"]
            })
        return results

    def _mic_callback(self, indata, frames, time_info, status):
        """Callback for microphone audio frames."""
        if not self._is_running:
            return
        data_copy = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

        # Resample to target self.sample_rate if device required a different native sample rate
        if self._mic_rate != self.sample_rate and len(data_copy) > 0:
            target_len = int(round(len(data_copy) * (self.sample_rate / self._mic_rate)))
            if target_len > 0:
                orig_indices = np.linspace(0.0, 1.0, len(data_copy), endpoint=False)
                target_indices = np.linspace(0.0, 1.0, target_len, endpoint=False)
                data_copy = np.interp(target_indices, orig_indices, data_copy).astype(np.float32)

        if self.audio_source == "dual":
            self._enqueue_to_queue(self._mic_queue, data_copy)
        else:
            self._enqueue_frame(data_copy, speaker="Candidate")

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        """Callback for speaker/meeting loopback audio frames."""
        import pyaudiowpatch as pyaudio
        if not self._is_running or not in_data:
            return (in_data, pyaudio.paContinue)

        try:
            raw = np.frombuffer(in_data, dtype=np.float32)
            if self._lb_channels > 1:
                raw = raw.reshape(-1, self._lb_channels)
                mono = np.mean(raw, axis=1)
            else:
                mono = raw

            # Clean linear interpolation resampling to exact sample_rate (e.g. 16000 Hz)
            if self._lb_rate != self.sample_rate and len(mono) > 0:
                target_len = int(round(len(mono) * (self.sample_rate / self._lb_rate)))
                if target_len > 0:
                    orig_indices = np.linspace(0.0, 1.0, len(mono), endpoint=False)
                    target_indices = np.linspace(0.0, 1.0, target_len, endpoint=False)
                    resampled = np.interp(target_indices, orig_indices, mono).astype(np.float32)
                else:
                    resampled = np.array([], dtype=np.float32)
            else:
                resampled = mono.astype(np.float32)

            # Route to dual buffer or standard single audio queue
            if self.audio_source == "dual":
                self._enqueue_to_queue(self._lb_queue, resampled)
            else:
                self._enqueue_frame(resampled, speaker="Interviewer")
        except Exception as e:
            logger.debug(f"Loopback conversion error: {e}")

        return (in_data, pyaudio.paContinue)

    def _enqueue_to_queue(self, q: queue.Queue, data: np.ndarray):
        """Helper to thread-safely insert raw audio frame into a buffer queue, dropping oldest if full."""
        try:
            q.put_nowait(data)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(data)
            except queue.Empty:
                pass

    def _enqueue_frame(self, data: np.ndarray, speaker: str = "Interviewer"):
        """Helper to thread-safely insert audio frame into main queue with speaker attribution."""
        try:
            self.audio_queue.put_nowait({
                "data": data,
                "timestamp": time.time(),
                "speaker": speaker
            })
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait({
                    "data": data,
                    "timestamp": time.time(),
                    "speaker": speaker
                })
            except queue.Empty:
                pass

    def start(self):
        """Starts audio recording stream based on configured audio_source."""
        if self._is_running:
            return

        self._is_running = True

        # 1. Start Video Call / Meeting Loopback Stream (Speakers)
        loopback_success = False
        if self.audio_source in ("loopback", "dual") or self.use_wasapi_loopback:
            try:
                import pyaudiowpatch as pyaudio
                self._pa = pyaudio.PyAudio()
                loopback = self._pa.get_default_wasapi_loopback()
                self._lb_rate = int(loopback['defaultSampleRate'])
                self._lb_channels = int(loopback['maxInputChannels'])
                lb_block = int(self._lb_rate * (self.chunk_samples / self.sample_rate))

                self._lb_stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=self._lb_channels,
                    rate=self._lb_rate,
                    input=True,
                    input_device_index=loopback['index'],
                    frames_per_buffer=lb_block,
                    stream_callback=self._loopback_callback
                )
                self._lb_stream.start_stream()
                loopback_success = True
                logger.info(f"Meeting/Call audio capture active from: {loopback['name']} ({self._lb_rate}Hz)")
            except Exception as e:
                logger.warning(f"Could not initialize WASAPI loopback: {e}")

        # 2. Start Microphone Stream (dual, mic mode, or automatic fallback if loopback unavailable)
        need_mic = (self.audio_source in ("mic", "dual")) or (self.audio_source == "loopback" and not loopback_success)
        if need_mic:
            rates_to_try = [self.sample_rate, 48000, 44100]
            opened = False
            for rate in rates_to_try:
                try:
                    self._mic_rate = rate
                    block = int(rate * (self.chunk_samples / self.sample_rate))
                    self._mic_stream = sd.InputStream(
                        samplerate=rate,
                        channels=self.channels,
                        dtype="float32",
                        blocksize=block,
                        device=self.device_index,
                        callback=self._mic_callback
                    )
                    self._mic_stream.start()
                    opened = True
                    logger.info(f"Microphone capture active (Rate={rate}Hz, Target={self.sample_rate}Hz, Block={block})")
                    break
                except Exception as e:
                    logger.debug(f"Could not open mic at {rate}Hz: {e}")

            if not opened:
                logger.warning("Could not open microphone stream on any sample rate.")

    def set_audio_source(self, source: str):
        """Switches active audio source dynamically ('loopback', 'dual', 'mic')."""
        if source not in ("loopback", "dual", "mic") or self.audio_source == source:
            return
        logger.info(f"Switching audio source to: {source}")
        was_running = self._is_running
        if was_running:
            self.stop()
        self.audio_source = source
        self.use_wasapi_loopback = source in ("loopback", "dual")
        if was_running:
            self.start()

    def read_chunk(self, timeout: float = 0.2) -> Optional[Dict]:
        """Fetches the next available captured chunk. In dual mode, cleanly mixes mic and loopback."""
        if self.audio_source == "dual":
            t0 = time.time()
            mic_data = None
            lb_data = None

            # Get microphone frame
            try:
                mic_data = self._mic_queue.get(timeout=timeout)
            except queue.Empty:
                pass

            # Try to grab simultaneous loopback frame without blocking
            try:
                lb_data = self._lb_queue.get_nowait()
            except queue.Empty:
                pass

            # If mic didn't arrive, wait remainder of timeout for loopback
            if mic_data is None and lb_data is None:
                elapsed = time.time() - t0
                rem = max(0.01, timeout - elapsed)
                try:
                    lb_data = self._lb_queue.get(timeout=rem)
                except queue.Empty:
                    pass

            if mic_data is None and lb_data is None:
                return None

            # Drain any excess backlog to keep real-time sync (< 200ms latency)
            while self._mic_queue.qsize() > 2:
                try:
                    self._mic_queue.get_nowait()
                except queue.Empty:
                    break
            while self._lb_queue.qsize() > 2:
                try:
                    self._lb_queue.get_nowait()
                except queue.Empty:
                    break

            # Synchronous mix with speaker attribution
            speaker = "Interviewer"
            mic_energy = float(np.sqrt(np.mean(np.square(mic_data)))) if mic_data is not None and len(mic_data) > 0 else 0.0
            lb_energy = float(np.sqrt(np.mean(np.square(lb_data)))) if lb_data is not None and len(lb_data) > 0 else 0.0

            if mic_data is not None and lb_data is not None:
                min_len = min(len(mic_data), len(lb_data))
                if lb_energy > 0.005 and mic_energy > 0.005:
                    mixed = np.clip(mic_data[:min_len] + lb_data[:min_len], -1.0, 1.0)
                    speaker = "Candidate" if mic_energy > lb_energy else "Interviewer"
                elif lb_energy > 0.005:
                    mixed = lb_data
                    speaker = "Interviewer"
                else:
                    mixed = mic_data
                    speaker = "Candidate"
            elif mic_data is not None:
                mixed = mic_data
                speaker = "Candidate"
            else:
                mixed = lb_data
                speaker = "Interviewer"

            return {
                "data": mixed,
                "timestamp": time.time(),
                "speaker": speaker
            }
        else:
            try:
                return self.audio_queue.get(timeout=timeout)
            except queue.Empty:
                return None

    def stop(self):
        """Stops both microphone and loopback audio recording streams and flushes queues."""
        self._is_running = False

        # Clear queue backlogs
        for q in (self.audio_queue, self._mic_queue, self._lb_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        if self._mic_stream:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception as e:
                logger.debug(f"Error stopping mic stream: {e}")
            self._mic_stream = None

        if self._lb_stream:
            try:
                self._lb_stream.stop_stream()
                self._lb_stream.close()
            except Exception as e:
                logger.debug(f"Error stopping loopback stream: {e}")
            self._lb_stream = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception as e:
                logger.debug(f"Error terminating pyaudio: {e}")
            self._pa = None

        logger.info("Audio capture stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running
