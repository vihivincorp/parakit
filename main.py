"""
Main application entry point for the Multimodal Real-Time AI Desktop Assistant.
Orchestrates background worker threads for screen capture, audio STT, multimodal fusion,
and runs the Tkinter desktop overlay on the main thread.
"""

import sys
import time
import threading
import argparse
from typing import Optional

from config import cfg, AppConfig
from utils.logger import logger
from utils.timing import latency_tracker
from audio.capture import AudioCapture
from audio.vad import VoiceActivityDetector, AudioChunk
from audio.speech_to_text import SpeechToText, SpeechSegment
from screen.capture import ScreenCapture
from screen.preprocess import ImagePreprocessor
from screen.ocr import ScreenOCR
from screen.language_detector import LanguageDetector
from intelligence.context_manager import MultimodalContextManager
from intelligence.question_detector import QuestionDetector
from intelligence.llm_client import LLMClient
from intelligence.response_manager import ResponseManager
from intelligence.resume_manager import ResumeManager
from ui.overlay import DesktopOverlay

class MultimodalAssistantApp:
    """Coordinates screen, audio, intelligence workers and the presentation overlay."""

    def __init__(self, config: AppConfig = cfg):
        self.cfg = config
        self.is_running = False
        self.is_paused = False

        # Initialize Subsystems
        logger.info("Initializing Multimodal Desktop Assistant subsystems...")

        # 1. Initialize Audio Subsystems first (faster-whisper / ctranslate2 native runtime)
        self.stt = SpeechToText(
            model_size=self.cfg.audio.stt_model,
            engine=self.cfg.audio.stt_engine,
            task=self.cfg.audio.stt_task,
            language=self.cfg.audio.stt_language,
            beam_size=self.cfg.audio.stt_beam_size,
            initial_prompt=self.cfg.audio.initial_prompt
        )
        self.audio_capture = AudioCapture(
            sample_rate=self.cfg.audio.sample_rate,
            channels=self.cfg.audio.channels,
            chunk_duration_sec=self.cfg.audio.chunk_duration_sec,
            device_index=self.cfg.audio.device_index,
            use_wasapi_loopback=self.cfg.audio.use_wasapi_loopback,
            audio_source=self.cfg.audio.audio_source
        )
        self.vad = VoiceActivityDetector(
            energy_threshold=self.cfg.audio.vad_energy_threshold,
            silence_timeout_sec=self.cfg.audio.vad_silence_duration_sec
        )

        # 2. Initialize Screen Subsystems (Media OCR / WinRT)
        self.screen_capture = ScreenCapture(monitor_index=self.cfg.screen.monitor_index)
        self.image_preprocessor = ImagePreprocessor(
            to_grayscale=self.cfg.screen.preprocess_grayscale,
            contrast_factor=self.cfg.screen.preprocess_contrast,
            resize_factor=self.cfg.screen.preprocess_resize_factor
        )
        self.ocr_engine = ScreenOCR(
            tesseract_cmd=self.cfg.screen.tesseract_cmd,
            similarity_threshold=self.cfg.screen.similarity_threshold
        )

        self.context_mgr = MultimodalContextManager(
            speech_window_sec=self.cfg.context.speech_window_sec,
            max_history_turns=self.cfg.context.max_history_turns,
            max_screen_chars=self.cfg.context.max_screen_chars
        )
        self.question_detector = QuestionDetector()
        self.llm_client = LLMClient(
            api_key=self.cfg.llm.api_key,
            provider=self.cfg.llm.provider,
            model_name=self.cfg.llm.model_name,
            max_tokens=self.cfg.llm.max_tokens,
            temperature=self.cfg.llm.temperature,
            mock_mode=self.cfg.llm.mock_mode
        )
        self.response_mgr = ResponseManager()

        # UI Overlay (runs on main thread)
        self.overlay = DesktopOverlay(
            width=self.cfg.ui.width,
            height=self.cfg.ui.height,
            min_height=self.cfg.ui.min_height,
            max_height=self.cfg.ui.max_height,
            opacity=self.cfg.ui.opacity,
            exclude_from_capture=self.cfg.ui.exclude_from_capture
        )
        self.overlay.on_toggle_pause = self._on_ui_toggle_pause
        self.overlay.on_audio_source_change = self._on_ui_audio_source_change
        self.overlay.on_manual_query = lambda q: threading.Thread(target=self._handle_manual_query, args=(q,), daemon=True).start()
        self.overlay.on_trigger_screen_analysis = lambda: threading.Thread(target=self._handle_screen_solve, daemon=True).start()
        self.overlay.on_resume_uploaded = lambda path: threading.Thread(target=self._handle_resume_uploaded, args=(path,), daemon=True).start()

        # Sync UI audio mode button with configured source
        self.overlay.set_audio_mode(self.cfg.audio.audio_source)

        # 3. Initialize Resume Manager & load cached resume if present
        self.resume_mgr = ResumeManager()
        if self.resume_mgr.get_resume_text():
            self.context_mgr.set_resume(self.resume_mgr.get_resume_text(), self.resume_mgr.get_resume_filename())
            self.overlay.set_resume_badge(self.resume_mgr.get_resume_filename())
            self._sync_resume_vocabulary(self.resume_mgr.get_resume_text())

        # Worker Threads
        self._screen_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None

    def _sync_resume_vocabulary(self, resume_text: str):
        """Extracts technical terms and project names from resume to bias Whisper STT vocabulary."""
        if not resume_text or not hasattr(self, "stt"):
            return
        import re
        tokens = re.findall(r'[A-Za-z0-9+#.-]{3,}', resume_text)
        candidates = set(["HireMatrix", "Neo4j", "FastAPI", "GitProof", "TextMorph", "LeetCode", "PyTorch", "MongoDB"])
        for t in tokens:
            if t[0].isupper() or any(c in t for c in "+#.") or t.isupper():
                if len(t) <= 25 and t.lower() not in ("the", "and", "for", "with", "from", "using", "built", "developed"):
                    candidates.add(t)
        self.stt.update_domain_vocabulary(list(candidates)[:35])

    def _handle_resume_uploaded(self, file_path: str):
        """Processes an uploaded candidate resume or profile document."""
        doc = self.resume_mgr.load_document(file_path)
        if doc:
            self.context_mgr.set_resume(doc["text"], doc["filename"])
            self._sync_resume_vocabulary(doc["text"])
            self.overlay.post_update("resume", {"filename": doc["filename"]})
            self.overlay.post_update("query", {
                "text": f"Loaded Resume: '{doc['filename']}' ({doc['char_count']} chars). Interview questions will be grounded in this profile!",
                "speaker": "System"
            })

    def _handle_screen_solve(self):
        """Immediately captures screen, runs OCR, detects language, and streams universal solution."""
        logger.info("Universal Screen Analysis triggered")
        e2e_start = time.perf_counter()
        try:
            self.overlay.post_update("status", {"status": "READING SCREEN", "color": "#3B82F6"})
            raw_frame = self.screen_capture.capture_frame(crop_box=self.cfg.screen.crop_region)
            processed_frame = self.image_preprocessor.preprocess(raw_frame)
            ocr_result = self.ocr_engine.extract_text(processed_frame)
            if ocr_result.text:
                self.context_mgr.update_screen(ocr_result.text)

            # 1. Language Detection & Selection
            selected_lang = self.overlay.get_selected_language()
            if selected_lang == "Auto-Detect":
                detected_lang, conf = LanguageDetector.detect(ocr_result.text or "")
                target_lang = detected_lang
                logger.info(f"Screen language auto-detected: {target_lang} (Confidence: {conf:.2f})")
            else:
                target_lang = selected_lang
                logger.info(f"Using user-selected language: {target_lang}")

            self.overlay.post_update("detected_language", {"lang": target_lang})

            prompt_question = (
                f"Comprehend and solve whatever is opened on this screen (assessment test / MCQ, coding problem, "
                f"terminal error log, system architecture diagram, or document). "
                f"Identify what it is and provide the direct, decisive, and optimal answer or solution."
            )
            self.overlay.post_update("query", {"text": "🧠 Analyze Screen: Analyzing visible content...", "speaker": "User"})
            self.overlay.post_update("status", {"status": "STREAMING", "color": "#818CF8"})
            self.overlay.post_update("stream_start", {})

            prompt_payload = self.context_mgr.format_context_for_llm(prompt_question)

            # 2. Token-by-Token Streaming directly into UI
            accumulated_chunks = []
            for chunk in self.llm_client.generate_response_stream(
                screen_context=prompt_payload["screen_context"],
                audio_context=prompt_payload["speech_context"],
                conversation_history=prompt_payload["conversation_history"],
                current_question=prompt_payload["current_question"],
                target_language=target_lang,
                resume_context=prompt_payload.get("resume_context", "")
            ):
                accumulated_chunks.append(chunk)
                self.overlay.post_update("stream_chunk", {"chunk": chunk})

            full_answer = "".join(accumulated_chunks).strip()
            e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000.0
            latency_tracker.record("e2e_response", e2e_latency_ms)
            ttft_ms = getattr(self.llm_client, "last_ttft_ms", 0.0)
            if ttft_ms > 0:
                latency_tracker.record("ttft", ttft_ms)

            self.overlay.post_update("stream_end", {"full_text": full_answer})
            self.overlay.post_update("latency", {"latency_ms": e2e_latency_ms, "ttft_ms": ttft_ms})
            self.overlay.post_update("status", {"status": "LISTENING", "color": "#10B981"})
            self.context_mgr.add_conversation_turn(prompt_question, full_answer, speaker="User")
            logger.info(f"Screen analysis generated in {e2e_latency_ms:.1f}ms (TTFT: {ttft_ms:.1f}ms)")

            # Check if screen solution contains code blocks or solution code - pin until session ends
            has_code = (
                "```" in full_answer or 
                "class Solution" in full_answer or 
                "def " in full_answer or 
                "public class" in full_answer or 
                "vector<" in full_answer or 
                "int main(" in full_answer
            )
            if has_code:
                logger.info(f"Screen analysis produced code solution - pinning in single chat: {target_lang}")
                self.overlay.post_update("pin_code", {
                    "code": full_answer,
                    "title": f"Screen Problem ({target_lang})",
                    "already_streamed": True
                })

        except Exception as e:
            logger.error(f"Error in manual screen analysis: {e}")
            self.overlay.post_update("status", {"status": "ERROR", "color": "#EF4444"})

    def _handle_manual_query(self, query: str):
        """Processes a query typed by the user or triggered by chip in the overlay with streaming."""
        logger.info(f"Manual query received: '{query}'")
        e2e_start = time.perf_counter()
        try:
            lower_q = query.lower()
            needs_screen = any(w in lower_q for w in ("screen", "direct answer", "solve this", "look at", "what is on", "error on", "terminal"))

            selected_lang = self.overlay.get_selected_language()
            target_lang = "Python" if selected_lang == "Auto-Detect" else selected_lang

            if needs_screen:
                self.overlay.post_update("status", {"status": "READING SCREEN", "color": "#3B82F6"})
                raw_frame = self.screen_capture.capture_frame(crop_box=self.cfg.screen.crop_region)
                processed_frame = self.image_preprocessor.preprocess(raw_frame)
                ocr_result = self.ocr_engine.extract_text(processed_frame)
                if ocr_result.text:
                    self.context_mgr.update_screen(ocr_result.text)
                    if selected_lang == "Auto-Detect":
                        target_lang, _ = LanguageDetector.detect(ocr_result.text or "")
            else:
                self.overlay.post_update("status", {"status": "STREAMING", "color": "#818CF8"})

            self.overlay.post_update("detected_language", {"lang": target_lang})
            self.overlay.post_update("status", {"status": "STREAMING", "color": "#818CF8"})
            self.overlay.post_update("stream_start", {})

            prompt_payload = self.context_mgr.format_context_for_llm(query, include_screen=needs_screen)

            accumulated_chunks = []
            for chunk in self.llm_client.generate_response_stream(
                screen_context=prompt_payload["screen_context"],
                audio_context=prompt_payload["speech_context"],
                conversation_history=prompt_payload["conversation_history"],
                current_question=prompt_payload["current_question"],
                target_language=target_lang,
                resume_context=prompt_payload.get("resume_context", "")
            ):
                accumulated_chunks.append(chunk)
                self.overlay.post_update("stream_chunk", {"chunk": chunk})

            full_answer = "".join(accumulated_chunks).strip()
            e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000.0
            latency_tracker.record("e2e_response", e2e_latency_ms)
            ttft_ms = getattr(self.llm_client, "last_ttft_ms", 0.0)
            if ttft_ms > 0:
                latency_tracker.record("ttft", ttft_ms)

            self.overlay.post_update("stream_end", {"full_text": full_answer})
            self.overlay.post_update("latency", {"latency_ms": e2e_latency_ms, "ttft_ms": ttft_ms})
            self.overlay.post_update("status", {"status": "LISTENING", "color": "#10B981"})
            self.context_mgr.add_conversation_turn(query, full_answer, speaker="Candidate")

            # If manual query asked for code or returned code blocks, pin it as well
            if "```" in full_answer or "class Solution" in full_answer:
                logger.info(f"Manual query produced code solution - pinning in single chat: {target_lang}")
                self.overlay.post_update("pin_code", {
                    "code": full_answer,
                    "title": f"Coding Solution ({target_lang})",
                    "already_streamed": True
                })

        except Exception as e:
            logger.error(f"Error in manual query: {e}")
            self.overlay.post_update("status", {"status": "ERROR", "color": "#EF4444"})

    def _on_ui_toggle_pause(self, paused: bool):
        self.is_paused = paused
        logger.info(f"Assistant processing {'PAUSED' if paused else 'RESUMED'}")

    def _on_ui_audio_source_change(self, mode: str):
        """Switches audio capture source at runtime ('loopback', 'dual', 'mic')."""
        logger.info(f"App switching audio capture source to '{mode}'")
        self.audio_capture.set_audio_source(mode)

    def _screen_worker(self):
        """Background thread monitoring the desktop screen (active only when auto_screen_capture is enabled)."""
        logger.info("Screen worker thread active")
        while self.is_running:
            if self.is_paused or not getattr(self.cfg.screen, "auto_screen_capture", False):
                time.sleep(1.0)
                continue

            try:
                # 1. Grab screen frame
                raw_frame = self.screen_capture.capture_frame(crop_box=self.cfg.screen.crop_region)
                # 2. Preprocess image
                processed_frame = self.image_preprocessor.preprocess(raw_frame)
                # 3. Extract text with delta-check
                ocr_result = self.ocr_engine.extract_text(processed_frame)

                if ocr_result.has_changed and ocr_result.text:
                    logger.debug(f"Screen text updated: {len(ocr_result.text)} chars")
                    self.context_mgr.update_screen(ocr_result.text)

            except Exception as e:
                logger.error(f"Error in screen worker: {e}")

            time.sleep(self.cfg.screen.capture_interval_sec)

    def _audio_worker(self):
        """Background thread processing real-time audio, VAD, STT, and reasoning."""
        logger.info("Audio worker thread active")
        self.audio_capture.start()

        while self.is_running:
            if self.is_paused:
                time.sleep(0.2)
                continue

            try:
                chunk_dict = self.audio_capture.read_chunk(timeout=0.2)
                if not chunk_dict:
                    continue

                chunk = AudioChunk(
                    data=chunk_dict["data"],
                    timestamp=chunk_dict["timestamp"],
                    speaker=chunk_dict.get("speaker", "Interviewer")
                )

                # Voice Activity Detection
                speech_segment = self.vad.process_chunk(chunk)
                if speech_segment:
                    threading.Thread(
                        target=self._handle_completed_speech,
                        args=(speech_segment,),
                        daemon=True
                    ).start()

            except Exception as e:
                logger.error(f"Error in audio worker: {e}")

        self.audio_capture.stop()

    def _handle_completed_speech(self, segment: SpeechSegment):
        """Pipeline execution when user completes an utterance."""
        e2e_start = time.perf_counter()
        speaker = getattr(segment, "speaker", "Interviewer")
        try:
            self.overlay.post_update("status", {"status": "TRANSCRIBING", "color": "#3B82F6"})

            # 1. Transcribe speech
            transcript = self.stt.transcribe(segment)
            if not transcript or not transcript.text:
                self.overlay.post_update("status", {"status": "LISTENING", "color": "#10B981"})
                return

            transcript.speaker = speaker
            logger.info(f"Utterance detected [{speaker}]: '{transcript.text}'")
            self.context_mgr.add_transcript(transcript)

            # 2. Gate with Question / Intent Detector
            intent = self.question_detector.detect(transcript.text)
            logger.info(f"Intent classified: {intent.category.value} (Trigger LLM: {intent.trigger_llm})")

            if not intent.trigger_llm:
                self.overlay.post_update("query", {"text": f"Heard: \"{intent.text}\"", "speaker": speaker})
                self.overlay.post_update("status", {"status": "LISTENING", "color": "#10B981"})
                return

            # 3. Update UI with detected question and start streaming immediately
            self.overlay.post_update("query", {"text": intent.text, "speaker": speaker})
            self.overlay.post_update("status", {"status": "STREAMING", "color": "#818CF8"})
            self.overlay.post_update("stream_start", {})

            selected_lang = self.overlay.get_selected_language()
            if selected_lang == "Auto-Detect":
                target_lang, _ = LanguageDetector.detect(self.context_mgr.get_current_screen() or "")
            else:
                target_lang = selected_lang

            self.overlay.post_update("detected_language", {"lang": target_lang})

            # 4. Formulate multimodal context (only include screen if explicitly asked)
            lower_intent = intent.text.lower()
            speech_needs_screen = any(w in lower_intent for w in ("screen", "look here", "this problem", "this error", "on my display", "solve this", "code here"))
            prompt_payload = self.context_mgr.format_context_for_llm(intent.text, include_screen=speech_needs_screen)

            # 5. Stream LLM tokens in real-time
            accumulated_chunks = []
            for chunk in self.llm_client.generate_response_stream(
                screen_context=prompt_payload["screen_context"],
                audio_context=prompt_payload["speech_context"],
                conversation_history=prompt_payload["conversation_history"],
                current_question=prompt_payload["current_question"],
                target_language=target_lang,
                resume_context=prompt_payload.get("resume_context", "")
            ):
                accumulated_chunks.append(chunk)
                self.overlay.post_update("stream_chunk", {"chunk": chunk})

            full_answer = "".join(accumulated_chunks).strip()
            e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000.0
            latency_tracker.record("e2e_response", e2e_latency_ms)
            ttft_ms = getattr(self.llm_client, "last_ttft_ms", 0.0)
            if ttft_ms > 0:
                latency_tracker.record("ttft", ttft_ms)

            self.overlay.post_update("stream_end", {"full_text": full_answer})
            self.overlay.post_update("latency", {"latency_ms": e2e_latency_ms, "ttft_ms": ttft_ms})
            self.overlay.post_update("status", {"status": "LISTENING", "color": "#10B981"})
            self.context_mgr.add_conversation_turn(intent.text, full_answer, speaker=speaker)
            # Clear consumed speech transcripts so past words do not linger into subsequent questions
            self.context_mgr.clear_transcripts()
            logger.info(f"Utterance answer synthesized in {e2e_latency_ms:.1f}ms (TTFT: {ttft_ms:.1f}ms) for [{speaker}]")

            # If spoken query returned code blocks, pin it until session ends
            if "```" in full_answer and any(w in intent.text.lower() for w in ("code", "write", "solve", "function", "implement", "class")):
                logger.info(f"Spoken coding question produced code - pinning in single chat: {target_lang}")
                self.overlay.post_update("pin_code", {
                    "code": full_answer,
                    "title": f"Live Coding Solution ({target_lang})",
                    "already_streamed": True
                })

        except Exception as e:
            logger.error(f"Error handling completed speech: {e}")
            self.overlay.post_update("status", {"status": "ERROR", "color": "#EF4444"})

    def _on_answer_ready(self, response, e2e_start: float):
        """Dispatches formatted answer to UI and records turn."""
        e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000.0
        latency_tracker.record("e2e_response", e2e_latency_ms)

        self.overlay.post_update("answer", {"text": response.answer})
        self.overlay.post_update("latency", {"latency_ms": e2e_latency_ms})
        self.overlay.post_update("status", {"status": "LISTENING", "color": "#10B981"})
        
        self.context_mgr.add_conversation_turn(response.question, response.answer)
        logger.info(f"Answer synthesized in {e2e_latency_ms:.1f}ms: '{response.answer}'")

    def simulate_event(self, question: str, screen_context: Optional[str] = None, speaker: str = "Interviewer"):
        """Simulates an end-to-end multimodal query for testing."""
        if screen_context:
            self.context_mgr.update_screen(screen_context)
        
        import numpy as np
        fake_audio = np.zeros(16000 * 2, dtype=np.float32)
        segment = SpeechSegment(
            audio=fake_audio,
            start_time=time.time() - 2.0,
            end_time=time.time(),
            duration_sec=2.0,
            speaker=speaker
        )
        # Override mock STT for explicit simulation test
        from audio.speech_to_text import TranscriptSegment
        self.stt.transcribe = lambda seg: TranscriptSegment(
            text=question,
            start_time=seg.start_time,
            end_time=seg.end_time,
            confidence=0.95,
            timestamp=time.time(),
            speaker=seg.speaker
        )
        self._handle_completed_speech(segment)

    def start(self):
        """Starts worker threads and blocks on the main UI loop."""
        self.is_running = True

        self._screen_thread = threading.Thread(target=self._screen_worker, daemon=True)
        self._screen_thread.start()

        self._audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self._audio_thread.start()

        logger.info("Multimodal Desktop Assistant running. Press Ctrl+C in console or [X] on overlay to exit.")
        try:
            self.overlay.run()
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown of all background services."""
        if not self.is_running:
            return
        logger.info("Shutting down assistant...")
        self.is_running = False
        self.audio_capture.stop()
        self.screen_capture.close()
        self.overlay.close()
        logger.info(latency_tracker.summary_string())
        logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Real-Time AI Desktop Assistant")
    parser.add_argument("--dry-run", action="store_true", help="Initialize and verify pipelines then exit")
    parser.add_argument("--simulate", action="store_true", help="Run simulated query test on startup")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index to capture")
    parser.add_argument("--mic", type=int, default=None, help="Audio input device index")
    args = parser.parse_args()

    cfg.screen.monitor_index = args.monitor
    cfg.audio.device_index = args.mic

    app = MultimodalAssistantApp(config=cfg)

    if args.dry_run:
        logger.info("Executing dry run test...")
        app.simulate_event(
            question="Why did you choose Random Forest?",
            screen_context="Random Forest: Accuracy = 91.4%. Logistic Regression: Accuracy = 78.1%."
        )
        print("\n" + latency_tracker.summary_string())
        logger.info("Dry run completed successfully.")
        return

    if args.simulate:
        # Schedule simulation 2 seconds after UI start
        threading.Timer(
            2.0,
            lambda: app.simulate_event(
                question="Why did you choose Random Forest over other models?",
                screen_context="Model Evaluation: Random Forest Accuracy: 91.4%, Logistic Regression: 78.1%."
            )
        ).start()

    app.start()

if __name__ == "__main__":
    main()
