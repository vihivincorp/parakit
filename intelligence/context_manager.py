"""
Multimodal Context Manager bridging screen understanding and speech transcripts.
Maintains a rolling temporal memory window for context-aware prompt generation.
"""

from dataclasses import dataclass, field
import time
from typing import List, Dict, Optional
from audio.speech_to_text import TranscriptSegment

@dataclass
class ConversationTurn:
    question: str
    answer: str
    timestamp: float
    speaker: str = "Interviewer"

@dataclass
class MultimodalSnapshot:
    screen_text: str
    recent_transcripts: List[TranscriptSegment]
    conversation_history: List[ConversationTurn]
    timestamp: float

class MultimodalContextManager:
    """Manages temporal alignment and rolling retention of multimodal signals."""

    def __init__(
        self,
        speech_window_sec: float = 60.0,
        max_history_turns: int = 5,
        max_screen_chars: int = 2500
    ):
        self.speech_window_sec = speech_window_sec
        self.max_history_turns = max_history_turns
        self.max_screen_chars = max_screen_chars

        self._current_screen_text: str = ""
        self._screen_updated_time: float = 0.0
        self._transcript_history: List[TranscriptSegment] = []
        self._conversation_turns: List[ConversationTurn] = []
        self._resume_text: str = ""
        self._resume_filename: str = ""

    def set_resume(self, text: str, filename: str = ""):
        """Stores active candidate resume for prompt grounding."""
        self._resume_text = text or ""
        self._resume_filename = filename or ""

    def get_resume(self) -> Dict[str, str]:
        """Returns loaded resume details."""
        return {
            "text": self._resume_text,
            "filename": self._resume_filename
        }

    def update_screen(self, text: str):
        """Updates the visible screen state."""
        self._current_screen_text = text[:self.max_screen_chars]
        self._screen_updated_time = time.time()

    def get_current_screen(self) -> str:
        """Returns the current screen text."""
        return self._current_screen_text

    def add_transcript(self, segment: TranscriptSegment):
        """Adds a new speech transcript segment and cleans up expired entries."""
        self._transcript_history.append(segment)
        self._prune_expired_transcripts()

    def _prune_expired_transcripts(self):
        """Removes transcripts older than the speech window."""
        now = time.time()
        cutoff = now - self.speech_window_sec
        self._transcript_history = [
            t for t in self._transcript_history if t.timestamp >= cutoff
        ]

    def add_conversation_turn(self, question: str, answer: str, speaker: str = "Interviewer"):
        """Records a successful Q&A turn with speaker attribution."""
        turn = ConversationTurn(
            question=question,
            answer=answer,
            timestamp=time.time(),
            speaker=speaker
        )
        self._conversation_turns.append(turn)
        if len(self._conversation_turns) > self.max_history_turns:
            self._conversation_turns.pop(0)

    def get_recent_speech_text(self, with_speakers: bool = False) -> str:
        """Returns joined recent speech within the rolling window."""
        self._prune_expired_transcripts()
        if with_speakers:
            return "\n".join([f"[{getattr(t, 'speaker', 'Interviewer')}]: {t.text}" for t in self._transcript_history]).strip()
        return " ".join([t.text for t in self._transcript_history]).strip()

    def clear_transcripts(self):
        """Clears consumed speech transcripts after an utterance has been answered."""
        self._transcript_history.clear()

    def clear_screen(self):
        """Clears screen text so it does not leak into subsequent verbal questions."""
        self._current_screen_text = ""

    def get_snapshot(self) -> MultimodalSnapshot:
        """Retrieves a consolidated snapshot of the current state."""
        self._prune_expired_transcripts()
        return MultimodalSnapshot(
            screen_text=self._current_screen_text,
            recent_transcripts=list(self._transcript_history),
            conversation_history=list(self._conversation_turns),
            timestamp=time.time()
        )

    def format_context_for_llm(self, current_question: str, include_screen: bool = True) -> Dict[str, str]:
        """Formats the unified prompt payload for the LLM client with speaker distinction and resume grounding."""
        snapshot = self.get_snapshot()
        speech_lines = []
        for t in snapshot.recent_transcripts:
            spk = getattr(t, "speaker", "Interviewer")
            speech_lines.append(f"[{spk}]: {t.text}")
        speech_context = "\n".join(speech_lines) if speech_lines else "[No recent speech audio]"
        
        # Keep at most the last 2 turns, and keep answers concise so the LLM doesn't latch onto past topics
        history_lines = []
        for turn in snapshot.conversation_history[-2:]:
            spk = getattr(turn, "speaker", "Interviewer")
            history_lines.append(f"[{spk}]: {turn.question}")
            ans_snippet = (turn.answer[:120] + "...") if len(turn.answer) > 120 else turn.answer
            history_lines.append(f"[Previous Turn Summary]: {ans_snippet}")
        formatted_history = "\n".join(history_lines) if history_lines else "None"

        resume_context = ""
        if self._resume_text:
            resume_context = f"[Candidate Resume: {self._resume_filename or 'Profile'}]\n{self._resume_text}"

        screen_val = snapshot.screen_text if (include_screen and snapshot.screen_text) else "[No screen context needed for this question]"

        return {
            "screen_context": screen_val,
            "speech_context": speech_context,
            "conversation_history": formatted_history,
            "current_question": current_question,
            "resume_context": resume_context
        }
