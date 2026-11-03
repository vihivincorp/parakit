import time
import tkinter as tk
import pytest
from ui.overlay import DesktopOverlay

@pytest.fixture(scope="module")
def overlay():
    inst = DesktopOverlay(width=840, height=480, exclude_from_capture=False)
    inst.root.update()
    yield inst
    inst.close()

def test_overlay_static_initialization(overlay):
    overlay.root.update()
    assert overlay.is_static_window is True
    assert overlay.current_height >= 460
    assert hasattr(overlay, "user_entry")
    assert overlay.user_entry.winfo_height() > 10
    assert overlay.user_entry.winfo_rooty() < overlay.answer_box.winfo_rooty()

def test_overlay_static_during_content_stream(overlay):
    """Verify window height stays strictly static and does not jitter or expand dynamically."""
    overlay.root.update()
    initial_h = overlay.current_height

    # Feed 8 lines
    text_8 = "\n".join([f"Bullet point {i}: detailed concept" for i in range(8)])
    overlay.post_update("answer", {"text": text_8})
    time.sleep(0.1)
    overlay.root.update()
    assert overlay.current_height == initial_h  # Static height preserved!

    # Feed 25 lines
    text_25 = "\n".join([f"Line {i}: code block implementation detail" for i in range(25)])
    overlay.post_update("answer", {"text": text_25})
    time.sleep(0.1)
    overlay.root.update()
    assert overlay.current_height == initial_h  # Static height preserved!

def test_question_history_navigation(overlay):
    """Verify currently asked question display and moving cursor up/down navigates past questions."""
    overlay.root.update()

    # Feed Question 1
    overlay.post_update("query", {"text": "What is binary search?", "speaker": "Interviewer"})
    time.sleep(0.05)
    overlay.root.update()
    assert "binary search" in overlay.query_text.cget("text")
    assert "CURRENTLY ASKED QUESTION" in overlay.q_header_lbl.cget("text")
    assert overlay.current_q_idx == 0

    # Feed Question 2
    overlay.post_update("query", {"text": "What is its time complexity?", "speaker": "Interviewer"})
    time.sleep(0.05)
    overlay.root.update()
    assert "time complexity" in overlay.query_text.cget("text")
    assert "CURRENTLY ASKED QUESTION" in overlay.q_header_lbl.cget("text")
    assert overlay.current_q_idx == 1

    # Move cursor UP (review previous question)
    overlay.show_prev_question()
    overlay.root.update()
    assert "binary search" in overlay.query_text.cget("text")
    assert "PREVIOUS QUESTION" in overlay.q_header_lbl.cget("text")
    assert overlay.current_q_idx == 0

    # Move cursor DOWN (return to current question)
    overlay.show_next_question()
    overlay.root.update()
    assert "time complexity" in overlay.query_text.cget("text")
    assert "CURRENTLY ASKED QUESTION" in overlay.q_header_lbl.cget("text")
    assert overlay.current_q_idx == 1

def test_pinned_code_session_retention(overlay):
    overlay.root.update()

    code_sol = (
        "### LeetCode Solution\n"
        "```python\n"
        "class Solution:\n"
        "    def twoSum(self, nums: list[int], target: int) -> list[int]:\n"
        "        seen = {}\n"
        "        for i, num in enumerate(nums):\n"
        "            diff = target - num\n"
        "            if diff in seen:\n"
        "                return [seen[diff], i]\n"
        "            seen[num] = i\n"
        "```\n"
        "### Complexity\n"
        "- Time: O(N)\n"
        "- Space: O(N)\n"
    )

    overlay.post_update("pin_code", {"code": code_sol, "title": "LeetCode TwoSum (Python)"})
    time.sleep(0.1)
    overlay.root.update()

    assert overlay.pinned_code_text == code_sol
    assert overlay.pinned_code_title == "LeetCode TwoSum (Python)"
    chat_content = overlay.answer_box.get("1.0", "end")
    assert "class Solution:" in chat_content
    assert "def twoSum" in chat_content
    assert overlay.jump_code_btn.cget("state") == "normal"

    overlay.post_update("stream_start", {"speaker": "Interviewer"})
    overlay.post_update("stream_chunk", {"chunk": "If the array is sorted, we can use two pointers for O(1) space."})
    overlay.post_update("stream_end", {"full_text": "If the array is sorted, we can use two pointers for O(1) space."})
    time.sleep(0.1)
    overlay.root.update()

    updated_chat = overlay.answer_box.get("1.0", "end")
    assert "two pointers" in updated_chat
    assert "class Solution:" in updated_chat

def test_overlay_minimize_and_restore(overlay):
    overlay.root.update()
    assert overlay.is_minimized is False
    assert hasattr(overlay, "minimize_btn")

    # Test minimize
    overlay.minimize()
    overlay.root.update()
    assert overlay.is_minimized is True
    assert overlay.root.state() == "withdrawn"

    # Test restore
    overlay.restore()
    overlay.root.update()
    assert overlay.is_minimized is False
    assert overlay.root.state() == "normal"

    # Test toggle
    overlay.toggle_minimize()
    overlay.root.update()
    assert overlay.is_minimized is True

    overlay.toggle_minimize()
    overlay.root.update()
    assert overlay.is_minimized is False
