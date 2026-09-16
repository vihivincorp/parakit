import time
import tkinter as tk
import pytest
from ui.overlay import DesktopOverlay

@pytest.fixture(scope="module")
def overlay():
    inst = DesktopOverlay(width=820, height=340, min_height=340, max_height=680, exclude_from_capture=False)
    inst.root.update()
    yield inst
    inst.close()

def test_overlay_dynamic_initialization(overlay):
    overlay.root.update()
    assert overlay.min_height == 340
    assert overlay.max_height == 680
    assert overlay.current_height == 340
    assert hasattr(overlay, "size_toggle_btn")
    # Verify chat option is present and entry is accessible
    assert hasattr(overlay, "user_entry")
    assert overlay.user_entry.winfo_height() > 10
    # Verify chat is positioned above the answer screen
    assert overlay.user_entry.winfo_rooty() < overlay.answer_box.winfo_rooty()

def test_overlay_progressive_expansion(overlay):
    overlay.root.update()
    initial_h = overlay.current_height

    # Feed 8 lines
    text_8 = "\n".join([f"Bullet point {i}: detailed concept" for i in range(8)])
    overlay.post_update("answer", {"text": text_8})
    time.sleep(0.1)
    overlay.root.update()
    h_8 = overlay.current_height
    assert h_8 > initial_h

    # Feed 25 lines (large content)
    text_25 = "\n".join([f"Line {i}: code block implementation detail" for i in range(25)])
    overlay.post_update("answer", {"text": text_25})
    time.sleep(0.1)
    overlay.root.update()
    h_25 = overlay.current_height
    assert h_25 > h_8
    assert h_25 == overlay.max_height  # Clamped at 680

def test_overlay_stream_reset(overlay):
    overlay.root.update()
    # Reset when stream starts
    overlay.post_update("stream_start", {})
    time.sleep(0.1)
    overlay.root.update()
    assert overlay.current_height == 340

def test_overlay_toggle_size(overlay):
    overlay.root.update()
    assert overlay.current_height == 340

    # Toggle to full expanded
    overlay.toggle_size()
    overlay.root.update()
    assert overlay.current_height == 680
    assert overlay.size_toggle_btn.cget("text") == "⤡"

    # Toggle back to compact
    overlay.toggle_size()
    overlay.root.update()
    assert overlay.current_height == 340
    assert overlay.size_toggle_btn.cget("text") == "⤢"

def test_pinned_code_session_retention(overlay):
    overlay.root.update()

    # 1. Pin a LeetCode code problem solution into the single chat
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

    # Verify code is pinned in the single chat feed
    assert overlay.pinned_code_text == code_sol
    assert overlay.pinned_code_title == "LeetCode TwoSum (Python)"
    chat_content = overlay.answer_box.get("1.0", "end")
    assert "class Solution:" in chat_content
    assert "def twoSum" in chat_content
    # Jump to code button should be enabled
    assert overlay.jump_code_btn.cget("state") == "normal"

    # 2. Simulate interviewer asking follow-up questions via live audio streams
    # Interviewer Question 1: "Can you do this in O(1) space?"
    overlay.post_update("stream_start", {"speaker": "Interviewer"})
    overlay.post_update("stream_chunk", {"chunk": "If the array is sorted, we can use two pointers for O(1) space."})
    overlay.post_update("stream_end", {"full_text": "If the array is sorted, we can use two pointers for O(1) space."})
    time.sleep(0.1)
    overlay.root.update()

    # Verify live answer is appended in the SAME single chat
    updated_chat = overlay.answer_box.get("1.0", "end")
    assert "two pointers" in updated_chat

    # CRITICAL: Verify pinned code was NOT wiped by stream_start or stream_end in the single chat!
    assert "class Solution:" in updated_chat
    assert "def twoSum" in updated_chat
    assert "seen[num] = i" in updated_chat

    # Interviewer Question 2: "What is the worst-case time complexity of hash map lookups?"
    overlay.post_update("stream_start", {"speaker": "Interviewer"})
    overlay.post_update("stream_chunk", {"chunk": "Average O(1), worst-case O(N) due to hash collisions."})
    overlay.post_update("stream_end", {"full_text": "Average O(1), worst-case O(N) due to hash collisions."})
    time.sleep(0.1)
    overlay.root.update()

    # Verify BOTH questions and the original code solution are STILL preserved in the single chat
    final_chat = overlay.answer_box.get("1.0", "end")
    assert "class Solution:" in final_chat
    assert "def twoSum" in final_chat
    assert "two pointers" in final_chat
    assert "hash collisions" in final_chat

    # 3. Test navigation shortcuts (jump to code and jump to latest)
    overlay.jump_to_code()
    overlay.root.update()
    overlay.jump_to_latest()
    overlay.root.update()


