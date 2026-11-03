"""
Unit tests for the Windows capture-exclusion research module.
"""

import sys
import tkinter as tk
from windows.capture_exclusion import WindowsCaptureExclusion, WDA_EXCLUDEFROMCAPTURE, WDA_NONE
from windows.detect_overlay import WindowsOverlayDetector

def test_display_affinity_constants():
    assert WDA_NONE == 0x00000000
    assert WDA_EXCLUDEFROMCAPTURE == 0x00000011

def test_windows_exclusion_lifecycle():
    mgr = WindowsCaptureExclusion()
    detector = WindowsOverlayDetector()

    if sys.platform != "win32":
        # Safe mock assertion on non-Windows
        assert mgr.set_exclusion(12345, exclude=True) is False
        return

    # On Windows, test with real Tkinter window handle
    root = tk.Tk()
    root.deiconify()
    root.update()
    hwnd = root.winfo_id()

    # Apply exclusion
    success = mgr.set_exclusion(hwnd, exclude=True)
    assert success is True

    # Audit via detector
    audit = detector.audit_capture_affinity(hwnd)
    assert audit["affinity_raw"] == WDA_EXCLUDEFROMCAPTURE
    assert audit["is_capture_excluded"] is True

    # Revert to standard
    revert_success = mgr.set_exclusion(hwnd, exclude=False)
    assert revert_success is True

    root.destroy()

def test_global_hotkey_parsing():
    from windows.global_hotkey import GlobalHotkeyListener, MOD_ALT, MOD_NOREPEAT
    listener = GlobalHotkeyListener(key="m", modifiers=("alt",))
    assert listener._parse_modifiers() == (MOD_ALT | MOD_NOREPEAT)
    assert listener._parse_vk() == ord('M')

def test_global_hotkey_listener_lifecycle():
    from windows.global_hotkey import GlobalHotkeyListener
    import time
    called = []
    # Use dedicated test combo and ID to prevent collision with running assistant
    listener = GlobalHotkeyListener(
        key="f11",
        modifiers=("ctrl", "shift", "alt"),
        callback=lambda: called.append(True),
        hotkey_id=9876
    )
    assert listener.start() is True
    time.sleep(0.2)
    assert listener._is_running is True
    listener.stop()
    assert listener._is_running is False
