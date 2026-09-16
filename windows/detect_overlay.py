"""
Windows Overlay and Display Affinity Detection Module.
Provides research tools to inspect window affinity states and capture visibility.
"""

import sys
import ctypes
from ctypes import wintypes
from typing import List, Dict, Optional
from utils.logger import logger
from .capture_exclusion import WindowsCaptureExclusion, WDA_EXCLUDEFROMCAPTURE

class WindowsOverlayDetector:
    """Research module to discover and evaluate capture-excluded windows on Windows."""

    def __init__(self):
        self.exclusion_mgr = WindowsCaptureExclusion()
        self._user32 = ctypes.windll.user32 if sys.platform == "win32" else None

    def find_windows_by_title(self, title_substring: str) -> List[Dict]:
        """Finds active top-level windows containing a title substring."""
        if not self._user32:
            return []

        matched = []

        def enum_window_callback(hwnd, extra):
            if self._user32.IsWindowVisible(hwnd):
                length = self._user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    self._user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    if title_substring.lower() in title.lower():
                        affinity = self.exclusion_mgr.get_affinity(hwnd)
                        matched.append({
                            "hwnd": hwnd,
                            "title": title,
                            "affinity": hex(affinity),
                            "is_excluded": affinity == WDA_EXCLUDEFROMCAPTURE
                        })
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self._user32.EnumWindows(WNDENUMPROC(enum_window_callback), 0)
        return matched

    def audit_capture_affinity(self, target_hwnd: int) -> Dict:
        """
        Returns a diagnostic audit report for a given window handle.
        """
        affinity = self.exclusion_mgr.get_affinity(target_hwnd)
        return {
            "hwnd": target_hwnd,
            "affinity_raw": affinity,
            "affinity_hex": hex(affinity),
            "is_capture_excluded": (affinity == WDA_EXCLUDEFROMCAPTURE),
            "protection_level": "WDA_EXCLUDEFROMCAPTURE (Hidden in Windows 10/11 recordings)" if affinity == WDA_EXCLUDEFROMCAPTURE else "STANDARD_CAPTURE_VISIBLE"
        }
