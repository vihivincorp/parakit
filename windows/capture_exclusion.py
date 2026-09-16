"""
Windows Capture Exclusion Research Layer using SetWindowDisplayAffinity.
Enables transparent hiding of the overlay from screen recording and video conferencing tools.
"""

import sys
import ctypes
from ctypes import wintypes
from utils.logger import logger

# Win32 Display Affinity Constants
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Windows 10 Version 2004+ (Build 19041)

class WindowsCaptureExclusion:
    """Manages window capture exclusion flags using Windows User32 APIs."""

    def __init__(self):
        self._is_windows = sys.platform == "win32"
        self._user32 = None

        if self._is_windows:
            try:
                self._user32 = ctypes.windll.user32
                self._user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
                self._user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
                self._user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
                self._user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
            except Exception as e:
                logger.error(f"Failed to load User32 display affinity APIs: {e}")

    def set_exclusion(self, hwnd: int, exclude: bool = True) -> bool:
        """
        Sets display affinity to hide the window from screen capture.
        
        Args:
            hwnd: Window handle.
            exclude: True to hide from screen capture (WDA_EXCLUDEFROMCAPTURE),
                     False to reset to default (WDA_NONE).
        """
        if not self._is_windows or not self._user32:
            logger.warning("Capture exclusion only supported on Windows.")
            return False

        affinity = WDA_EXCLUDEFROMCAPTURE if exclude else WDA_NONE
        GA_ROOT = 2
        try:
            # Resolve top-level window handle if a child/client window handle was passed
            top_hwnd = self._user32.GetAncestor(hwnd, GA_ROOT)
            target = top_hwnd if top_hwnd else hwnd

            success = bool(self._user32.SetWindowDisplayAffinity(target, affinity))
            if not success and target != hwnd:
                success = bool(self._user32.SetWindowDisplayAffinity(hwnd, affinity))

            if success:
                mode_str = "EXCLUDED from capture (0x11)" if exclude else "VISIBLE in capture (0x0)"
                logger.info(f"SetWindowDisplayAffinity on HWND {target}: {mode_str}")
            else:
                err = ctypes.get_last_error()
                logger.warning(f"SetWindowDisplayAffinity failed on HWND {target}. Error: {err}")
            return success
        except Exception as e:
            logger.error(f"Exception calling SetWindowDisplayAffinity: {e}")
            return False

    def get_affinity(self, hwnd: int) -> int:
        """Retrieves the current display affinity setting of a window."""
        if not self._is_windows or not self._user32:
            return WDA_NONE

        affinity = wintypes.DWORD()
        GA_ROOT = 2
        try:
            top_hwnd = self._user32.GetAncestor(hwnd, GA_ROOT)
            target = top_hwnd if top_hwnd else hwnd
            success = bool(self._user32.GetWindowDisplayAffinity(target, ctypes.byref(affinity)))
            if not success and target != hwnd:
                success = bool(self._user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)))
            if success:
                return affinity.value
            return WDA_NONE
        except Exception:
            return WDA_NONE
