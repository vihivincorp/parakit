"""
Global Windows Hotkey Listener using Win32 RegisterHotKey API.
Enables system-wide shortcut detection (e.g. Alt+M to minimize/restore overlay)
without requiring administrative privileges or external dependencies.
"""

import sys
import threading
import ctypes
from ctypes import wintypes
from typing import Callable, Optional, Tuple, List

from utils.logger import logger

# Win32 Modifier Constants
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000  # Requires Windows 7+

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

MODIFIER_MAP = {
    "alt": MOD_ALT,
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "windows": MOD_WIN
}

class GlobalHotkeyListener:
    """Listens for a global hotkey across all Windows applications."""

    def __init__(
        self,
        key: str = "m",
        modifiers: Tuple[str, ...] = ("alt",),
        callback: Optional[Callable[[], None]] = None,
        hotkey_id: int = 1001
    ):
        self.key = key.lower()
        self.modifiers = tuple(m.lower() for m in modifiers)
        self.callback = callback
        self.hotkey_id = hotkey_id

        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._is_running = False
        self._registered = False
        self._is_windows = sys.platform == "win32"

    def _parse_modifiers(self) -> int:
        """Combines modifier flags with MOD_NOREPEAT to prevent repeat flooding."""
        flags = MOD_NOREPEAT
        for m in self.modifiers:
            flags |= MODIFIER_MAP.get(m, 0)
        return flags

    def _parse_vk(self) -> int:
        """Converts key string to virtual key code."""
        k = self.key.strip()
        if len(k) == 1:
            return ord(k.upper())
        # Function keys F1-F12
        if k.startswith("f") and k[1:].isdigit():
            fn_num = int(k[1:])
            if 1 <= fn_num <= 12:
                return 0x6F + fn_num  # VK_F1 is 0x70
        return ord('M')

    def start(self) -> bool:
        """Starts the background hotkey listener thread."""
        if not self._is_windows:
            logger.info("Global hotkey listener skipped (non-Windows platform).")
            return False

        if self._is_running:
            return True

        self._is_running = True
        self._thread = threading.Thread(target=self._message_loop, daemon=True, name="GlobalHotkeyThread")
        self._thread.start()
        return True

    def _message_loop(self):
        """Worker thread message pump handling WM_HOTKEY events."""
        from . import ensure_interactive_desktop
        ensure_interactive_desktop()

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        self._thread_id = kernel32.GetCurrentThreadId()
        mod_flags = self._parse_modifiers()
        vk_code = self._parse_vk()

        success = user32.RegisterHotKey(None, self.hotkey_id, mod_flags, vk_code)
        if success:
            self._registered = True
            mod_str = "+".join(m.title() for m in self.modifiers)
            logger.info(f"Global hotkey registered: [{mod_str}+{self.key.upper()}] (Press to minimize/restore)")
        else:
            err = ctypes.get_last_error()
            logger.warning(f"Could not register global hotkey (Error {err}). Shortcut might be in use by another app.")
            self._is_running = False
            return

        msg = wintypes.MSG()
        try:
            while self._is_running:
                # GetMessageW blocks until a message is posted to this thread's queue
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret == 0 or ret == -1:  # WM_QUIT or error
                    break

                if msg.message == WM_HOTKEY and msg.wParam == self.hotkey_id:
                    logger.debug("Global hotkey triggered")
                    if self.callback:
                        try:
                            self.callback()
                        except Exception as e:
                            logger.error(f"Error in hotkey callback: {e}")

                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            if self._registered:
                user32.UnregisterHotKey(None, self.hotkey_id)
                self._registered = False
            self._is_running = False

    def stop(self):
        """Stops the listener thread and unregisters hotkey."""
        if not self._is_running:
            return

        self._is_running = False
        if self._thread_id and self._is_windows:
            try:
                # Post WM_QUIT to wake up GetMessageW
                ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._thread_id = None

