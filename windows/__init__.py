"""Windows display affinity, interactive desktop attachment, and capture-exclusion package."""
import sys

def ensure_interactive_desktop() -> bool:
    """Attaches the current thread to the user's interactive 'Default' desktop on Windows."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        DESKTOP_ALL = 0x01FF
        h_default = user32.OpenDesktopW("Default", 0, False, DESKTOP_ALL)
        if h_default:
            res = bool(user32.SetThreadDesktop(h_default))
            user32.CloseDesktop(h_default)
            return res
    except Exception:
        pass
    return False

from .capture_exclusion import WindowsCaptureExclusion
from .detect_overlay import WindowsOverlayDetector
from .global_hotkey import GlobalHotkeyListener

__all__ = ["WindowsCaptureExclusion", "WindowsOverlayDetector", "GlobalHotkeyListener", "ensure_interactive_desktop"]
