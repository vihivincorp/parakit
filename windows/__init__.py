"""Windows display affinity and capture-exclusion research package."""
from .capture_exclusion import WindowsCaptureExclusion
from .detect_overlay import WindowsOverlayDetector

__all__ = ["WindowsCaptureExclusion", "WindowsOverlayDetector"]
