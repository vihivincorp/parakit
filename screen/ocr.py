"""
Screen OCR engine using Tesseract with change detection to prevent redundant processing.
"""

import os
import shutil
import difflib
import time
from dataclasses import dataclass
from typing import Optional
from PIL import Image
import pytesseract

from utils.logger import logger
from utils.timing import time_it

@dataclass
class OCRResult:
    text: str
    has_changed: bool
    similarity_with_previous: float
    timestamp: float
    duration_ms: float

class ScreenOCR:
    """Extracts text from images and detects changes across frames."""

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        similarity_threshold: float = 0.95
    ):
        self.similarity_threshold = similarity_threshold
        self.last_text: str = ""
        self._tesseract_available = self._configure_tesseract(tesseract_cmd)
        self._winocr_available = False
        if not self._tesseract_available:
            self._winocr_available = self._configure_winocr()

    def _configure_tesseract(self, explicit_path: Optional[str]) -> bool:
        """Finds and configures tesseract executable path on Windows."""
        candidate_paths = [
            explicit_path,
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
        ]
        
        for path in candidate_paths:
            if path and os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract OCR found at: {path}")
                return True
        return False

    def _configure_winocr(self) -> bool:
        """Checks and initializes Windows 10/11 built-in Media OCR."""
        try:
            import winocr
            logger.info("Using Windows built-in Media OCR (winocr) for screen text extraction.")
            return True
        except ImportError:
            logger.warning(
                "Neither Tesseract nor winocr is available. Falling back to simulated text."
            )
            return False

    def _run_winocr(self, img: Image.Image) -> str:
        """Extracts text using Windows native OCR asynchronously."""
        import winocr
        import asyncio
        async def _ocr():
            res = await winocr.recognize_pil(img, lang="en")
            return res.text
        try:
            return asyncio.run(_ocr())
        except RuntimeError:
            # If already inside an active thread loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, _ocr()).result()

    @property
    def is_available(self) -> bool:
        return self._tesseract_available or self._winocr_available

    @time_it("screen_ocr")
    def extract_text(self, img: Image.Image) -> OCRResult:
        """
        Runs OCR on the given image and compares with previous output.
        """
        start_time = time.perf_counter()
        raw_text = ""

        if self._tesseract_available:
            try:
                raw_text = pytesseract.image_to_string(img)
            except Exception as e:
                logger.error(f"Tesseract execution error: {e}")
                raw_text = ""
        elif self._winocr_available:
            try:
                raw_text = self._run_winocr(img)
            except Exception as e:
                logger.error(f"WinOCR execution error: {e}")
                raw_text = ""
        else:
            # Simulated fallback for development/testing when no OCR is available
            raw_text = self._mock_screen_text()

        # Clean text
        cleaned_text = " ".join(raw_text.split())
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Change Detection
        if not self.last_text:
            similarity = 0.0
            has_changed = bool(cleaned_text)
        else:
            matcher = difflib.SequenceMatcher(None, self.last_text, cleaned_text)
            similarity = matcher.quick_ratio()
            has_changed = similarity < self.similarity_threshold

        if has_changed:
            self.last_text = cleaned_text

        return OCRResult(
            text=cleaned_text,
            has_changed=has_changed,
            similarity_with_previous=round(similarity, 4),
            timestamp=time.time(),
            duration_ms=round(elapsed_ms, 2)
        )

    def _mock_screen_text(self) -> str:
        """Provides sample desktop context when Tesseract is not installed."""
        return (
            "Model Comparison Report:\n"
            "Random Forest: Accuracy = 91.4%, F1-Score = 0.89, Training Time = 1.2s\n"
            "Logistic Regression: Accuracy = 78.1%, F1-Score = 0.76, Training Time = 0.3s\n"
            "Status: Optimization complete on validation dataset."
        )
