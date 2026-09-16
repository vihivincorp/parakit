"""
Resume and Document Ingestion Manager for grounding interview answers and chat in candidate credentials.
Extracts text from PDF, TXT, and Markdown files with local caching.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
from utils.logger import logger

CACHE_FILE = Path(__file__).resolve().parent.parent / ".resume_cache.json"

class ResumeManager:
    """Ingests and parses candidate resumes and documents for grounded reasoning."""

    def __init__(self):
        self.active_resume: Optional[Dict[str, str]] = None
        self._load_cached_resume()

    def _load_cached_resume(self):
        """Attempts to load the last uploaded resume from local cache."""
        if not CACHE_FILE.exists():
            return
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_path = data.get("path")
            if cached_path and os.path.isfile(cached_path):
                self.active_resume = data
                logger.info(f"Loaded cached resume: '{data.get('filename')}' ({data.get('char_count', 0)} chars)")
        except Exception as e:
            logger.debug(f"Could not load cached resume: {e}")

    def _save_cache(self):
        """Persists active resume metadata to disk."""
        if not self.active_resume:
            return
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.active_resume, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not persist resume cache: {e}")

    def load_document(self, file_path: str) -> Optional[Dict[str, str]]:
        """
        Parses a candidate resume file (PDF, TXT, MD) and extracts clean text.
        
        Returns:
            Dictionary with 'filename', 'path', 'text', and 'char_count', or None if failed.
        """
        path_obj = Path(file_path)
        if not path_obj.exists() or not path_obj.is_file():
            logger.error(f"File not found: {file_path}")
            return None

        ext = path_obj.suffix.lower()
        extracted_text = ""

        try:
            if ext == ".pdf":
                extracted_text = self._extract_pdf(str(path_obj))
            elif ext in (".txt", ".md", ".markdown", ".json"):
                extracted_text = self._extract_text_file(str(path_obj))
            else:
                logger.warning(f"Unsupported document format '{ext}'. Attempting raw text extraction.")
                extracted_text = self._extract_text_file(str(path_obj))

            if not extracted_text or len(extracted_text.strip()) < 20:
                logger.error(f"Failed to extract meaningful text from: {file_path}")
                return None

            clean_text = self._clean_resume_text(extracted_text)
            self.active_resume = {
                "filename": path_obj.name,
                "path": str(path_obj.resolve()),
                "text": clean_text,
                "char_count": len(clean_text)
            }
            self._save_cache()
            logger.info(f"Successfully loaded resume '{path_obj.name}' ({len(clean_text)} chars)")
            return self.active_resume

        except Exception as e:
            logger.error(f"Error loading resume document '{file_path}': {e}")
            return None

    def _extract_pdf(self, file_path: str) -> str:
        """Extracts text from PDF using PyPDF2."""
        try:
            import PyPDF2
            pages_text = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_content = page.extract_text() or ""
                    if page_content.strip():
                        pages_text.append(page_content)
            return "\n\n".join(pages_text)
        except Exception as e:
            logger.error(f"PyPDF2 extraction error on '{file_path}': {e}")
            raise

    def _extract_text_file(self, file_path: str) -> str:
        """Reads plain text / markdown file with encoding fallbacks."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode text file '{file_path}' with common encodings.")

    def _clean_resume_text(self, text: str) -> str:
        """Normalizes whitespace and removes unwanted artifacts."""
        lines = [line.strip() for line in text.splitlines()]
        # Collapse multiple blank lines
        clean_lines = []
        consecutive_blank = 0
        for line in lines:
            if not line:
                consecutive_blank += 1
                if consecutive_blank <= 1:
                    clean_lines.append("")
            else:
                consecutive_blank = 0
                clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    def get_resume_text(self) -> str:
        """Returns clean text of currently loaded resume or empty string."""
        if not self.active_resume:
            return ""
        return self.active_resume.get("text", "")

    def get_resume_filename(self) -> str:
        """Returns filename of active resume or empty string."""
        if not self.active_resume:
            return ""
        return self.active_resume.get("filename", "")

    def clear_resume(self):
        """Removes the active resume and purges cache."""
        self.active_resume = None
        if CACHE_FILE.exists():
            try:
                os.remove(CACHE_FILE)
            except Exception:
                pass
        logger.info("Active resume cleared.")
