"""
Language Detector module for LeetCode screens and code editors.
Analyzes OCR text to detect the selected programming language from UI elements or starter code syntax.
"""

import re
from typing import Tuple

class LanguageDetector:
    """Detects active programming language from screen OCR text."""

    SUPPORTED_LANGUAGES = [
        "Auto-Detect",
        "C++",
        "Java",
        "Python 3",
        "JavaScript",
        "TypeScript",
        "C#",
        "Go",
        "Rust",
        "SQL"
    ]

    @classmethod
    def detect(cls, ocr_text: str) -> Tuple[str, float]:
        """
        Analyzes OCR text from screen to identify the selected programming language.
        
        Returns:
            (language_name, confidence)
        """
        if not ocr_text:
            return "Python 3", 0.5

        text = ocr_text

        # 1. High Confidence: Explicit UI Language Dropdown / Label in LeetCode Header
        # LeetCode displays buttons like "C++", "Java", "Python3", etc. right above the editor.
        if re.search(r'\b(C\+\+|C\+\+17|C\+\+20|cpp)\b', text, re.IGNORECASE):
            return "C++", 0.95

        # Check for C#
        if re.search(r'\b(C#|CSharp)\b', text, re.IGNORECASE):
            return "C#", 0.95

        # Check for TypeScript before JavaScript
        if re.search(r'\b(TypeScript|TS)\b', text, re.IGNORECASE):
            return "TypeScript", 0.95

        # Check for JavaScript
        if re.search(r'\b(JavaScript|JS)\b', text, re.IGNORECASE):
            return "JavaScript", 0.95

        # Check for Python / Python3
        if re.search(r'\b(Python3?|py3)\b', text, re.IGNORECASE):
            # Guard against C++ code where Python might be accidentally in page text
            if not re.search(r'\b(public:|vector\s*<|std::|#include|public\s+class)\b', text):
                return "Python 3", 0.95

        # Check for Java (exclude JavaScript)
        if re.search(r'\bJava\b(?!\s*Script)', text, re.IGNORECASE):
            if not re.search(r'\b(vector\s*<|std::|#include)\b', text):
                return "Java", 0.95

        # Check for Rust
        if re.search(r'\bRust\b', text, re.IGNORECASE):
            return "Rust", 0.95

        # Check for Go / Golang
        if re.search(r'\b(Golang|Go)\b', text):
            return "Go", 0.90

        # Check for SQL problem on LeetCode
        if re.search(r'\b(MySQL|PostgreSQL|SQL\s+Schema|Write a solution to|SELECT\s+.*FROM)\b', text, re.IGNORECASE):
            return "SQL", 0.90

        # 2. Medium Confidence: Syntax Signature Heuristics from Code Editor
        # C++ distinct signatures:
        if re.search(r'\b(vector\s*<|std::|#include|unordered_map\s*<|public\s*:|ListNode\s*\*|TreeNode\s*\*|nullptr|cout\s*<<)', text):
            return "C++", 0.90

        # Java distinct signatures:
        if re.search(r'\b(public\s+class\s+Solution|public\s+(int\[\]|void|String|boolean|List<|int|double)|System\.out\.println|HashMap<)', text):
            return "Java", 0.90

        # Python distinct signatures:
        if re.search(r'\b(def\s+\w+\s*\(self|class\s+Solution\s*:|->\s*List\[|import\s+typing)', text):
            return "Python 3", 0.90

        # TypeScript distinct signatures:
        if re.search(r'(:\s*(number\[\]|string\[\]|boolean\[\])|function\s+\w+\(.*\):\s*(number|string|void)|interface\s+\w+)', text):
            return "TypeScript", 0.85

        # JavaScript distinct signatures:
        if re.search(r'\b(var\s+\w+\s*=\s*function|const\s+\w+\s*=\s*function|let\s+\w+|console\.log)', text):
            return "JavaScript", 0.85

        # Go distinct signatures:
        if re.search(r'\b(func\s+\w+\s*\(|\[\]int|package\s+\w+)', text):
            return "Go", 0.85

        # Rust distinct signatures:
        if re.search(r'\b(impl\s+Solution|pub\s+fn\s+\w+|Vec<i32>)', text):
            return "Rust", 0.85

        # Default fallback
        return "Python 3", 0.60
