"""
Desktop UI Overlay using Tkinter with modern dark theme, transparent see-through glassmorphism,
real-time token streaming, multi-language code highlighting (C++, Java, Python, JS, TS, etc.),
and interactive language selector.
"""

import sys
import queue
import time
import re
import ctypes
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable

from utils.logger import logger
from windows.capture_exclusion import WindowsCaptureExclusion

class DesktopOverlay:
    """Spacious, transparent floating overlay displaying colorful code and real-time assistance."""

    def __init__(
        self,
        width: int = 820,
        height: int = 340,
        min_height: int = 340,
        max_height: int = 680,
        opacity: float = 0.50,
        exclude_from_capture: bool = True
    ):
        self.width = width
        self.min_height = min(min_height, height)
        self.max_height = max(max_height, height)
        self.current_height = self.min_height
        self.height = height
        self.opacity = opacity
        self.exclude_from_capture = exclude_from_capture
        self.entry_placeholder = "💬 Type any question (resume, code, error, or screen)..."

        self.root = tk.Tk()
        self.message_queue = queue.Queue()
        self.is_paused = False
        self.exclusion_mgr = WindowsCaptureExclusion()

        # Drag state variables
        self._drag_start_x = 0
        self._drag_start_y = 0

        # Callback hooks
        self.on_toggle_pause: Optional[Callable[[bool], None]] = None
        self.on_mode_change: Optional[Callable[[str], None]] = None
        self.on_manual_query: Optional[Callable[[str], None]] = None
        self.on_trigger_screen_analysis: Optional[Callable[[], None]] = None
        self.on_audio_source_change: Optional[Callable[[str], None]] = None
        self.on_resume_uploaded: Optional[Callable[[str], None]] = None

        # Audio capture mode options (Meeting/Call focus is default)
        self.audio_modes = [
            ("loopback", "🎧 Call Audio", "#6EE7B7", "#064E3B"),
            ("dual", "🎙️ Dual Audio", "#A5B4FC", "#312E81"),
            ("mic", "🎤 Mic Only", "#FCD34D", "#78350F")
        ]
        self.current_audio_mode_idx = 0

        # Single unified chat session state (all code & conversation kept in one feed)
        self.pinned_code_text: str = ""
        self.pinned_code_title: str = ""
        self.current_view_mode: str = "single_chat"
        self.is_first_turn: bool = True
        self._stream_start_index: Optional[str] = None

        self._setup_window()
        self._build_widgets()
        self._setup_text_tags()
        self._apply_capture_exclusion()
        self._start_queue_poller()

    def _setup_window(self):
        """Sets window properties for floating overlay with dynamic compact start."""
        self.root.title("Multimodal AI Assistant")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", self.opacity)
        self.root.overrideredirect(True)  # Frameless for custom clean aesthetics

        # Position at top-right of screen by default
        screen_w = self.root.winfo_screenwidth()
        x_pos = max(20, screen_w - self.width - 25)
        y_pos = 35
        self.root.geometry(f"{self.width}x{self.current_height}+{x_pos}+{y_pos}")
        self.root.configure(bg="#0B0F17")

    def set_opacity(self, alpha: float):
        """Dynamically adjusts window transparency so the background is see-through."""
        self.opacity = max(0.25, min(1.0, alpha))
        self.root.attributes("-alpha", self.opacity)
        logger.info(f"Overlay transparency set to: {int(self.opacity * 100)}%")

    def _build_widgets(self):
        """Constructs UI layout with modern cards and controls."""
        # 1. Draggable Top Navigation Bar
        header = tk.Frame(self.root, bg="#111827", height=46)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        # Enable dragging from header
        header.bind("<ButtonPress-1>", self._start_drag)
        header.bind("<B1-Motion>", self._do_drag)

        # App Brand & Title
        title_box = tk.Frame(header, bg="#111827")
        title_box.pack(side=tk.LEFT, padx=10)
        title_box.bind("<ButtonPress-1>", self._start_drag)
        title_box.bind("<B1-Motion>", self._do_drag)

        logo_lbl = tk.Label(
            title_box,
            text="⚡",
            font=("Segoe UI", 12),
            fg="#818CF8",
            bg="#111827"
        )
        logo_lbl.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_box,
            text="MULTIMODAL AI",
            font=("Segoe UI", 9, "bold"),
            fg="#F1F5F9",
            bg="#111827"
        )
        title_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # Primary Action: Solve Screen (LeetCode)
        solve_btn = tk.Button(
            header,
            text="🧠 Analyze Screen",
            font=("Segoe UI", 9, "bold"),
            bg="#4F46E5",
            fg="#FFFFFF",
            activebackground="#4338CA",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            cursor="hand2",
            command=self._handle_solve_screen_clicked
        )
        solve_btn.pack(side=tk.LEFT, padx=(6, 4))

        # Language Selector Combobox
        lang_box = tk.Frame(header, bg="#111827")
        lang_box.pack(side=tk.LEFT, padx=4)

        self.lang_var = tk.StringVar(value="Auto-Detect")
        self.lang_dropdown = ttk.Combobox(
            lang_box,
            textvariable=self.lang_var,
            values=["Auto-Detect", "C++", "Java", "Python", "JavaScript", "TypeScript", "C#", "Go", "Rust", "SQL"],
            width=11,
            state="readonly",
            font=("Segoe UI", 8)
        )
        self.lang_dropdown.pack(side=tk.LEFT, pady=2)

        # Audio Capture Mode Selector Button (Call Focus / Dual / Mic)
        self.audio_mode_btn = tk.Button(
            header,
            text="🎧 Call Audio",
            font=("Segoe UI", 8, "bold"),
            bg="#064E3B",
            fg="#6EE7B7",
            activebackground="#059669",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._cycle_audio_mode
        )
        self.audio_mode_btn.pack(side=tk.LEFT, padx=(4, 4))

        # Resume Upload Button
        self.resume_btn = tk.Button(
            header,
            text="📄 Upload Resume",
            font=("Segoe UI", 8, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._handle_upload_resume_clicked
        )
        self.resume_btn.pack(side=tk.LEFT, padx=(4, 4))

        # Active Resume Label
        self.resume_badge = tk.Label(
            header,
            text="",
            font=("Segoe UI", 8, "italic"),
            fg="#34D399",
            bg="#111827"
        )
        self.resume_badge.pack(side=tk.LEFT, padx=(2, 4))

        # Size Expand/Collapse Toggle Button (Dynamic Screen Control)
        self.size_toggle_btn = tk.Label(
            header,
            text="⤢",
            font=("Segoe UI", 11, "bold"),
            fg="#94A3B8",
            bg="#111827",
            cursor="hand2"
        )
        self.size_toggle_btn.pack(side=tk.RIGHT, padx=(4, 6))
        self.size_toggle_btn.bind("<Button-1>", lambda e: self.toggle_size())

        # Close Button
        close_btn = tk.Label(
            header,
            text="✕",
            font=("Segoe UI", 12, "bold"),
            fg="#94A3B8",
            bg="#111827",
            cursor="hand2"
        )
        close_btn.pack(side=tk.RIGHT, padx=12)
        close_btn.bind("<Button-1>", lambda e: self.close())

        # Status Badge
        self.status_lbl = tk.Label(
            header,
            text="● LISTENING",
            font=("Segoe UI", 8, "bold"),
            fg="#10B981",
            bg="#111827"
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=10)

        # 2. Bottom Footer Bar (Pack first with side=BOTTOM so it always claims its 36px at bottom)
        footer = tk.Frame(self.root, bg="#080C14", height=36, padx=12)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        # Pause / Resume Button
        self.pause_btn = tk.Button(
            footer,
            text="Pause",
            font=("Segoe UI", 8),
            bg="#1E293B",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2",
            command=self._handle_toggle_pause
        )
        self.pause_btn.pack(side=tk.LEFT, pady=5)

        # Latency Metric Display
        self.latency_lbl = tk.Label(
            footer,
            text="E2E: -- ms",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#080C14"
        )
        self.latency_lbl.pack(side=tk.LEFT, padx=12, pady=5)

        # Capture Exclusion Status Indicator
        self.exclusion_lbl = tk.Label(
            footer,
            text="🔒 Hidden from Screen Share" if self.exclude_from_capture else "👁 Capture Visible",
            font=("Segoe UI", 8),
            fg="#34D399" if self.exclude_from_capture else "#F59E0B",
            bg="#080C14"
        )
        self.exclusion_lbl.pack(side=tk.RIGHT, pady=5)

        # 3. Main Content Body (Fills all space between header and footer)
        content = tk.Frame(self.root, bg="#0B0F17", padx=12, pady=6)
        content.pack(fill=tk.BOTH, expand=True)

        # Card 1: Detected Query / Speech
        query_card = tk.Frame(
            content,
            bg="#131B2A",
            padx=10,
            pady=5,
            highlightbackground="#1E293B",
            highlightthickness=1
        )
        query_card.pack(fill=tk.X, side=tk.TOP, pady=(0, 4))

        q_header = tk.Label(
            query_card,
            text="🎤 DETECTED SPEECH / ACTIVE QUERY",
            font=("Segoe UI", 8, "bold"),
            fg="#94A3B8",
            bg="#131B2A"
        )
        q_header.pack(anchor="w")

        self.query_text = tk.Label(
            query_card,
            text="Listening for speech or ready to analyze screen...",
            font=("Segoe UI", 11, "bold"),
            fg="#FFFFFF",
            bg="#131B2A",
            wraplength=self.width - 60,
            justify="left"
        )
        self.query_text.pack(anchor="w", pady=(2, 0))

        # Quick Action Prompt Chips Bar
        chips_frame = tk.Frame(content, bg="#0B0F17")
        chips_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 4))

        chips = [
            ("💡 Direct Answer", "Analyze whatever is on screen and provide the direct answer immediately."),
            ("🎯 90s Pitch", "Give me a compelling 90-second elevator pitch based on my resume."),
            ("💼 Past Projects", "Highlight my key projects and technical architecture from my resume."),
            ("⚡ Optimal Code / Fix", "Provide clean, optimal code or bug fix for what is visible on screen."),
            ("⚠️ Traps & Edge Cases", "Identify the key edge cases, pitfalls, or tradeoffs for this problem.")
        ]

        for label, prompt in chips:
            btn = tk.Button(
                chips_frame,
                text=label,
                font=("Segoe UI", 9, "bold"),
                bg="#1E293B",
                fg="#F8FAFC",
                activebackground="#334155",
                activeforeground="#FFFFFF",
                relief=tk.FLAT,
                padx=10,
                pady=3,
                cursor="hand2",
                command=lambda p=prompt, l=label: self._trigger_chip_query(l, p)
            )
            btn.pack(side=tk.LEFT, padx=(0, 6))

        # 4. Interactive Chat Bar (Positioned above the answer screen)
        chat_container = tk.Frame(content, bg="#0B0F17")
        chat_container.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        chat_lbl = tk.Label(
            chat_container,
            text="💬 CHAT & ASK AI (Ask anything about screen, code, error, or your resume):",
            font=("Segoe UI", 8, "bold"),
            fg="#94A3B8",
            bg="#0B0F17"
        )
        chat_lbl.pack(anchor="w", pady=(0, 2))

        input_bar = tk.Frame(chat_container, bg="#0B0F17")
        input_bar.pack(fill=tk.X)

        self.user_entry = tk.Entry(
            input_bar,
            bg="#0F172A",
            fg="#94A3B8",
            insertbackground="#FFFFFF",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            highlightbackground="#4F46E5",
            highlightthickness=1
        )
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 8))
        self.user_entry.insert(0, self.entry_placeholder)
        self.user_entry.bind("<FocusIn>", self._on_entry_focus_in)
        self.user_entry.bind("<FocusOut>", self._on_entry_focus_out)
        self.user_entry.bind("<Return>", lambda e: self._handle_submit_entry())

        send_btn = tk.Button(
            input_bar,
            text="💬 Ask AI ➔",
            font=("Segoe UI", 9, "bold"),
            bg="#4F46E5",
            fg="#FFFFFF",
            activebackground="#4338CA",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=14,
            pady=4,
            cursor="hand2",
            command=self._handle_submit_entry
        )
        send_btn.pack(side=tk.RIGHT)

        # 5. AI Reasoning & Single Unified Chat Box (All code and conversation kept in this single feed)
        answer_card = tk.Frame(
            content,
            bg="#0F172A",
            padx=10,
            pady=6,
            highlightbackground="#4F46E5",
            highlightthickness=2
        )
        answer_card.pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=(0, 2))

        header_row = tk.Frame(answer_card, bg="#0F172A")
        header_row.pack(fill=tk.X, side=tk.TOP, pady=(0, 4))

        self.answer_title_lbl = tk.Label(
            header_row,
            text="💡 SESSION CHAT & CODE",
            font=("Segoe UI", 9, "bold"),
            fg="#38BDF8",
            bg="#0F172A"
        )
        self.answer_title_lbl.pack(side=tk.LEFT)

        # Action shortcuts in chat header (Jump to Code, Latest, Clear)
        chat_actions = tk.Frame(header_row, bg="#0F172A")
        chat_actions.pack(side=tk.RIGHT)

        self.jump_code_btn = tk.Button(
            chat_actions,
            text="📌 Jump to Code",
            font=("Segoe UI", 8, "bold"),
            bg="#1E293B",
            fg="#64748B",
            relief=tk.FLAT,
            padx=8,
            pady=1,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.jump_to_code
        )
        self.jump_code_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.latest_btn = tk.Button(
            chat_actions,
            text="⬇ Latest",
            font=("Segoe UI", 8, "bold"),
            bg="#1E293B",
            fg="#94A3B8",
            relief=tk.FLAT,
            padx=8,
            pady=1,
            cursor="hand2",
            command=self.jump_to_latest
        )
        self.latest_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.clear_btn = tk.Button(
            chat_actions,
            text="🗑️ Clear",
            font=("Segoe UI", 8),
            bg="#1E293B",
            fg="#94A3B8",
            relief=tk.FLAT,
            padx=6,
            pady=1,
            cursor="hand2",
            command=self.clear_chat
        )
        self.clear_btn.pack(side=tk.LEFT)

        # Single Scrollable Chat Window
        chat_body = tk.Frame(answer_card, bg="#0F172A")
        chat_body.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        chat_scroll = tk.Scrollbar(chat_body)
        chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.answer_box = tk.Text(
            chat_body,
            wrap=tk.WORD,
            bg="#030712",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            selectbackground="#4338CA",
            selectforeground="#FFFFFF",
            font=("Consolas", 12, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=8,
            height=5,
            yscrollcommand=chat_scroll.set
        )
        self.answer_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chat_scroll.config(command=self.answer_box.yview)

        # Compatibility aliases for legacy tests and references
        self.pinned_code_box = self.answer_box
        self.pinned_hdr_lbl = self.answer_title_lbl
        self.tab_live_btn = self.latest_btn
        self.tab_pinned_btn = self.jump_code_btn
        self.tab_split_btn = self.latest_btn

        initial_msg = "⚡ **Multimodal AI Ready**: Ask questions, analyze screen, or practice interviews. Code solutions and conversation will be kept in this single chat."
        self._render_formatted_content(initial_msg)

    def _setup_text_tags(self):
        """Sets up high-contrast, large, bold font styling tags for crystal-clear letter visibility."""
        box = self.answer_box
        # Code Syntax Colors (Vibrant high-contrast palette, 12pt bold)
        box.tag_configure("kw", foreground="#FF79C6", font=("Consolas", 12, "bold"))          # Vivid Dracula Pink
        box.tag_configure("builtin", foreground="#50FA7B", font=("Consolas", 12, "bold"))     # Neon Mint / Green
        box.tag_configure("string", foreground="#FFF04D", font=("Consolas", 12, "bold"))      # Neon Bright Yellow
        box.tag_configure("comment", foreground="#94A3B8", font=("Consolas", 11, "bold"))     # High-contrast Slate
        box.tag_configure("number", foreground="#BD93F9", font=("Consolas", 12, "bold"))      # Neon Purple
        box.tag_configure("operator", foreground="#FF79C6", font=("Consolas", 12, "bold"))

        # Markdown / Prose Formatting (High-contrast bold headers and crisp white text)
        box.tag_configure("heading", foreground="#38BDF8", font=("Segoe UI", 13, "bold"))    # Sky Blue 13pt bold
        box.tag_configure("subheading", foreground="#818CF8", font=("Segoe UI", 12, "bold"))
        box.tag_configure("bullet", foreground="#FB7185", font=("Segoe UI", 12, "bold"))     # Rose / Coral bullet
        box.tag_configure("bold", foreground="#FFFFFF", font=("Consolas", 12, "bold"))
        box.tag_configure("plain", foreground="#FFFFFF", font=("Consolas", 12, "bold"))

        # Chat Turn Structure Tags
        box.tag_configure("turn_header", foreground="#38BDF8", font=("Segoe UI", 10, "bold"))
        box.tag_configure("code_badge", foreground="#34D399", font=("Segoe UI", 10, "bold"))
        box.tag_configure("divider", foreground="#334155", font=("Segoe UI", 9))

    def get_selected_language(self) -> str:
        """Returns the currently selected language from the dropdown."""
        return self.lang_var.get()

    def set_detected_language(self, lang: str):
        """Updates the detected language badge if present."""
        if hasattr(self, "lang_badge") and self.lang_badge:
            try:
                self.lang_badge.configure(text=f"[{lang}]", fg="#38BDF8")
            except Exception:
                pass

    def jump_to_code(self):
        """Scrolls the single chat window directly to the pinned code."""
        try:
            if "code_anchor" in self.answer_box.mark_names():
                self.answer_box.see("code_anchor")
                logger.info("Scrolled single chat to pinned code anchor")
        except Exception as e:
            logger.debug(f"jump_to_code error: {e}")

    def jump_to_latest(self):
        """Scrolls the single chat window to the latest conversation turn."""
        try:
            self.answer_box.see(tk.END)
        except Exception as e:
            logger.debug(f"jump_to_latest error: {e}")

    def clear_chat(self):
        """Clears the session chat history upon explicit user request."""
        self.answer_box.config(state=tk.NORMAL)
        self.answer_box.delete("1.0", tk.END)
        self.is_first_turn = True
        self.pinned_code_text = ""
        self.pinned_code_title = ""
        self.jump_code_btn.configure(state=tk.DISABLED, bg="#1E293B", fg="#64748B", text="📌 No Code Yet")
        initial_msg = "⚡ **Multimodal AI Ready**: Ask questions, analyze screen, or practice interviews. Code solutions and conversation will be kept in this single chat."
        self._render_formatted_content(initial_msg)
        self.set_window_height(self.min_height)

    def switch_view_mode(self, mode: str):
        """Switches view focus in the single chat (e.g. jump to pinned code or latest turn)."""
        self.current_view_mode = mode
        if mode == "pinned":
            self.jump_to_code()
        else:
            self.jump_to_latest()

    def pin_code(self, code_text: str, title: str = "Problem Code", already_streamed: bool = False):
        """Preserves code solution into the single chat session and enables quick jump shortcut."""
        if not code_text:
            return
        self.pinned_code_text = code_text
        self.pinned_code_title = title
        self.jump_code_btn.configure(state=tk.NORMAL, bg="#065F46", fg="#34D399", text="📌 Jump to Code")

        if already_streamed:
            if hasattr(self, "_stream_start_index") and self._stream_start_index:
                try:
                    self.answer_box.mark_set("code_anchor", self._stream_start_index)
                except Exception:
                    pass
            logger.info(f"Code anchored in single chat session: '{title}' ({len(code_text)} chars)")
            return

        self.answer_box.config(state=tk.NORMAL)
        if self.is_first_turn:
            self.answer_box.delete("1.0", tk.END)
            self.is_first_turn = False
        else:
            self.answer_box.insert(tk.END, "\n\n───────────────────────────────────────────────────\n\n", "divider")

        anchor_idx = self.answer_box.index("end-1c")
        try:
            self.answer_box.mark_set("code_anchor", anchor_idx)
        except Exception:
            pass

        self.answer_box.insert(tk.END, f"📌 PINNED CODE: {title}\n\n", "code_badge")
        self._render_turn_content(code_text)
        self.answer_box.see(tk.END)
        self.answer_box.config(state=tk.DISABLED)
        self.auto_resize_to_content(force=True)
        logger.info(f"Code inserted into single chat session: '{title}' ({len(code_text)} chars)")

    def _render_turn_content(self, raw_text: str):
        """Formats and inserts text lines with colorful syntax highlighting at the end of the chat."""
        lines = raw_text.split("\n")
        in_code_block = False

        # Multi-Language Keywords (C++, Java, Python, JS, TS, Go, Rust)
        KEYWORDS = {
            # C++ / C
            "class", "struct", "public", "private", "protected", "virtual", "override",
            "nullptr", "auto", "const", "static", "new", "delete", "this", "using",
            "namespace", "template", "typename", "typedef", "sizeof",
            # Common / Flow control
            "return", "if", "else", "elif", "while", "for", "in", "of", "do",
            "break", "continue", "switch", "case", "default", "goto",
            # Python
            "def", "import", "from", "as", "self", "None", "True", "False", "and", "or",
            "not", "is", "pass", "raise", "try", "except", "finally", "with", "yield", "lambda",
            # Java
            "package", "implements", "extends", "throws", "throw", "catch", "instanceof",
            "final", "super", "null", "true", "false",
            # JavaScript / TypeScript
            "let", "var", "function", "const", "await", "async", "export", "interface",
            "type", "declare", "enum",
            # Go / Rust
            "func", "fn", "mut", "pub", "impl", "trait", "match", "select", "defer", "go"
        }

        BUILTINS_AND_TYPES = {
            # C++
            "vector", "string", "unordered_map", "map", "unordered_set", "set",
            "pair", "queue", "stack", "priority_queue", "deque", "ListNode", "TreeNode",
            "int", "long", "short", "char", "float", "double", "bool", "void", "size_t",
            "cout", "cin", "endl", "std",
            # Java
            "Integer", "String", "Boolean", "Double", "Float", "Long", "Character",
            "List", "ArrayList", "Map", "HashMap", "Set", "HashSet", "Queue", "LinkedList",
            "System", "Math", "Arrays", "Collections",
            # Python
            "len", "range", "max", "min", "sum", "enumerate", "list", "dict", "tuple",
            "print", "sorted", "zip", "abs", "any", "all", "Dict", "Tuple", "Optional",
            # JS / TS
            "console", "Array", "Object", "Number", "Promise", "undefined", "NaN"
        }

        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                self._insert_highlighted_code_line(line, KEYWORDS, BUILTINS_AND_TYPES)
                self.answer_box.insert(tk.END, "\n")
            else:
                if line.startswith("### "):
                    self.answer_box.insert(tk.END, line[4:] + "\n", "heading")
                elif line.startswith("## ") or line.startswith("# "):
                    self.answer_box.insert(tk.END, line.lstrip("# ") + "\n", "heading")
                elif trimmed.startswith("•") or trimmed.startswith("- ") or trimmed.startswith("* "):
                    bullet_sym = "  • "
                    content = trimmed.lstrip("•-* ")
                    self.answer_box.insert(tk.END, bullet_sym, "bullet")
                    self.answer_box.insert(tk.END, content + "\n", "plain")
                else:
                    self.answer_box.insert(tk.END, line + "\n", "plain")

    def _render_formatted_content(self, raw_text: str, target_box: Optional[tk.Text] = None):
        """Renders initial or replacement content with colorful syntax highlighting in the chat."""
        box = target_box if target_box is not None else self.answer_box
        box.config(state=tk.NORMAL)
        box.delete("1.0", tk.END)
        self._render_turn_content(raw_text)
        box.see("1.0")
        box.config(state=tk.DISABLED)
        self.auto_resize_to_content()

    def set_window_height(self, new_height: int):
        """Resizes overlay height dynamically while keeping window on-screen and updating toggle icon."""
        target_h = max(self.min_height, min(self.max_height, new_height))
        if target_h == self.current_height:
            return

        try:
            curr_x = self.root.winfo_x()
            curr_y = self.root.winfo_y()
            screen_h = self.root.winfo_screenheight()

            # Prevent window from overflowing bottom of display
            if curr_y + target_h > screen_h - 40:
                curr_y = max(10, screen_h - target_h - 40)

            self.current_height = target_h
            self.root.geometry(f"{self.width}x{self.current_height}+{curr_x}+{curr_y}")

            if hasattr(self, "size_toggle_btn"):
                if self.current_height >= self.max_height - 30:
                    self.size_toggle_btn.configure(text="⤡")
                else:
                    self.size_toggle_btn.configure(text="⤢")
        except Exception as e:
            logger.debug(f"Resize overlay error: {e}")

    def auto_resize_to_content(self, force: bool = False):
        """Calculates needed height based on text content and smoothly expands window downwards."""
        try:
            content = self.answer_box.get("1.0", "end-1c").strip()
            if not content:
                content_lines = 1
            else:
                lines = content.split("\n")
                content_lines = 0
                for line in lines:
                    # In an ~820px window, ~68 characters per line for Consolas 12
                    content_lines += max(1, (len(line) + 67) // 68)

            # 2. Measure lines in query_text
            query_str = self.query_text.cget("text")
            query_display_lines = max(1, (len(query_str) + 64) // 65)
            extra_query_h = max(0, (query_display_lines - 1) * 20)

            # Base compact height (340px) accommodates ~3 lines of answer text + 1 line query
            extra_content_lines = max(0, content_lines - 3)
            target_h = self.min_height + extra_query_h + (extra_content_lines * 22)
            target_h = max(self.min_height, min(self.max_height, target_h))

            # Only resize when difference >= 15px to avoid jitter during fast token streaming
            if force or abs(target_h - self.current_height) >= 15:
                self.set_window_height(target_h)
        except Exception as e:
            logger.debug(f"auto_resize_to_content error: {e}")

    def toggle_size(self):
        """Toggles between compact minimum screen height and fully expanded maximum height."""
        if self.current_height >= self.max_height - 30:
            self.set_window_height(self.min_height)
        else:
            self.set_window_height(self.max_height)

    def _insert_highlighted_code_line(self, line: str, keywords: set, builtins: set, target_box: Optional[tk.Text] = None):
        """Tokenizes and colorizes code with multi-language syntax support (//, #, keywords, types)."""
        box = target_box if target_box is not None else self.answer_box

        # Detect comments in C++ / Java (//) or Python (#)
        hash_idx = line.find("#")
        slash_idx = line.find("//")
        if hash_idx != -1 and slash_idx != -1:
            comment_idx = min(hash_idx, slash_idx)
        elif hash_idx != -1:
            comment_idx = hash_idx
        elif slash_idx != -1:
            comment_idx = slash_idx
        else:
            comment_idx = -1

        code_part = line if comment_idx == -1 else line[:comment_idx]
        comment_part = "" if comment_idx == -1 else line[comment_idx:]

        # Tokenize code part into words, strings, numbers, operators, whitespace
        token_pattern = re.compile(r'("[^"]*"|\'[^\']*\'|\b\d+\.?\d*\b|\b[a-zA-Z_]\w*\b|[^\s\w]+|\s+)')
        tokens = token_pattern.findall(code_part)

        for tok in tokens:
            if (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
                box.insert(tk.END, tok, "string")
            elif tok in keywords:
                box.insert(tk.END, tok, "kw")
            elif tok in builtins:
                box.insert(tk.END, tok, "builtin")
            elif re.match(r'^\d+\.?\d*$', tok):
                box.insert(tk.END, tok, "number")
            elif tok in ("=", "==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "//", "%", "->", "::", ".", ":"):
                box.insert(tk.END, tok, "operator")
            else:
                box.insert(tk.END, tok, "plain")

        if comment_part:
            box.insert(tk.END, comment_part, "comment")

    def _start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _do_drag(self, event):
        delta_x = event.x - self._drag_start_x
        delta_y = event.y - self._drag_start_y
        new_x = self.root.winfo_x() + delta_x
        new_y = self.root.winfo_y() + delta_y
        self.root.geometry(f"+{new_x}+{new_y}")

    def _handle_toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.configure(text="Resume", bg="#059669")
            self.set_status("PAUSED", color="#F59E0B")
        else:
            self.pause_btn.configure(text="Pause", bg="#1E293B")
            self.set_status("LISTENING", color="#10B981")

        if self.on_toggle_pause:
            self.on_toggle_pause(self.is_paused)

    def _handle_solve_screen_clicked(self):
        if self.on_trigger_screen_analysis:
            self.set_status("READING SCREEN", color="#3B82F6")
            self.query_text.configure(
                text="[Screen Analysis]: Reading screen & analyzing content (MCQ, code, error, or architecture)...",
                font=("Segoe UI", 11, "bold")
            )
            self.on_trigger_screen_analysis()

    def _trigger_chip_query(self, label: str, prompt: str):
        self.query_text.configure(text=f"[{label}]: {prompt}", font=("Segoe UI", 11, "bold"))
        self.set_status("THINKING", color="#F59E0B")
        self.auto_resize_to_content()
        if self.on_manual_query:
            self.on_manual_query(prompt)

    def _on_entry_focus_in(self, event):
        if self.user_entry.get() == getattr(self, "entry_placeholder", ""):
            self.user_entry.delete(0, tk.END)
            self.user_entry.configure(fg="#FFFFFF", font=("Segoe UI", 10, "bold"))

    def _on_entry_focus_out(self, event):
        if not self.user_entry.get().strip():
            self.user_entry.delete(0, tk.END)
            self.user_entry.insert(0, getattr(self, "entry_placeholder", ""))
            self.user_entry.configure(fg="#94A3B8", font=("Segoe UI", 10))

    def _handle_submit_entry(self):
        text = self.user_entry.get().strip()
        if not text or text == getattr(self, "entry_placeholder", ""):
            return
        self.user_entry.delete(0, tk.END)
        self.query_text.configure(text=f"[Chat]: {text}", font=("Segoe UI", 11, "bold"))
        self.set_status("THINKING", color="#F59E0B")
        self.auto_resize_to_content()
        if self.on_manual_query:
            self.on_manual_query(text)

    def _cycle_audio_mode(self):
        """Cycles audio capture mode between Call Audio (Loopback), Dual (Call+Mic), and Mic Only."""
        self.current_audio_mode_idx = (self.current_audio_mode_idx + 1) % len(self.audio_modes)
        mode_id, label, fg_color, bg_color = self.audio_modes[self.current_audio_mode_idx]
        self.audio_mode_btn.configure(text=label, fg=fg_color, bg=bg_color)
        logger.info(f"UI switched audio mode to: {mode_id} ({label})")
        if self.on_audio_source_change:
            self.on_audio_source_change(mode_id)

    def set_audio_mode(self, mode_id: str):
        """Sets the audio mode button label and styling matching mode_id."""
        for idx, (mid, label, fg_color, bg_color) in enumerate(self.audio_modes):
            if mid == mode_id:
                self.current_audio_mode_idx = idx
                self.audio_mode_btn.configure(text=label, fg=fg_color, bg=bg_color)
                break

    def _handle_upload_resume_clicked(self):
        """Opens native file picker dialog to upload resume PDF or text file."""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select Candidate Resume or Profile Document",
            filetypes=[
                ("Supported Documents", "*.pdf;*.txt;*.md;*.markdown;*.json"),
                ("PDF Documents (*.pdf)", "*.pdf"),
                ("Text Files (*.txt, *.md)", "*.txt;*.md"),
                ("All Files (*.*)", "*.*")
            ]
        )
        if file_path:
            logger.info(f"User selected resume: {file_path}")
            if self.on_resume_uploaded:
                self.on_resume_uploaded(file_path)

    def set_resume_badge(self, filename: str):
        """Displays badge with loaded resume name."""
        if filename:
            display_name = filename if len(filename) <= 18 else f"{filename[:15]}..."
            self.resume_badge.configure(text=f"📄 {display_name}")
            self.resume_btn.configure(text="📄 Replace Resume", bg="#065F46", fg="#A7F3D0")
        else:
            self.resume_badge.configure(text="")
            self.resume_btn.configure(text="📄 Upload Resume", bg="#1E293B", fg="#F8FAFC")

    def _apply_capture_exclusion(self):
        """Applies Windows SetWindowDisplayAffinity to prevent overlay being captured."""
        if not self.exclude_from_capture or sys.platform != "win32":
            return

        self.root.update_idletasks()
        try:
            hwnd = self.root.winfo_id()
            success = self.exclusion_mgr.set_exclusion(hwnd, exclude=True)
            if success:
                logger.info("Overlay window set to WDA_EXCLUDEFROMCAPTURE")
        except Exception as e:
            logger.warning(f"Could not apply display affinity: {e}")

    # Thread-Safe UI Updates via Message Queue
    def post_update(self, msg_type: str, data: dict):
        """Safely queues an update from background worker threads."""
        self.message_queue.put({"type": msg_type, "data": data})

    def _start_queue_poller(self):
        """Regularly consumes messages on the Tkinter main thread."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                mtype = msg.get("type")
                data = msg.get("data", {})

                if mtype == "query":
                    spk = data.get("speaker", "")
                    qtext = data.get("text", "")
                    prefix = f"[{spk}] " if spk else ""
                    self.query_text.configure(text=f"{prefix}{qtext}", font=("Segoe UI", 11, "bold"))
                    self.auto_resize_to_content()
                elif mtype == "audio_mode":
                    self.set_audio_mode(data.get("mode", "loopback"))
                elif mtype == "stream_start":
                    self.answer_box.config(state=tk.NORMAL)
                    if self.is_first_turn:
                        self.answer_box.delete("1.0", tk.END)
                        self.is_first_turn = False
                    else:
                        self.answer_box.insert(tk.END, "\n\n───────────────────────────────────────────────────\n\n", "divider")

                    speaker = data.get("speaker") or self.query_text.cget("text")
                    self.answer_box.insert(tk.END, f"💡 {speaker}\n\n", "turn_header")
                    self._stream_start_index = self.answer_box.index("end-1c")
                    self.set_window_height(self.min_height)
                    self.answer_box.see(tk.END)
                elif mtype == "stream_chunk":
                    self.answer_box.config(state=tk.NORMAL)
                    self.answer_box.insert(tk.END, data.get("chunk", ""), "plain")
                    self.answer_box.see(tk.END)
                    self.auto_resize_to_content()
                elif mtype == "stream_end":
                    full_text = data.get("full_text", "")
                    self.answer_box.config(state=tk.NORMAL)
                    if hasattr(self, "_stream_start_index") and self._stream_start_index:
                        try:
                            self.answer_box.delete(self._stream_start_index, tk.END)
                        except Exception:
                            pass
                    # If this turn has code, set anchor and activate Jump button
                    if "```" in full_text or "class Solution" in full_text:
                        self.pinned_code_text = full_text
                        anchor_idx = self.answer_box.index("end-1c")
                        try:
                            self.answer_box.mark_set("code_anchor", anchor_idx)
                        except Exception:
                            pass
                        self.jump_code_btn.configure(state=tk.NORMAL, bg="#065F46", fg="#34D399", text="📌 Jump to Code")

                    self._render_turn_content(full_text)
                    self.answer_box.see(tk.END)
                    self.answer_box.config(state=tk.DISABLED)
                    self.auto_resize_to_content(force=True)
                elif mtype == "pin_code":
                    self.pin_code(data.get("code", ""), data.get("title", "Problem Code"), already_streamed=data.get("already_streamed", False))
                elif mtype == "view_mode":
                    self.switch_view_mode(data.get("mode", "live"))
                elif mtype == "detected_language":
                    self.set_detected_language(data.get("lang", "Auto"))
                elif mtype == "resume":
                    self.set_resume_badge(data.get("filename", ""))
                elif mtype == "answer":
                    self.answer_box.config(state=tk.NORMAL)
                    if self.is_first_turn:
                        self.answer_box.delete("1.0", tk.END)
                        self.is_first_turn = False
                    else:
                        self.answer_box.insert(tk.END, "\n\n───────────────────────────────────────────────────\n\n", "divider")
                    self._render_turn_content(data.get("text", ""))
                    self.answer_box.see(tk.END)
                    self.answer_box.config(state=tk.DISABLED)
                    self.auto_resize_to_content(force=True)
                elif mtype == "status":
                    self.set_status(data.get("status", "IDLE"), data.get("color", "#10B981"))
                elif mtype == "latency":
                    latency_ms = data.get("latency_ms", 0)
                    ttft_ms = data.get("ttft_ms", 0)
                    if ttft_ms > 0:
                        self.latency_lbl.configure(text=f"TTFT: {ttft_ms:.0f}ms | E2E: {latency_ms:.0f}ms")
                    else:
                        self.latency_lbl.configure(text=f"E2E: {latency_ms:.0f}ms")
        except queue.Empty:
            pass

        # Poll again in 30ms for snappy real-time streaming
        self.root.after(30, self._start_queue_poller)

    def set_status(self, text: str, color: str = "#10B981"):
        self.status_lbl.configure(text=f"● {text}", fg=color)

    def run(self):
        """Starts the Tkinter UI event loop on the main thread."""
        self.root.mainloop()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
