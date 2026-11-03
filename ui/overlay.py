"""
Desktop UI Overlay using Tkinter with modern dark theme, transparent see-through glassmorphism,
real-time token streaming, multi-language code highlighting (C++, Java, Python, JS, TS, etc.),
and interactive language selector.
"""

import sys

# Ensure calling thread attaches to interactive Default desktop before Tkinter loads
if sys.platform == "win32":
    try:
        import ctypes
        h_default = ctypes.windll.user32.OpenDesktopW("Default", 0, False, 0x01FF)
        if h_default:
            ctypes.windll.user32.SetThreadDesktop(h_default)
    except Exception:
        pass

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
        width: int = 920,
        height: int = 660,
        min_height: int = 660,
        max_height: int = 660,
        opacity: float = 0.92,
        exclude_from_capture: bool = True
    ):
        self.width = width
        # Static fixed window: default to 660px static height
        self.height = height
        self.min_height = self.height
        self.max_height = self.height
        self.current_height = self.height
        self.is_static_window = True  # Static window: no dynamic resizing jitter
        self.opacity = opacity
        self.exclude_from_capture = exclude_from_capture
        self.entry_placeholder = "💬 Type any question (resume, code, error, or screen)..."
        self.question_history = []  # List of {"text": str, "speaker": str, "time": float}
        self.current_q_idx = -1

        from windows import ensure_interactive_desktop
        ensure_interactive_desktop()

        self.root = tk.Tk()
        self.message_queue = queue.Queue()
        self.is_paused = False
        self.is_minimized = False
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
        self._hide_from_taskbar_and_alt_tab()
        self._apply_capture_exclusion()
        # Guarantee DWM applies display affinity after window mapped
        self.root.after(200, self._apply_capture_exclusion)
        self.root.after(1000, self._apply_capture_exclusion)
        self._start_queue_poller()

    def _setup_window(self):
        """Sets window properties for floating overlay with dynamic compact start."""
        self.root.title("Workspace Overlay")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", self.opacity)
        self.root.overrideredirect(True)  # Frameless for custom clean aesthetics

        # Position at top-center of screen by default
        screen_w = self.root.winfo_screenwidth()
        x_pos = max(20, (screen_w - self.width) // 2)
        y_pos = 25
        self.root.geometry(f"{self.width}x{self.current_height}+{x_pos}+{y_pos}")
        self.root.configure(bg="#0B0F17")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

        # Local window shortcut bindings
        self.root.bind_all("<Alt-m>", lambda e: self.toggle_minimize())
        self.root.bind_all("<Alt-M>", lambda e: self.toggle_minimize())

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

        # Drag Anchor & App Logo
        drag_box = tk.Frame(header, bg="#111827", cursor="fleur")
        drag_box.pack(side=tk.LEFT, padx=(12, 6))
        drag_box.bind("<ButtonPress-1>", self._start_drag)
        drag_box.bind("<B1-Motion>", self._do_drag)

        logo_lbl = tk.Label(
            drag_box,
            text="⚡",
            font=("Segoe UI", 12),
            fg="#818CF8",
            bg="#111827",
            cursor="fleur"
        )
        logo_lbl.pack(side=tk.LEFT)
        logo_lbl.bind("<ButtonPress-1>", self._start_drag)
        logo_lbl.bind("<B1-Motion>", self._do_drag)

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

        # API Keys & Failover Pool Manager Button
        self.keys_btn = tk.Button(
            header,
            text="🔑 Keys",
            font=("Segoe UI", 8, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self.open_keys_manager_dialog
        )
        self.keys_btn.pack(side=tk.LEFT, padx=(4, 4))

        # Active Resume Label
        self.resume_badge = tk.Label(
            header,
            text="",
            font=("Segoe UI", 8, "italic"),
            fg="#34D399",
            bg="#111827"
        )
        self.resume_badge.pack(side=tk.LEFT, padx=(2, 4))

        # Close Button (✕)
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

        # Size Expand/Collapse Toggle Button (Dynamic Screen Control ⤢)
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

        # Minimize Button (—, Alt+M)
        self.minimize_btn = tk.Label(
            header,
            text="—",
            font=("Segoe UI", 11, "bold"),
            fg="#94A3B8",
            bg="#111827",
            cursor="hand2"
        )
        self.minimize_btn.pack(side=tk.RIGHT, padx=(4, 6))
        self.minimize_btn.bind("<Button-1>", lambda e: self.toggle_minimize())

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
            bg="#080C14",
            cursor="hand2"
        )
        self.exclusion_lbl.pack(side=tk.RIGHT, pady=5)
        self.exclusion_lbl.bind("<Button-1>", lambda e: self.toggle_capture_exclusion())

        # 3. Main Content Body (Fills all space between header and footer)
        content = tk.Frame(self.root, bg="#0B0F17", padx=12, pady=6)
        content.pack(fill=tk.BOTH, expand=True)

        # Card 1: Detected Query / Speech (Static card with scrollable history)
        self.query_card = tk.Frame(
            content,
            bg="#131B2A",
            padx=10,
            pady=6,
            highlightbackground="#1E293B",
            highlightthickness=1
        )
        self.query_card.pack(fill=tk.X, side=tk.TOP, pady=(0, 4))

        # Header row: Title + Navigation controls
        q_hdr_row = tk.Frame(self.query_card, bg="#131B2A")
        q_hdr_row.pack(fill=tk.X, side=tk.TOP)

        self.q_header_lbl = tk.Label(
            q_hdr_row,
            text="🎤 CURRENTLY ASKED QUESTION (Latest)",
            font=("Segoe UI", 8, "bold"),
            fg="#38BDF8",
            bg="#131B2A"
        )
        self.q_header_lbl.pack(side=tk.LEFT)

        q_nav_f = tk.Frame(q_hdr_row, bg="#131B2A")
        q_nav_f.pack(side=tk.RIGHT)

        self.q_counter_lbl = tk.Label(
            q_nav_f,
            text="Move cursor ▲ to see past questions",
            font=("Segoe UI", 7),
            fg="#64748B",
            bg="#131B2A"
        )
        self.q_counter_lbl.pack(side=tk.LEFT, padx=(0, 6))

        self.q_prev_btn = tk.Button(
            q_nav_f,
            text="▲ Older",
            font=("Segoe UI", 7, "bold"),
            bg="#1E293B",
            fg="#94A3B8",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=6,
            pady=0,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.show_prev_question
        )
        self.q_prev_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.q_next_btn = tk.Button(
            q_nav_f,
            text="▼ Current",
            font=("Segoe UI", 7, "bold"),
            bg="#1E293B",
            fg="#475569",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=6,
            pady=0,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.show_next_question
        )
        self.q_next_btn.pack(side=tk.LEFT)

        self.query_text = tk.Label(
            self.query_card,
            text="Listening for speech or ready to analyze screen...",
            font=("Segoe UI", 11, "bold"),
            fg="#FFFFFF",
            bg="#131B2A",
            wraplength=self.width - 60,
            justify="left",
            anchor="w"
        )
        self.query_text.pack(fill=tk.X, anchor="w", pady=(3, 0))

        # Mouse wheel & cursor navigation bindings on query card
        def _on_q_wheel(event):
            # Delta > 0 indicates scrolled UP -> display older previous questions
            if event.delta > 0:
                self.show_prev_question()
            elif event.delta < 0:
                self.show_next_question()

        for w_item in (self.query_card, q_hdr_row, self.q_header_lbl, self.q_counter_lbl, self.query_text):
            w_item.bind("<MouseWheel>", _on_q_wheel)
            w_item.bind("<Button-4>", lambda e: self.show_prev_question())
            w_item.bind("<Button-5>", lambda e: self.show_next_question())

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

        initial_msg = "⚡ **Assistant Ready**: Ask questions, analyze screen, or practice interviews. Code solutions and conversation will be kept in this single chat."
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
        initial_msg = "⚡ **Assistant Ready**: Ask questions, analyze screen, or practice interviews. Code solutions and conversation will be kept in this single chat."
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

    def set_window_height(self, new_height: int, force: bool = False):
        """Resizes overlay height manually only. Kept static during streaming/speech."""
        if getattr(self, "is_static_window", True) and not force:
            return
        target_h = max(self.min_height, min(self.max_height, new_height))
        if target_h == self.current_height and not force:
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
        """Kept strictly static per user request - no dynamic resizing jitter."""
        if getattr(self, "is_static_window", True):
            return
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
        if sys.platform != "win32":
            return

        self.root.update_idletasks()
        try:
            hwnd = self.root.winfo_id()
            self.exclusion_mgr.set_exclusion(hwnd, exclude=self.exclude_from_capture)
        except Exception as e:
            logger.warning(f"Could not apply display affinity: {e}")

        # Ensure modal dialogs are also hidden from screen share
        if hasattr(self, "_keys_dialog") and self._keys_dialog and self._keys_dialog.winfo_exists():
            try:
                self._keys_dialog.update_idletasks()
                self.exclusion_mgr.set_exclusion(self._keys_dialog.winfo_id(), exclude=self.exclude_from_capture)
            except Exception:
                pass

    def toggle_capture_exclusion(self):
        """Toggles between hidden from screen share (WDA_EXCLUDEFROMCAPTURE) and normal visibility."""
        self.exclude_from_capture = not self.exclude_from_capture
        self._apply_capture_exclusion()
        if self.exclude_from_capture:
            self.exclusion_lbl.configure(text="🔒 Hidden from Screen Share", fg="#34D399")
            logger.info("Screen capture exclusion ACTIVATED (hidden from Zoom/Teams/OBS/Meet).")
        else:
            self.exclusion_lbl.configure(text="👁 Capture Visible", fg="#F59E0B")
            logger.info("Screen capture exclusion DEACTIVATED (normal visibility).")

    def _hide_from_taskbar_and_alt_tab(self):
        """Ensures the borderless overlay appears in the Taskbar and stays in the topmost z-order."""
        if sys.platform != "win32":
            return
        try:
            self.root.update_idletasks()
            hwnd = self.root.winfo_id()
            GA_ROOT = 2
            user32 = ctypes.windll.user32
            top_hwnd = user32.GetAncestor(hwnd, GA_ROOT)
            target = top_hwnd if top_hwnd else hwnd

            GWL_EXSTYLE = -20
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_APPWINDOW = 0x00040000

            style = user32.GetWindowLongW(target, GWL_EXSTYLE)
            new_style = (style | WS_EX_APPWINDOW) & ~WS_EX_TOOLWINDOW
            user32.SetWindowLongW(target, GWL_EXSTYLE, new_style)

            # Enforce Topmost z-order so it never gets hidden behind other windows
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_SHOWWINDOW = 0x0040
            SWP_FRAMECHANGED = 0x0020
            user32.SetWindowPos(target, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED | SWP_SHOWWINDOW)
            self.root.lift()
            self.root.attributes("-topmost", True)
            logger.info(f"Overlay configured with Taskbar presence and topmost z-order (HWND: {target})")
        except Exception as e:
            logger.debug(f"Could not apply WS_EX_APPWINDOW: {e}")

    # Thread-Safe UI Updates via Message Queue
    def post_update(self, msg_type: str, data: dict):
        """Safely queues an update from background worker threads."""
        self.message_queue.put({"type": msg_type, "data": data})

    def _add_question_to_history(self, text: str, speaker: str = ""):
        """Stores asked questions into session history and displays latest currently asked question."""
        if not text:
            return
        clean_q = text.strip()
        if self.question_history and self.question_history[-1]["text"] == clean_q:
            return
        self.question_history.append({
            "text": clean_q,
            "speaker": speaker,
            "time": time.time()
        })
        self.current_q_idx = len(self.question_history) - 1
        self._render_question_at_index(self.current_q_idx)

    def show_prev_question(self):
        """Moves cursor/view up to review previous questions."""
        if not self.question_history or self.current_q_idx <= 0:
            return
        self.current_q_idx -= 1
        self._render_question_at_index(self.current_q_idx)

    def show_next_question(self):
        """Moves view down toward latest currently asked question."""
        if not self.question_history or self.current_q_idx >= len(self.question_history) - 1:
            return
        self.current_q_idx += 1
        self._render_question_at_index(self.current_q_idx)

    def _render_question_at_index(self, idx: int):
        """Displays question at history index with clear labels and navigation buttons."""
        if not self.question_history or idx < 0 or idx >= len(self.question_history):
            return
        entry = self.question_history[idx]
        total = len(self.question_history)
        is_latest = (idx == total - 1)
        spk = entry.get("speaker", "")
        prefix = f"[{spk}] " if spk else ""
        self.query_text.configure(text=f"{prefix}{entry['text']}")

        if is_latest:
            self.q_header_lbl.configure(text="🎤 CURRENTLY ASKED QUESTION (Latest)", fg="#38BDF8")
            self.query_card.configure(highlightbackground="#1E293B")
            self.q_counter_lbl.configure(text=f"Q {idx+1}/{total} (Move cursor ▲ for past)", fg="#64748B")
            self.q_next_btn.configure(state=tk.DISABLED, fg="#475569")
        else:
            self.q_header_lbl.configure(text=f"📜 PREVIOUS QUESTION ({idx+1} of {total})", fg="#F59E0B")
            self.query_card.configure(highlightbackground="#F59E0B")
            self.q_counter_lbl.configure(text=f"Q {idx+1}/{total} (Move cursor ▼ for current)", fg="#F59E0B")
            self.q_next_btn.configure(state=tk.NORMAL, fg="#38BDF8")

        self.q_prev_btn.configure(
            state=tk.NORMAL if idx > 0 else tk.DISABLED,
            fg="#94A3B8" if idx > 0 else "#475569"
        )

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
                    self._add_question_to_history(qtext, spk)
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
                elif mtype == "toggle_visibility":
                    self.toggle_minimize()
                elif mtype == "minimize":
                    self.minimize()
                elif mtype == "restore":
                    self.restore()
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

    def toggle_minimize(self):
        """Toggles overlay visibility between minimized (hidden) and restored."""
        if self.is_minimized:
            self.restore()
        else:
            self.minimize()

    def minimize(self):
        """Hides the overlay window into the background while audio/screen processing continues."""
        if self.is_minimized:
            return
        self.is_minimized = True
        try:
            self.root.withdraw()
            logger.info("Overlay minimized to background (Press Alt+M anytime to restore).")
        except Exception as e:
            logger.debug(f"Error minimizing overlay: {e}")

    def restore(self):
        """Restores the overlay window to foreground with capture-exclusion preserved."""
        if not self.is_minimized:
            return
        self.is_minimized = False
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.attributes("-alpha", self.opacity)
            self._apply_capture_exclusion()
            self._hide_from_taskbar_and_alt_tab()
            self.root.focus_force()
            logger.info("Overlay restored via Alt+M shortcut.")
        except Exception as e:
            logger.debug(f"Error restoring overlay: {e}")

    def open_keys_manager_dialog(self):
        """Opens a modern dark-themed modal dialog to view, test, add, and failover multiple API keys."""
        from intelligence.key_manager import key_manager
        
        # Don't open duplicates
        if hasattr(self, "_keys_dialog") and self._keys_dialog and self._keys_dialog.winfo_exists():
            self._keys_dialog.lift()
            self._keys_dialog.focus_force()
            return

        dlg = tk.Toplevel(self.root)
        self._keys_dialog = dlg
        dlg.title("API Key & Failover Manager")
        dlg.configure(bg="#0B0F17")
        dlg.attributes("-topmost", True)
        dlg.resizable(True, True)
        dlg.update_idletasks()
        if self.exclude_from_capture:
            self.exclusion_mgr.set_exclusion(dlg.winfo_id(), exclude=True)

        # Geometry and positioning near main overlay
        dlg_w = 660
        dlg_h = 580
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        pos_x = max(20, root_x - dlg_w - 15)
        pos_y = max(30, root_y)
        dlg.geometry(f"{dlg_w}x{dlg_h}+{pos_x}+{pos_y}")

        # Top Header
        header_f = tk.Frame(dlg, bg="#111827", padx=16, pady=12)
        header_f.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_f,
            text="🔑 API Key & Fallback Pool",
            font=("Segoe UI", 12, "bold"),
            fg="#F1F5F9",
            bg="#111827"
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header_f,
            text="Add backup keys across Google Gemini, OpenAI, OpenRouter, Groq & Anthropic. The assistant automatically fails over if one hits a rate limit (429) or error.",
            font=("Segoe UI", 8),
            fg="#94A3B8",
            bg="#111827",
            wraplength=600,
            justify="left"
        )
        sub_lbl.pack(anchor="w", pady=(2, 0))

        content_f = tk.Frame(dlg, bg="#0B0F17", padx=16, pady=10)
        content_f.pack(fill=tk.BOTH, expand=True)

        # Section 1: Active Keys Pool List
        sec1_lbl = tk.Label(
            content_f,
            text="ACTIVE KEY POOL (PRIORITY ORDER)",
            font=("Segoe UI", 8, "bold"),
            fg="#6366F1",
            bg="#0B0F17"
        )
        sec1_lbl.pack(anchor="w", pady=(0, 4))

        # Scrollable list frame
        list_container = tk.Frame(content_f, bg="#131B2A", highlightbackground="#1E293B", highlightthickness=1)
        list_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        canvas = tk.Canvas(list_container, bg="#131B2A", highlightthickness=0)
        scrollbar = tk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        keys_list_f = tk.Frame(canvas, bg="#131B2A")

        keys_list_f.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=keys_list_f, anchor="nw", width=620)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def refresh_keys_view():
            for child in keys_list_f.winfo_children():
                child.destroy()

            current_keys = key_manager.get_all_keys()
            if not current_keys:
                empty_lbl = tk.Label(
                    keys_list_f,
                    text="No API keys registered yet. Add one below to enable cloud reasoning!",
                    font=("Segoe UI", 9, "italic"),
                    fg="#64748B",
                    bg="#131B2A",
                    pady=20
                )
                empty_lbl.pack()
                return

            prov_colors = {
                "gemini": ("#38BDF8", "#0369A1"),
                "google": ("#38BDF8", "#0369A1"),
                "openai": ("#10B981", "#064E3B"),
                "openrouter": ("#818CF8", "#312E81"),
                "groq": ("#F59E0B", "#78350F"),
                "anthropic": ("#EC4899", "#831843"),
                "ollama": ("#A78BFA", "#4C1D95")
            }

            for idx, entry in enumerate(current_keys):
                item_row = tk.Frame(keys_list_f, bg="#172033", pady=6, padx=10)
                item_row.pack(fill=tk.X, expand=True, pady=3, padx=4)

                fg_c, bg_c = prov_colors.get(entry.provider.lower(), ("#E2E8F0", "#1E293B"))
                prov_badge = tk.Label(
                    item_row,
                    text=f"#{idx+1} {entry.provider.upper()}",
                    font=("Segoe UI", 8, "bold"),
                    fg=fg_c,
                    bg=bg_c,
                    padx=6,
                    pady=2
                )
                prov_badge.pack(side=tk.LEFT, padx=(0, 8))

                info_f = tk.Frame(item_row, bg="#172033")
                info_f.pack(side=tk.LEFT, fill=tk.X, expand=True)

                key_txt = tk.Label(
                    info_f,
                    text=f"{entry.masked_key}  ({entry.model})",
                    font=("Consolas", 9, "bold"),
                    fg="#F1F5F9",
                    bg="#172033"
                )
                key_txt.pack(anchor="w")

                lbl_txt = f"Label: {entry.label}" if entry.label else ""
                if entry.last_error:
                    lbl_txt += f" | Last Error: {entry.last_error[:40]}..."
                status_sub = tk.Label(
                    info_f,
                    text=lbl_txt or "Ready for requests",
                    font=("Segoe UI", 7),
                    fg="#94A3B8" if not entry.last_error else "#F87171",
                    bg="#172033"
                )
                status_sub.pack(anchor="w")

                status_fg = "#10B981" if entry.status == "Active" else ("#F59E0B" if "429" in entry.status else ("#EF4444" if "401" in entry.status else "#60A5FA"))
                st_lbl = tk.Label(
                    item_row,
                    text=f"● {entry.status}",
                    font=("Segoe UI", 8, "bold"),
                    fg=status_fg,
                    bg="#172033"
                )
                st_lbl.pack(side=tk.LEFT, padx=8)

                btn_box = tk.Frame(item_row, bg="#172033")
                btn_box.pack(side=tk.RIGHT)

                def make_active_cmd(k=entry.key, i=idx):
                    with key_manager._lock:
                        for k_item in key_manager.keys:
                            if k_item.status == "Active":
                                k_item.status = "Ready"
                        key_manager.current_index = i
                        key_manager.keys[i].status = "Active"
                    refresh_keys_view()

                if entry.status != "Active":
                    act_btn = tk.Button(
                        btn_box,
                        text="Promote",
                        font=("Segoe UI", 7),
                        bg="#334155",
                        fg="#FFFFFF",
                        relief=tk.FLAT,
                        padx=5,
                        pady=1,
                        cursor="hand2",
                        command=make_active_cmd
                    )
                    act_btn.pack(side=tk.LEFT, padx=2)

                def test_cmd(e=entry, row=item_row):
                    test_res_lbl = tk.Label(row, text="Testing...", font=("Segoe UI", 7), fg="#F59E0B", bg="#172033")
                    test_res_lbl.pack(side=tk.LEFT, padx=4)
                    def _run():
                        ok, msg = key_manager.test_key(e.key, e.provider, e.model)
                        dlg.after(0, lambda: test_res_lbl.configure(
                            text="✓ " + msg if ok else "✗ " + msg[:30],
                            fg="#34D399" if ok else "#F87171"
                        ))
                    import threading
                    threading.Thread(target=_run, daemon=True).start()

                t_btn = tk.Button(
                    btn_box,
                    text="Test",
                    font=("Segoe UI", 7),
                    bg="#1E293B",
                    fg="#94A3B8",
                    relief=tk.FLAT,
                    padx=5,
                    pady=1,
                    cursor="hand2",
                    command=test_cmd
                )
                t_btn.pack(side=tk.LEFT, padx=2)

                def del_cmd(k=entry.key):
                    key_manager.remove_key(k)
                    refresh_keys_view()

                del_btn = tk.Button(
                    btn_box,
                    text="✕",
                    font=("Segoe UI", 7, "bold"),
                    bg="#451A1A",
                    fg="#FCA5A5",
                    relief=tk.FLAT,
                    padx=6,
                    pady=1,
                    cursor="hand2",
                    command=del_cmd
                )
                del_btn.pack(side=tk.LEFT, padx=2)

        refresh_keys_view()

        # Section 2: Add New Key Form
        add_box = tk.LabelFrame(
            content_f,
            text=" ➕ Add New Primary or Fallback Key ",
            font=("Segoe UI", 8, "bold"),
            fg="#94A3B8",
            bg="#0B0F17",
            padx=12,
            pady=8
        )
        add_box.pack(fill=tk.X, pady=(0, 6))

        # Row 1: Provider & Model
        r1 = tk.Frame(add_box, bg="#0B0F17")
        r1.pack(fill=tk.X, pady=2)

        tk.Label(r1, text="Provider:", font=("Segoe UI", 8), fg="#CBD5E1", bg="#0B0F17").pack(side=tk.LEFT, padx=(0, 4))
        prov_var = tk.StringVar(value="Gemini")
        prov_cb = ttk.Combobox(r1, textvariable=prov_var, values=["Gemini", "OpenAI", "OpenRouter", "Groq", "Anthropic", "Local/Ollama"], width=12, state="readonly")
        prov_cb.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(r1, text="Model:", font=("Segoe UI", 8), fg="#CBD5E1", bg="#0B0F17").pack(side=tk.LEFT, padx=(0, 4))
        model_var = tk.StringVar(value="gemini-1.5-flash")
        model_ent = tk.Entry(r1, textvariable=model_var, font=("Consolas", 9), bg="#131B2A", fg="#FFFFFF", insertbackground="#FFFFFF", width=24, relief=tk.FLAT)
        model_ent.pack(side=tk.LEFT, padx=(0, 14))

        def _on_prov_changed(e=None):
            p = prov_var.get().lower()
            if "gemini" in p or "google" in p:
                model_var.set("gemini-1.5-flash")
            elif "openrouter" in p:
                model_var.set("openai/gpt-4o-mini")
            elif "groq" in p:
                model_var.set("llama-3.3-70b-versatile")
            elif "anthropic" in p:
                model_var.set("claude-3-5-sonnet-latest")
            elif "local" in p or "ollama" in p:
                model_var.set("llama3.2")
            else:
                model_var.set("gpt-4o-mini")

        prov_cb.bind("<<ComboboxSelected>>", _on_prov_changed)

        tk.Label(r1, text="Label:", font=("Segoe UI", 8), fg="#CBD5E1", bg="#0B0F17").pack(side=tk.LEFT, padx=(0, 4))
        label_var = tk.StringVar(value="Backup Key")
        label_ent = tk.Entry(r1, textvariable=label_var, font=("Segoe UI", 8), bg="#131B2A", fg="#FFFFFF", insertbackground="#FFFFFF", width=14, relief=tk.FLAT)
        label_ent.pack(side=tk.LEFT)

        # Row 2: API Key input
        r2 = tk.Frame(add_box, bg="#0B0F17")
        r2.pack(fill=tk.X, pady=(6, 4))

        tk.Label(r2, text="API Key: ", font=("Segoe UI", 8), fg="#CBD5E1", bg="#0B0F17").pack(side=tk.LEFT, padx=(0, 4))
        key_var = tk.StringVar()
        key_ent = tk.Entry(r2, textvariable=key_var, show="*", font=("Consolas", 9), bg="#131B2A", fg="#FFFFFF", insertbackground="#FFFFFF", relief=tk.FLAT)
        key_ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        def _on_key_typed(e=None):
            raw = key_var.get().strip()
            if raw.startswith("AIzaSy") or raw.startswith("AIza") or raw.startswith("AQ."):
                prov_var.set("Gemini")
                model_var.set("gemini-flash-lite-latest")
            elif raw.startswith("sk-or-v1-"):
                prov_var.set("OpenRouter")
                model_var.set("openai/gpt-4o-mini")
            elif raw.startswith("gsk_"):
                prov_var.set("Groq")
                model_var.set("llama-3.3-70b-versatile")
            elif raw.startswith("sk-ant-"):
                prov_var.set("Anthropic")
                model_var.set("claude-3-5-sonnet-latest")
            elif raw.startswith("sk-") and not raw.startswith("sk-or-") and not raw.startswith("sk-ant-"):
                prov_var.set("OpenAI")
                model_var.set("gpt-4o-mini")

        key_ent.bind("<KeyRelease>", _on_key_typed)

        def _toggle_mask():
            if key_ent.cget("show") == "*":
                key_ent.configure(show="")
                show_btn.configure(text="🔒")
            else:
                key_ent.configure(show="*")
                show_btn.configure(text="👁")

        show_btn = tk.Button(r2, text="👁", font=("Segoe UI", 8), bg="#1E293B", fg="#CBD5E1", relief=tk.FLAT, command=_toggle_mask, padx=4)
        show_btn.pack(side=tk.LEFT, padx=(0, 6))

        # Status & Action Buttons
        r3 = tk.Frame(add_box, bg="#0B0F17")
        r3.pack(fill=tk.X, pady=(4, 0))

        status_msg = tk.Label(r3, text="", font=("Segoe UI", 8), fg="#94A3B8", bg="#0B0F17")
        status_msg.pack(side=tk.LEFT)

        def _do_test():
            raw_k = key_var.get().strip()
            if not raw_k:
                status_msg.configure(text="Please enter an API key first.", fg="#F87171")
                return
            status_msg.configure(text="Testing key connection...", fg="#F59E0B")
            def _bg():
                ok, msg = key_manager.test_key(raw_k, prov_var.get(), model_var.get())
                dlg.after(0, lambda: status_msg.configure(
                    text=("✓ " + msg) if ok else ("✗ " + msg),
                    fg="#34D399" if ok else "#F87171"
                ))
            import threading
            threading.Thread(target=_bg, daemon=True).start()

        test_btn = tk.Button(
            r3,
            text="🧪 Test Key",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2",
            command=_do_test
        )
        test_btn.pack(side=tk.RIGHT, padx=4)

        def _do_add():
            raw_k = key_var.get().strip()
            if not raw_k:
                status_msg.configure(text="Please enter an API key.", fg="#F87171")
                return
            p = prov_var.get()
            m = model_var.get()
            l = label_var.get()
            key_manager.add_key(raw_k, provider=p, model=m, label=l)
            key_var.set("")
            status_msg.configure(text="✓ Key added & saved to pool!", fg="#34D399")
            refresh_keys_view()

        add_btn = tk.Button(
            r3,
            text="➕ Save to Pool",
            font=("Segoe UI", 8, "bold"),
            bg="#4F46E5",
            fg="#FFFFFF",
            activebackground="#4338CA",
            relief=tk.FLAT,
            padx=12,
            cursor="hand2",
            command=_do_add
        )
        add_btn.pack(side=tk.RIGHT, padx=4)

        # Bottom Close button
        bot_f = tk.Frame(dlg, bg="#080C14", pady=8, padx=16)
        bot_f.pack(fill=tk.X, side=tk.BOTTOM)

        tip_lbl = tk.Label(
            bot_f,
            text="💡 Tip: You can also specify keys in .env (e.g. GEMINI_API_KEY, OPENAI_API_KEYS=sk-1,sk-2)",
            font=("Segoe UI", 7),
            fg="#64748B",
            bg="#080C14"
        )
        tip_lbl.pack(side=tk.LEFT)

        close_btn = tk.Button(
            bot_f,
            text="Done",
            font=("Segoe UI", 8, "bold"),
            bg="#1E293B",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=16,
            pady=2,
            cursor="hand2",
            command=dlg.destroy
        )
        close_btn.pack(side=tk.RIGHT)

