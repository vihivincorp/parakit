"""
LLM Reasoning Layer interfacing with OpenRouter, OpenAI, Groq, or Anthropic.
Provides real-time token streaming, multi-key failover pool, and dynamic multi-language generation.
"""

import os
import time
from typing import Optional, Dict, Generator
from utils.logger import logger
from utils.timing import time_it
from .key_manager import key_manager, KeyManager, APIKeyEntry

class LLMClient:
    """Multimodal reasoning client with automatic multi-key failover and screen-language intelligence."""

    SYSTEM_PROMPT = (
        "You are an expert real-time AI desktop copilot for technical assessments, coding interviews, and live meetings (Zoom, Google Meet, MS Teams).\n"
        "You observe whatever is visible on the user's screen, capture live audio/questions, and support the user during technical interviews.\n\n"
        "ROLE & IDENTITY DIRECTIVE (CRITICAL - HIGHEST PRIORITY):\n"
        "- When a candidate resume is provided OR when asked about projects, past experience, technical challenges, architecture decisions, or behavioral scenarios:\n"
        "  YOU ARE THE CANDIDATE. Speak strictly and authentically in the FIRST PERSON ('I', 'my project', 'we').\n"
        "- NEVER speak as a third-party observer, detached advisor, or textbook (e.g. NEVER say 'Challenges faced while developing X often include...', 'Candidates typically...', or 'For instance, in my experience with...').\n"
        "- When a project name or alias is mentioned (e.g. 'HireMatrix', 'career intelligence', 'GitProof', 'developer intelligence', 'TextMorph'):\n"
        "  Map it directly to the corresponding project on the resume (e.g. 'HireMatrix' -> 'AI Career Intelligence Platform') and speak about it directly as your own project.\n"
        "- For project and technical challenge questions, use the STAR / Technical Deep-Dive structure:\n"
        "   1. Project Context (1 sentence): State what the platform is and its core tech stack (e.g. Python, FastAPI, React, NLP, LLMs, Neo4j).\n"
        "   2. Engineering Challenges & Solutions (2-3 crisp, numbered points): Specific technical hurdles (e.g. unstructured resume parsing & entity normalization, Neo4j graph traversal/schema bottlenecks, LLM latency & hallucination control) and the exact architectural solution implemented.\n"
        "   3. Measurable Impact (1 sentence): Concrete performance, latency, or accuracy result.\n"
        "- When asked for an elevator pitch (e.g. 60s or 90s pitch), construct it 100% dynamically from the candidate's resume:\n"
        "  Structure: 1) Who I am & technical focus, 2) Key technical projects & architectural highlights, 3) Proven experience / leadership, 4) Core value proposition. For a 90-second pitch, aim for ~200-230 spoken words in natural, fluent first-person conversational cadence.\n\n"
        "PHONETIC & SPEECH ERROR REPAIR (CRITICAL):\n"
        "- Speech-to-text transcripts frequently contain phonetic misspellings, soundalike words, homophones, or accent artifacts (e.g. 'higher matrix' for 'HireMatrix', 'neo 4 j' for 'Neo4j', 'lead code' for 'LeetCode', 'pie torch' for 'PyTorch', 'sequel' for 'SQL', 'to sum' for 'Two Sum').\n"
        "- Always cross-reference the visible screen context (code editor, problem statement, terminal) and candidate resume to deduce the speaker's true intended question or term.\n"
        "- Automatically correct the error in your reasoning and directly answer the true intended question seamlessly, without mentioning or criticizing the transcription typo.\n\n"
        "STRICT FOCUS ON CURRENT QUESTION (NEVER ANCHOR TO PREVIOUS TOPICS):\n"
        "- The user and interviewer move rapidly between different topics, questions, and projects.\n"
        "- Always stick strictly and exclusively to the IMMEDIATE question being asked right now.\n"
        "- DO NOT reference, repeat, or drift back into previous topics, earlier projects, or past screen code unless the user explicitly asks a follow-up question (e.g. 'why did you do that?', 'tell me more about that').\n"
        "- When a new question is recorded or asked, answer THAT question cleanly from the very first word without dragging in past history.\n\n"
        "UNIVERSAL SCREEN INTELLIGENCE:\n"
        "Analyze whatever is visible on the user's screen and adapt your answer format precisely to the content type:\n\n"
        "1. ASSESSMENT / MULTIPLE CHOICE QUESTION (MCQ):\n"
        "   - Immediately state the correct answer in the very first line (e.g. '**Correct Answer: Option B: O(log N)**').\n"
        "   - Give a 2-3 sentence clear, decisive explanation proving why this option is correct and why alternative options fail.\n\n"
        "2. CODING & ALGORITHMIC PROBLEMS (LeetCode, HackerRank, IDE, Terminal):\n"
        "   - Explain the approach & intuition in 2 clear sentences.\n"
        "   - Note critical edge cases (null/empty inputs, negative values, duplicates, overflow).\n"
        "   - Provide clean, optimal code strictly in the target programming language (matching any visible method signature or editor).\n"
        "   - State Time & Space Complexity in 1 line.\n\n"
        "3. TERMINAL / ERROR LOG / DEBUGGING:\n"
        "   - Identify the root cause in 1-2 sentences.\n"
        "   - Provide the exact terminal command or code fix to resolve the error immediately.\n\n"
        "4. SYSTEM DESIGN / ARCHITECTURE / DIAGRAMS:\n"
        "   - Break down core components, data flow, database/cache choices, and key scalability tradeoffs in concise bullet points.\n\n"
        "5. GENERAL TECHNICAL QUESTIONS & MEETING CONVERSATION (When NOT about candidate's projects/resume):\n"
        "   - 3 short conversational paragraphs: Core Explanation (2 sentences), Real-World Example (2 sentences), In Simple Terms Summary (1 sentence).\n\n"
        "RULES:\n"
        "- ZERO conversational filler (never say 'Sure!', 'Here is what you can say:', or 'Great question!'). Start directly with the answer.\n"
        "- Use **bold** sparingly for key terms, options, and critical takeaways.\n"
        "- Be concise, direct, and actionable."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model_name: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
        mock_mode: bool = False,
        mgr: Optional[KeyManager] = None
    ):
        self.key_manager = mgr or key_manager
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.mock_mode = (self.provider == "mock") or mock_mode
        self.last_ttft_ms: float = 0.0
        self._clients_cache: Dict[tuple, object] = {}

        # If a specific API key was passed directly, add/promote it in key manager
        if api_key and not self.mock_mode:
            detected_prov = self.key_manager.detect_provider(api_key)
            prov = detected_prov or (self.provider if self.provider != "mock" else "openai")
            mod = self.model_name or self.key_manager.DEFAULT_MODELS.get(prov, "gpt-4o-mini")
            if prov == "openrouter" and "/" not in mod:
                mod = f"openai/{mod}"
            self.key_manager.add_key(api_key, provider=prov, model=mod, label="Config Key")

    def _resolve_model_for_provider(self, provider: str, entry_model: str = "") -> str:
        """Determines the appropriate model name matching the target provider to prevent 404s."""
        prov = provider.lower()
        candidate = (entry_model or "").strip()

        if prov in ("gemini", "google"):
            if candidate and "gemini" in candidate.lower() and candidate not in ("gemini-1.5-flash", "gemini-2.5-flash"):
                return candidate
            if self.model_name and "gemini" in self.model_name.lower() and self.model_name not in ("gemini-1.5-flash", "gemini-2.5-flash"):
                return self.model_name
            return "gemini-flash-lite-latest"

        elif prov == "openrouter":
            if candidate and "/" in candidate:
                return candidate
            if self.model_name and "/" in self.model_name:
                return self.model_name
            if candidate:
                return f"openai/{candidate}"
            return "openai/gpt-4o-mini"

        elif prov == "groq":
            if candidate and any(k in candidate.lower() for k in ("llama", "mixtral", "gemma")):
                return candidate
            if self.model_name and any(k in self.model_name.lower() for k in ("llama", "mixtral", "gemma")):
                return self.model_name
            return "llama-3.3-70b-versatile"

        elif prov == "anthropic":
            if candidate and "claude" in candidate.lower():
                return candidate
            if self.model_name and "claude" in self.model_name.lower():
                return self.model_name
            return "claude-3-5-sonnet-latest"

        elif prov in ("ollama", "local", "lmstudio"):
            if candidate and "gpt" not in candidate.lower():
                return candidate
            return "llama3.2"

        else:  # openai
            if candidate and not any(p in candidate.lower() for p in ("gemini", "claude", "llama")):
                return candidate[7:] if candidate.startswith("openai/") else candidate
            if self.model_name and not any(p in self.model_name.lower() for p in ("gemini", "claude", "llama")):
                return self.model_name[7:] if self.model_name.startswith("openai/") else self.model_name
            return "gpt-4o-mini"

    def _get_or_create_client(self, entry: APIKeyEntry):
        """Returns a cached client or initializes a new one for the given API key entry."""
        cache_key = (entry.provider, entry.key)
        if cache_key in self._clients_cache:
            return self._clients_cache[cache_key]

        try:
            if entry.provider in ("openai", "gemini", "google", "openrouter", "groq", "ollama", "local", "lmstudio"):
                import openai
                base_url = None
                default_headers = None
                if entry.provider in ("gemini", "google"):
                    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                elif entry.provider == "openrouter":
                    base_url = "https://openrouter.ai/api/v1"
                    default_headers = {
                        "HTTP-Referer": "https://github.com/vihivincorp/parakit",
                        "X-Title": "Workspace Assistant"
                    }
                elif entry.provider == "groq":
                    base_url = "https://api.groq.com/openai/v1"
                elif entry.provider in ("ollama", "local", "lmstudio"):
                    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")

                client = openai.OpenAI(
                    api_key=entry.key,
                    base_url=base_url,
                    default_headers=default_headers,
                    max_retries=0,  # Handle retries across keys in key_manager
                    timeout=20.0
                )
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=entry.key, timeout=15.0)

            self._clients_cache[cache_key] = client
            return client
        except Exception as e:
            logger.warning(f"Could not initialize client for {entry.provider}: {e}")
            return None

    def generate_response_stream(
        self,
        screen_context: str,
        audio_context: str,
        conversation_history: str,
        current_question: str,
        target_language: str = "Auto-Detect",
        resume_context: str = ""
    ) -> Generator[str, None, None]:
        """
        Streams generated response token-by-token in real time.
        Automatically fails over across multiple API keys if rate limits (429) or errors occur.
        Falls back to offline heuristic reasoning if all keys are exhausted.
        """
        start_t = time.perf_counter()
        first_token_received = False
        self.last_ttft_ms = 0.0

        if self.mock_mode:
            self.last_ttft_ms = 10.0
            full_mock = self._mock_reasoning(current_question, screen_context, audio_context, target_language, resume_context)
            for line in full_mock.split("\n"):
                yield line + "\n"
            return

        lang_instruction = ""
        if target_language and target_language != "Auto-Detect":
            lang_instruction = (
                f"--- TARGET PROGRAMMING LANGUAGE ---\n"
                f"The user has {target_language} selected on their screen code editor.\n"
                f"CRITICAL REQUIREMENT: If writing code, write strictly in {target_language}. "
                f"Match the exact starter code signature visible on screen.\n"
                f"DO NOT output Python or any other language unless requested.\n\n"
            )

        resume_section = ""
        if resume_context:
            resume_section = (
                f"--- CANDIDATE RESUME / PROFILE (GROUND TRUTH) ---\n"
                f"{resume_context}\n\n"
                f"INTERVIEW DIRECTIVE FOR CANDIDATE PROFILE:\n"
                f"You are the candidate described above. Answer questions about background, projects, technical challenges, and achievements strictly in the FIRST PERSON ('I', 'my project', 'we').\n"
                f"If the question references 'HireMatrix', recognize that it refers to the AI Career Intelligence Platform on your resume and answer directly as the author of that project.\n\n"
            )

        prompt_content = (
            f"{lang_instruction}"
            f"{resume_section}"
            f"--- SCREEN CONTEXT (OCR) ---\n{screen_context}\n\n"
            f"--- RECENT AUDIO TRANSCRIPT ---\n{audio_context}\n\n"
            f"--- RECENT CONVERSATION ---\n{conversation_history}\n\n"
            f"--- TASK / QUESTION ---\n{current_question}"
        )

        # Attempt calls across available keys in rotation pool
        keys_pool = self.key_manager.get_all_keys()
        max_attempts = max(1, len(keys_pool))

        for attempt in range(max_attempts):
            entry = self.key_manager.get_active_key()
            if not entry or not entry.key:
                break

            client = self._get_or_create_client(entry)
            if not client:
                self.key_manager.rotate_to_next_working_key("Client initialization failed")
                continue

            model_to_use = self._resolve_model_for_provider(entry.provider, entry.model)

            try:
                if entry.provider in ("openai", "gemini", "google", "openrouter", "groq", "ollama", "local", "lmstudio"):
                    response = client.chat.completions.create(
                        model=model_to_use,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt_content}
                        ],
                        stream=True
                    )
                    tokens_streamed = 0
                    for chunk in response:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            if not first_token_received:
                                first_token_received = True
                                self.last_ttft_ms = (time.perf_counter() - start_t) * 1000.0
                            tokens_streamed += 1
                            yield delta

                    if tokens_streamed > 0:
                        self.key_manager.report_success(entry.key)
                        return

                else:
                    # Anthropic streaming
                    with client.messages.stream(
                        model=model_to_use,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=self.SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt_content}]
                    ) as stream:
                        tokens_streamed = 0
                        for text in stream.text_stream:
                            if not first_token_received and text:
                                first_token_received = True
                                self.last_ttft_ms = (time.perf_counter() - start_t) * 1000.0
                            if text:
                                tokens_streamed += 1
                                yield text

                        if tokens_streamed > 0:
                            self.key_manager.report_success(entry.key)
                            return

            except Exception as e:
                err_str = str(e)
                status_code = getattr(e, "status_code", None)
                logger.warning(
                    f"API call failed with key {entry.masked_key} ({entry.provider}): {err_str[:90]}"
                )
                next_entry = self.key_manager.report_failure(entry.key, err_str, status_code)
                if next_entry and next_entry.key != entry.key:
                    yield f"⚠️ *API Error on {entry.provider} ({entry.masked_key}). Automatically switching to fallback key ({next_entry.provider}/{next_entry.model})...*\n\n"
                    continue
                else:
                    break

        # All keys failed or no working key available
        active_k = self.key_manager.get_active_key()
        prov_info = f"{active_k.provider.upper()}" if active_k else "Cloud"
        yield f"⚠️ **API Key Error**: All {prov_info} API keys in your rotation pool are exhausted, invalid, or disconnected.\n\nPlease click 🔑 in the top header to enter a valid Gemini, OpenAI, or Groq API key."

    @time_it("llm_reasoning")
    def generate_response(
        self,
        screen_context: str,
        audio_context: str,
        conversation_history: str,
        current_question: str,
        target_language: str = "Auto-Detect",
        resume_context: str = ""
    ) -> str:
        """Synchronously gathers all tokens from the stream."""
        tokens = []
        for token in self.generate_response_stream(
            screen_context=screen_context,
            audio_context=audio_context,
            conversation_history=conversation_history,
            current_question=current_question,
            target_language=target_language,
            resume_context=resume_context
        ):
            tokens.append(token)
        return "".join(tokens).strip()

    def _mock_reasoning(self, question: str, screen: str, audio: str, target_lang: str = "Python 3", resume: str = "") -> str:
        """Internal mock generator used strictly for unit testing mock_mode without real API keys."""
        lower_q = (question or "").lower()
        lower_screen = (screen or "").lower()

        if "hirematrix" in lower_q or "challenge" in lower_q:
            return "In developing my AI Career Intelligence Platform (HireMatrix), I tackled challenges in resume parsing and Neo4j graph modeling."

        if "pitch" in lower_q or "elevator" in lower_q:
            return "Elevator Pitch: Hello! I'm Jayakrishna Gude, a Computer Science Engineer specializing in Full-Stack and AI engineering with Python, FastAPI, and LLM systems. Most notably, I built HireMatrix, an AI Career Intelligence Platform, and GitProof."

        if resume or "resume" in lower_q or "self intro" in lower_q or "introduce yourself" in lower_q:
            return "I am an AI Engineer specializing in Python, FastAPI, and LLM systems based on my candidate profile."

        if "option b" in lower_screen or "bubble sort" in lower_screen or "quick sort" in lower_screen:
            return "Correct Answer: Option B (Quick Sort runs in O(N log N))."

        if "modulenotfounderror" in lower_screen or "traceback" in lower_screen:
            return "Root Cause: Missing module. Fix: Run pip install foo to resolve the issue."

        lang = target_lang if target_lang and target_lang != "Auto-Detect" else "Python"
        if "c++" in lang.lower() or "c++" in lower_screen:
            return "Coding Problem Analysis (C++): Use a Hash Map for O(N) complexity.\n```cpp\nclass Solution { unordered_map<int, int> seen; vector<int> twoSum(vector<int>& nums, int target) {} };\n```"

        return f"Mock Analysis ({lang}): Solution for {question}"

