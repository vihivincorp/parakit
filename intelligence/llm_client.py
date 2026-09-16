"""
LLM Reasoning Layer interfacing with OpenRouter, OpenAI, Groq, or Anthropic.
Provides real-time token streaming and dynamic multi-language generation matching the user's screen.
"""

import os
from typing import Optional, Dict, Generator
from utils.logger import logger
from utils.timing import time_it

class LLMClient:
    """Multimodal reasoning client with real-time streaming and screen-language intelligence."""

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
        mock_mode: bool = False
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        if self.provider == "mock":
            self.mock_mode = True
            self.api_key = ""
            self.model_name = "offline-heuristic"
        elif self.provider in ("ollama", "local", "lmstudio"):
            self.local_url = os.getenv("LOCAL_LLM_BASE_URL") or (
                "http://localhost:11434/v1" if self.provider == "ollama" else "http://localhost:1234/v1"
            )
            self.api_key = api_key or "ollama"
            self.model_name = model_name or os.getenv("ASSISTANT_MODEL", "llama3.2")
            self.mock_mode = mock_mode
        elif self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
            self.model_name = model_name or os.getenv("ASSISTANT_MODEL", "llama-3.3-70b-versatile")
        elif self.provider in ("openrouter", "openai"):
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
            # Auto-detect OpenRouter keys starting with sk-or-v1-
            if self.api_key.startswith("sk-or-v1-") or self.provider == "openrouter":
                self.provider = "openrouter"
                default_model = "openai/gpt-4o-mini"
            else:
                default_model = "gpt-4o-mini"
            self.model_name = model_name or os.getenv("ASSISTANT_MODEL", default_model)
            if self.provider == "openrouter" and "/" not in self.model_name:
                self.model_name = f"openai/{self.model_name}"
        else:
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            self.model_name = model_name or os.getenv("ASSISTANT_MODEL", "claude-3-5-sonnet-latest")

        self.max_tokens = max_tokens
        self.temperature = temperature
        if not hasattr(self, "mock_mode") or self.mock_mode is None:
            self.mock_mode = mock_mode or not bool(self.api_key)
        self.last_ttft_ms: float = 0.0

        self._client = None
        if not self.mock_mode:
            try:
                if self.provider in ("ollama", "local", "lmstudio"):
                    import openai
                    self._client = openai.OpenAI(
                        base_url=self.local_url,
                        api_key="ollama",
                        max_retries=1
                    )
                    logger.info(f"Local Offline LLM client ({self.provider}) initialized with model '{self.model_name}' at {self.local_url}")
                elif self.provider == "openrouter":
                    import openai
                    self._client = openai.OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=self.api_key,
                        max_retries=1
                    )
                    logger.info(f"OpenRouter client initialized with model '{self.model_name}'")
                elif self.provider == "groq":
                    import openai
                    self._client = openai.OpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=self.api_key,
                        max_retries=1
                    )
                    logger.info(f"Groq (Free Ultra-Fast) client initialized with model '{self.model_name}'")
                elif self.provider == "openai":
                    import openai
                    self._client = openai.OpenAI(api_key=self.api_key, max_retries=1)
                    logger.info(f"OpenAI (ChatGPT) client initialized with model '{self.model_name}'")
                else:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                    logger.info(f"Claude LLM client initialized with model '{self.model_name}'")
            except Exception as e:
                logger.warning(f"Could not initialize {self.provider} client: {e}. Switching to mock mode.")
                self.mock_mode = True
        else:
            logger.info(f"LLM client ({self.provider}) running in mock/offline mode.")

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
        Respects target_language detected from screen or selected by user.
        Integrates candidate resume context for grounded interview questions.
        Tracks TTFT (Time To First Token) for real-time latency profiling.
        """
        import time
        start_t = time.perf_counter()
        first_token_received = False
        self.last_ttft_ms = 0.0

        if self.mock_mode or not self._client:
            full_mock = self._mock_reasoning(current_question, screen_context, audio_context, target_language, resume_context)
            self.last_ttft_ms = 15.0
            # Yield line by line to simulate streaming
            lines = full_mock.split("\n")
            for line in lines:
                yield line + "\n"
            return

        lang_instruction = ""
        if target_language and target_language != "Auto-Detect":
            lang_instruction = (
                f"--- TARGET PROGRAMMING LANGUAGE ---\n"
                f"The user has {target_language} selected on their screen code editor.\n"
                f"CRITICAL REQUIREMENT: If writing code, write strictly in {target_language}. "
                f"Match the exact starter code signature (e.g. class Solution method) visible on screen.\n"
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

        try:
            if self.provider in ("openai", "groq", "openrouter", "ollama", "local", "lmstudio"):
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_content}
                    ],
                    stream=True
                )
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        if not first_token_received:
                            first_token_received = True
                            self.last_ttft_ms = (time.perf_counter() - start_t) * 1000.0
                        yield delta
            else:
                # Anthropic streaming
                with self._client.messages.stream(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt_content}]
                ) as stream:
                    for text in stream.text_stream:
                        if not first_token_received and text:
                            first_token_received = True
                            self.last_ttft_ms = (time.perf_counter() - start_t) * 1000.0
                        yield text

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Error streaming from {self.provider} API: {err_str}. Switching to offline fallback reasoning.")
            yield f"[Offline Fallback Active - Network Unavailable]\n\n"
            fallback = self._mock_reasoning(current_question, screen_context, audio_context, target_language, resume_context)
            for line in fallback.split("\n"):
                yield line + "\n"

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
        """
        Synchronously gathers all tokens from the stream.
        """
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
        """Generates realistic synthesized response matching universal screen context or resume."""
        lower_q = (question or "").lower()
        lower_screen = (screen or "").lower()

        # 0. Resume & Project Grounded Answers (First-person candidate voice)
        if "hirematrix" in lower_q or "career intelligence" in lower_q or ("challenge" in lower_q and ("project" in lower_q or "experience" in lower_q)):
            return (
                "In developing my **AI Career Intelligence Platform (HireMatrix)** (built with Python, FastAPI, React, NLP, LLMs, and Neo4j), I tackled three primary engineering challenges:\n\n"
                "1. **Unstructured Resume Parsing & Skill Normalization**: Candidate resumes varied wildly in layout, format, and terminology (e.g. 'React.js' vs 'React', 'FastAPI' vs 'Python Web'). I implemented an NLP extraction pipeline with embedding-based taxonomy deduplication to map heterogeneous terms into a standardized skill graph.\n\n"
                "2. **Graph-Based Skill-Gap Modeling in Neo4j**: Traversing relationships between candidate skills, target industry roles, and learning trajectories caused performance bottlenecks. I optimized the Cypher graph schema (`Candidate-[:HAS_SKILL]->Skill<-[:REQUIRES]-Role`) and applied index lookups to achieve sub-second match scoring.\n\n"
                "3. **LLM Latency & Roadmap Determinism**: Raw LLM generation for career roadmaps was high latency and occasionally hallucinated non-existent skills. I implemented FastAPI async endpoints, Pydantic schema validation, and structured prompt caching to deliver deterministic, structured roadmaps in real time.\n\n"
                "This architecture achieved sub-second graph query responses, eliminated extraction redundancies, and generated accurate, data-driven skill gap roadmaps."
            )

        if "pitch" in lower_q or "elevator" in lower_q:
            return (
                "**90-Second Interview Elevator Pitch**:\n\n"
                "\"Hello! I'm **Jayakrishna Gude**, a Computer Science Engineer specializing in Full-Stack and AI engineering with a focus on high-performance Python, FastAPI, and LLM systems.\n\n"
                "Over the past few years, I've focused on engineering data-driven intelligence platforms. "
                "Most notably, I built the **AI Career Intelligence Platform (HireMatrix)**, where I utilized NLP, LLMs, and Neo4j graph modeling to parse candidate resumes, perform multi-hop skill-gap analysis, and generate personalized learning trajectories.\n\n"
                "I also developed **GitProof**, a developer intelligence platform built with Next.js, TypeScript, and the GitHub REST API that evaluates real commit histories, pull requests, and code contributions into verifiable Proof-of-Work developer cards. "
                "Additionally, during my internship with Infosys Springboard on **TextMorph AI**, I integrated DistilBART and RoBERTa models to build high-efficiency document summarization and Q&A microservices, and served as a Teaching Assistant at Apna College resolving 100+ application bugs and mentoring in DSA.\n\n"
                "I thrive at the intersection of robust backend engineering, graph databases, and modern generative AI. I'm excited to bring this end-to-end engineering mindset to your team and build scalable, intelligent software.\""
            )

        if resume or "resume" in lower_q or "self intro" in lower_q or "introduce yourself" in lower_q:
            return (
                "I am a Full-Stack & AI Engineer specializing in Python, FastAPI, React, and LLM-powered intelligent systems.\n\n"
                "• **AI Career Intelligence Platform (HireMatrix)**: Built an NLP & Neo4j platform for automated resume skill-gap analysis and personalized learning trajectories.\n"
                "• **GitProof Platform**: Engineered a developer intelligence platform using Next.js, TypeScript, and GitHub REST APIs for verifiable Proof of Work evaluation.\n"
                "• **Core Strengths**: High-concurrency Python/FastAPI backends, graph data modeling in Neo4j, LLM orchestration/RAG, and structured debugging."
            )

        # 1. Assessment / Multiple Choice Question (MCQ)
        if "which of the following" in lower_screen or "option a" in lower_screen or "option b" in lower_screen or "mcq" in lower_q:
            return (
                "**Correct Answer: Option B: O(log N)**\n\n"
                "Explanation: Binary search operates by repeatedly dividing the sorted search space in half, "
                "achieving logarithmic time complexity O(log N). Linear search is O(N) and quadratic comparisons are O(N^2)."
            )

        # 2. Terminal / Error Log / Debugging
        if "traceback" in lower_screen or "error:" in lower_screen or "exception" in lower_screen:
            return (
                "**Root Cause**: Unhandled exception or missing package in runtime environment.\n\n"
                "**Recommended Fix**:\n"
                "```bash\n"
                "pip install -r requirements.txt\n"
                "```\n"
                "Ensure all environment variables in .env are populated before relaunching."
            )

        # 3. Model Accuracy / ML Questions
        if "random forest" in lower_q or "random forest" in lower_screen:
            return (
                "Random Forest was chosen because it achieved 91.4% validation accuracy "
                "and an F1-score of 0.89, outperforming Logistic Regression's 78.1%."
            )
        elif "accuracy" in lower_q or "metric" in lower_q:
            return (
                "The screen reports a model accuracy of 91.4% with an F1 score of 0.89 "
                "on the validation set."
            )

        # 4. Coding & LeetCode Solution
        lang = "Python 3" if target_lang == "Auto-Detect" else target_lang
        
        if "c++" in lang.lower():
            code_block = (
                "```cpp\n"
                "class Solution {\n"
                "public:\n"
                "    vector<int> twoSum(vector<int>& nums, int target) {\n"
                "        unordered_map<int, int> seen;\n"
                "        for (int i = 0; i < nums.size(); ++i) {\n"
                "            int diff = target - nums[i];\n"
                "            if (seen.count(diff)) return {seen[diff], i};\n"
                "            seen[nums[i]] = i;\n"
                "        }\n"
                "        return {};\n"
                "    }\n"
                "};\n"
                "```"
            )
        elif "java" in lang.lower():
            code_block = (
                "```java\n"
                "class Solution {\n"
                "    public int[] twoSum(int[] nums, int target) {\n"
                "        Map<Integer, Integer> seen = new HashMap<>();\n"
                "        for (int i = 0; i < nums.length; i++) {\n"
                "            int diff = target - nums[i];\n"
                "            if (seen.containsKey(diff)) return new int[]{seen.get(diff), i};\n"
                "            seen.put(nums[i], i);\n"
                "        }\n"
                "        return new int[]{};\n"
                "    }\n"
                "}\n"
                "```"
            )
        else:
            code_block = (
                "```python\n"
                "class Solution:\n"
                "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
                "        seen = {}\n"
                "        for i, num in enumerate(nums):\n"
                "            diff = target - num\n"
                "            if diff in seen:\n"
                "                return [seen[diff], i]\n"
                "            seen[num] = i\n"
                "        return []\n"
                "```"
            )

        return (
            f"Screen Solution ({lang}):\n"
            f"• Approach: Use a Hash Map to store seen elements and lookup complement in O(1) time.\n"
            f"• Edge Cases: Single pair matching, negative integers, duplicate values.\n"
            f"• Complexity: O(n) Time, O(n) Space.\n\n"
            f"{code_block}"
        )
