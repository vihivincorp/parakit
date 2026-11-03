"""
API Key Manager & Failover Pool.
Manages multiple API keys across OpenAI, OpenRouter, Groq, and Anthropic with automatic
failover on rate limits (429), quota exhaustion, or authorization errors (401).
Persists custom keys locally and provides real-time validation.
"""

import os
import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_FILE_PATH = os.path.join(BASE_DIR, "api_keys.json")

@dataclass
class APIKeyEntry:
    key: str
    provider: str  # 'openai', 'openrouter', 'groq', 'anthropic', 'ollama'
    model: str
    label: str = ""
    is_active: bool = True
    status: str = "Ready"  # 'Ready', 'Active', 'Rate Limited (429)', 'Invalid (401)', 'Error'
    failure_count: int = 0
    last_error: str = ""
    last_used: float = 0.0

    @property
    def masked_key(self) -> str:
        if not self.key or len(self.key) < 10:
            return "******"
        return f"{self.key[:6]}...{self.key[-4:]}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "APIKeyEntry":
        return cls(
            key=data.get("key", ""),
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            label=data.get("label", ""),
            is_active=data.get("is_active", True),
            status=data.get("status", "Ready"),
            failure_count=data.get("failure_count", 0),
            last_error=data.get("last_error", ""),
            last_used=data.get("last_used", 0.0)
        )


class KeyManager:
    """Manages a prioritized pool of API keys with automatic failover and live validation."""

    DEFAULT_MODELS = {
        "gemini": "gemini-flash-lite-latest",
        "google": "gemini-flash-lite-latest",
        "openai": "gpt-4o-mini",
        "openrouter": "openai/gpt-4o-mini",
        "groq": "llama-3.3-70b-versatile",
        "anthropic": "claude-3-5-sonnet-latest",
        "ollama": "llama3.2"
    }

    def __init__(self, storage_path: str = KEYS_FILE_PATH):
        self.storage_path = storage_path
        self._lock = threading.RLock()
        self.keys: List[APIKeyEntry] = []
        self.current_index: int = 0
        self.load_keys()

    @staticmethod
    def detect_provider(key: str) -> str:
        """Infers the LLM provider based on standard API key prefixes."""
        k = key.strip()
        if k.startswith("AIzaSy") or k.startswith("AIza") or k.startswith("AQ."):
            return "gemini"
        if k.startswith("sk-or-v1-"):
            return "openrouter"
        if k.startswith("gsk_"):
            return "groq"
        if k.startswith("sk-ant-"):
            return "anthropic"
        if k.startswith("sk-"):
            return "openai"
        return "openai"

    def load_keys(self):
        """Loads keys from local JSON persistence and environment variables."""
        with self._lock:
            loaded_keys: List[APIKeyEntry] = []
            seen_keys = set()

            # 1. Load from api_keys.json if present
            if os.path.exists(self.storage_path):
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data.get("keys", []):
                            entry = APIKeyEntry.from_dict(item)
                            if entry.key and entry.key not in seen_keys:
                                loaded_keys.append(entry)
                                seen_keys.add(entry.key)
                except Exception as e:
                    logger.warning(f"Could not read {self.storage_path}: {e}")

            # 2. Load from .env multi-key variables
            env_mappings = [
                ("GEMINI_API_KEYS", "gemini", "gemini-flash-lite-latest"),
                ("GOOGLE_API_KEYS", "gemini", "gemini-flash-lite-latest"),
                ("OPENAI_API_KEYS", "openai", "gpt-4o-mini"),
                ("OPENROUTER_API_KEYS", "openrouter", "openai/gpt-4o-mini"),
                ("GROQ_API_KEYS", "groq", "llama-3.3-70b-versatile"),
                ("ANTHROPIC_API_KEYS", "anthropic", "claude-3-5-sonnet-latest"),
            ]
            for env_var, prov, default_m in env_mappings:
                val = os.getenv(env_var, "").strip()
                if val:
                    for raw_k in val.split(","):
                        k = raw_k.strip()
                        if k and k not in seen_keys:
                            loaded_keys.append(APIKeyEntry(
                                key=k,
                                provider=prov,
                                model=default_m,
                                label=f"Env {prov.title()}"
                            ))
                            seen_keys.add(k)

            # 3. Load single environment variables (primary keys)
            single_vars = [
                ("GEMINI_API_KEY", "gemini"),
                ("GOOGLE_API_KEY", "gemini"),
                ("OPENAI_API_KEY", os.getenv("LLM_PROVIDER", "openai").lower()),
                ("OPENROUTER_API_KEY", "openrouter"),
                ("GROQ_API_KEY", "groq"),
                ("ANTHROPIC_API_KEY", "anthropic")
            ]
            for var_name, default_prov in single_vars:
                val = os.getenv(var_name, "").strip()
                if val and val not in seen_keys:
                    prov = self.detect_provider(val) or default_prov
                    mod = os.getenv("ASSISTANT_MODEL", self.DEFAULT_MODELS.get(prov, "gpt-4o-mini"))
                    if prov == "openrouter" and "/" not in mod:
                        mod = f"openai/{mod}"
                    loaded_keys.append(APIKeyEntry(
                        key=val,
                        provider=prov,
                        model=mod,
                        label=f"Default ({var_name})"
                    ))
                    seen_keys.add(val)

            # 4. Check FALLBACK_API_KEYS (format: key or prov:model:key)
            fallback_var = os.getenv("FALLBACK_API_KEYS", "").strip()
            if fallback_var:
                for item in fallback_var.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    parts = item.split(":")
                    if len(parts) == 3:
                        p, m, k = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    else:
                        k = item
                        p = self.detect_provider(k)
                        m = self.DEFAULT_MODELS.get(p, "gpt-4o-mini")
                    if k and k not in seen_keys:
                        loaded_keys.append(APIKeyEntry(
                            key=k,
                            provider=p,
                            model=m,
                            label="Fallback Key"
                        ))
                        seen_keys.add(k)

            self.keys = loaded_keys
            if self.keys:
                self.keys[0].status = "Active"
                self.current_index = 0
            logger.info(f"KeyManager initialized with {len(self.keys)} API keys in rotation pool.")

    def save_keys(self):
        """Persists custom keys to api_keys.json."""
        with self._lock:
            try:
                data = {"keys": [k.to_dict() for k in self.keys]}
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved {len(self.keys)} keys to {self.storage_path}")
            except Exception as e:
                logger.error(f"Error saving keys to {self.storage_path}: {e}")

    def get_active_key(self) -> Optional[APIKeyEntry]:
        """Returns the currently active key entry, or None if pool is empty."""
        with self._lock:
            if not self.keys:
                return None
            idx = min(self.current_index, len(self.keys) - 1)
            return self.keys[idx]

    def rotate_to_next_working_key(self, reason: str = "") -> Optional[APIKeyEntry]:
        """
        Advances to the next valid key in the pool, skipping permanently failed keys.
        Returns the new active entry, or None if all keys have failed.
        """
        with self._lock:
            if not self.keys:
                return None

            start_idx = self.current_index
            total = len(self.keys)

            for step in range(1, total + 1):
                next_idx = (start_idx + step) % total
                candidate = self.keys[next_idx]

                # Check if candidate is permanently disabled (401 invalid)
                if not candidate.is_active or "401" in candidate.status:
                    continue

                # Check if rate limit cooldown (60s) has passed
                if "429" in candidate.status and (time.time() - candidate.last_used < 60.0):
                    continue

                # Valid candidate found!
                self.keys[self.current_index].status = "Standby"
                self.current_index = next_idx
                candidate.status = "Active"
                candidate.last_used = time.time()
                logger.info(
                    f"Rotated to backup API key #{next_idx + 1} ({candidate.provider}/{candidate.model}, {candidate.masked_key}). Reason: {reason}"
                )
                return candidate

            logger.warning("All API keys in the failover pool are exhausted or rate-limited.")
            return None

    def report_failure(self, key_str: str, error_msg: str, status_code: Optional[int] = None) -> Optional[APIKeyEntry]:
        """
        Records an error against a key. If rate-limited or unauthorized, automatically
        rotates to the next available fallback key.
        """
        code_str = str(status_code or "")
        err_lower = error_msg.lower()

        is_auth_error = (
            "401" in code_str or "unauthorized" in err_lower or "user not found" in err_lower
            or "invalid api key" in err_lower or "please pass a valid api key" in err_lower
            or "api_key_invalid" in err_lower or "permission_denied" in err_lower
        )
        is_rate_limit = (
            "429" in code_str or "rate limit" in err_lower or "quota" in err_lower
            or "insufficient" in err_lower or "resource_exhausted" in err_lower
        )

        with self._lock:
            for entry in self.keys:
                if entry.key == key_str:
                    entry.failure_count += 1
                    entry.last_error = error_msg[:120]
                    entry.last_used = time.time()

                    if is_auth_error:
                        entry.status = "Invalid (401)"
                        entry.is_active = False
                        logger.warning(f"Key {entry.masked_key} marked Invalid (401 Unauthorized)")
                    elif is_rate_limit:
                        entry.status = "Rate Limited (429)"
                        logger.warning(f"Key {entry.masked_key} marked Rate Limited (429)")
                    else:
                        entry.status = f"Error ({code_str or 'Fail'})"
                    break

        return self.rotate_to_next_working_key(reason=error_msg[:60])

    def report_success(self, key_str: str):
        """Records a successful API call for the key, resetting temporary errors."""
        with self._lock:
            for entry in self.keys:
                if entry.key == key_str:
                    entry.status = "Active"
                    entry.last_used = time.time()
                    entry.last_error = ""
                    break

    def add_key(self, key: str, provider: str = "", model: str = "", label: str = "") -> APIKeyEntry:
        """Adds a new key to the pool and saves to local storage."""
        cleaned_key = key.strip()
        prov = (provider or self.detect_provider(cleaned_key)).lower()
        mod = model.strip() or self.DEFAULT_MODELS.get(prov, "gpt-4o-mini")
        if prov == "openrouter" and "/" not in mod:
            mod = f"openai/{mod}"

        lbl = label.strip() or f"{prov.title()} Key"

        with self._lock:
            # Check if key already exists
            for existing in self.keys:
                if existing.key == cleaned_key:
                    existing.provider = prov
                    existing.model = mod
                    existing.label = lbl
                    existing.is_active = True
                    existing.status = "Ready"
                    self.save_keys()
                    return existing

            new_entry = APIKeyEntry(
                key=cleaned_key,
                provider=prov,
                model=mod,
                label=lbl,
                is_active=True,
                status="Ready"
            )
            self.keys.append(new_entry)

        self.save_keys()
        logger.info(f"Added new {prov} API key ({new_entry.masked_key}) to pool. Total keys: {len(self.keys)}")
        return new_entry

    def remove_key(self, key: str) -> bool:
        """Removes a key from the pool and persists changes."""
        with self._lock:
            orig_len = len(self.keys)
            self.keys = [k for k in self.keys if k.key != key]
            if len(self.keys) != orig_len:
                if self.current_index >= len(self.keys):
                    self.current_index = max(0, len(self.keys) - 1)
                self.save_keys()
                return True
        return False

    def test_key(self, key: str, provider: str, model: str) -> Tuple[bool, str]:
        """
        Validates an API key live with a 1-token test prompt.
        Returns (is_valid, message).
        """
        prov = provider.lower()
        t0 = time.perf_counter()

        try:
            if prov in ("openai", "gemini", "google", "openrouter", "groq", "ollama", "local"):
                import openai
                base_url = None
                if prov in ("gemini", "google"):
                    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                elif prov == "openrouter":
                    base_url = "https://openrouter.ai/api/v1"
                elif prov == "groq":
                    base_url = "https://api.groq.com/openai/v1"
                elif prov in ("ollama", "local"):
                    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")

                client = openai.OpenAI(
                    api_key=key,
                    base_url=base_url,
                    timeout=8.0,
                    max_retries=0
                )
                test_model = model or self.DEFAULT_MODELS.get(prov, "gpt-4o-mini")
                if prov in ("gemini", "google"):
                    if not test_model or not test_model.startswith("gemini") or test_model in ("gemini-1.5-flash", "gemini-2.5-flash"):
                        test_model = "gemini-flash-lite-latest"
                if prov in ("gemini", "google") and not test_model.startswith("gemini"):
                    test_model = "gemini-1.5-flash"
                res = client.chat.completions.create(
                    model=test_model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1
                )
                latency = int((time.perf_counter() - t0) * 1000)
                return True, f"Verified! ({latency}ms latency)"

            elif prov == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=key, timeout=8.0)
                test_model = model or "claude-3-5-sonnet-latest"
                res = client.messages.create(
                    model=test_model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1
                )
                latency = int((time.perf_counter() - t0) * 1000)
                return True, f"Verified! ({latency}ms latency)"

            else:
                return False, f"Unsupported provider: {prov}"

        except Exception as e:
            err = str(e)
            if any(term in err.lower() for term in ("401", "unauthorized", "user not found", "please pass a valid api key", "api_key_invalid", "permission_denied", "invalid api key")):
                return False, "401/400 Invalid: Invalid or revoked API key"
            elif any(term in err.lower() for term in ("429", "rate limit", "quota", "resource_exhausted")):
                return False, "429 Rate Limit / Quota Exhausted"
            elif "not found" in err.lower() or "404" in err:
                return False, f"Model '{model}' not found for provider '{prov}'"
            return False, f"Connection error: {err[:80]}"

    def get_all_keys(self) -> List[APIKeyEntry]:
        """Returns copies of all key entries."""
        with self._lock:
            return list(self.keys)


# Global singleton instance
key_manager = KeyManager()
