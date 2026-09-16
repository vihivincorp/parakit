# Multimodal Real-Time AI Desktop Assistant ⚡

A Windows-based real-time multimodal AI desktop assistant combining **computer vision (screen context via OCR)** and **speech understanding (meeting/system audio via VAD & STT)** with **Anthropic Claude** reasoning and a floating desktop overlay.

---

## 🌟 Key Architecture & Features

```
                   USER DESKTOP
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   SCREEN INPUT                      AUDIO INPUT
        │                                 │
   MSS CAPTURE                     WASAPI / AUDIO I/O
        │                                 │
IMAGE PREPROCESSING               VAD / SEGMENTATION
        │                                 │
  TESSERACT OCR                 WHISPER / FASTER-WHISPER
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             MULTIMODAL CONTEXT MANAGER
                         │
                         ▼
             QUESTION / INTENT DETECTOR
                         │
                         ▼
                    LLM / CLAUDE
                         │
                         ▼
                  RESPONSE MANAGER
                         │
                         ▼
                  TKINTER OVERLAY
```

1. **Visual Perception (Screen)**:
   - Screen capture via `MSS` (`screen/capture.py`).
   - Image enhancement & resizing with Pillow (`screen/preprocess.py`).
   - OCR via `Tesseract` with delta-change detection to avoid re-processing static frames (`screen/ocr.py`).

2. **Auditory Perception (Audio)**:
   - Live stream capture from default microphone or WASAPI loopback (`audio/capture.py`).
   - Adaptive Voice Activity Detection (`audio/vad.py`) stripping silence.
   - Timestamped Speech-to-Text via `faster-whisper` / `whisper` (`audio/speech_to_text.py`).

3. **Intelligence & Orchestration**:
   - **Multimodal Context Manager** (`intelligence/context_manager.py`): Maintains a rolling temporal window fusing visual text, recent speech utterances, and conversation turns.
   - **Question / Intent Detector** (`intelligence/question_detector.py`): Categorizes inputs (`QUESTION`, `STATEMENT`, `COMMAND`, `IRRELEVANT`) to gate expensive LLM calls.
   - **LLM Reasoning Client** (`intelligence/llm_client.py`): Multi-provider support for **OpenAI (ChatGPT: `gpt-4o-mini`, `gpt-4o`)** and **Anthropic (`claude-3-5-sonnet`)**. Synthesizes screen + audio context into concise answers.
   - **Response Manager** (`intelligence/response_manager.py`): Deduplicates, formats, and debounces UI updates.

4. **Desktop UI & Windows Research Layer**:
   - Floating, draggable, semi-transparent Tkinter overlay (`ui/overlay.py`).
   - Windows Display Affinity research layer (`windows/capture_exclusion.py`) leveraging `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)` to keep the overlay hidden in screen shares/recordings.

---

## 🚀 Getting Started

### 1. Configuration (`.env`)
Create a `.env` file or copy from `.env.example`:
```ini
# Choose 'openai' or 'anthropic'
LLM_PROVIDER=openai

# OpenAI (ChatGPT)
OPENAI_API_KEY=your_openai_api_key_here
ASSISTANT_MODEL=gpt-4o-mini

# Or Anthropic Claude:
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# ASSISTANT_MODEL=claude-3-5-sonnet-latest

OVERLAY_TRANSPARENCY=0.92
EXCLUDE_FROM_CAPTURE=true
```
*(Note: If no API key is provided, the assistant automatically runs in mock/simulation mode so you can test all UI, audio, screen, and capture-exclusion features offline).*

### 2. Running the Application

* **Run Assistant Live**:
  ```powershell
  python main.py
  ```

* **Run Automated Simulation (Simulates Screen Data + Spoken Question)**:
  ```powershell
  python main.py --simulate
  ```

* **Run Quick Dry-Run Verification**:
  ```powershell
  python main.py --dry-run
  ```

* **Select Specific Monitor or Microphone**:
  ```powershell
  python main.py --monitor 1 --mic 1
  ```

---

## 🧪 Testing

Run the automated pytest test suite:
```powershell
python -m pytest tests/ -v
```
All 12 unit and integration tests validate the Screen, Audio, Intelligence, and Windows Affinity modules.
