"""
Unit tests for Resume and Document Ingestion, Context Grounding, and Interactive Answering.
"""

import os
from pathlib import Path
from intelligence.resume_manager import ResumeManager
from intelligence.context_manager import MultimodalContextManager
from intelligence.llm_client import LLMClient

def test_resume_text_file_loading(tmp_path):
    mgr = ResumeManager()
    sample_resume = tmp_path / "sample_resume.txt"
    sample_resume.write_text(
        "John Doe\n"
        "Senior Software Engineer\n"
        "Experience: 5 years leading distributed Python and AWS microservices at CloudTech.\n"
        "Key Project: Real-time fraud detection engine handling 100k req/sec with Redis and Kafka.\n"
        "Education: B.S. in Computer Science.\n",
        encoding="utf-8"
    )

    doc = mgr.load_document(str(sample_resume))
    assert doc is not None
    assert doc["filename"] == "sample_resume.txt"
    assert "CloudTech" in doc["text"]
    assert "fraud detection" in doc["text"]
    assert mgr.get_resume_filename() == "sample_resume.txt"
    assert "John Doe" in mgr.get_resume_text()

def test_context_manager_with_resume():
    ctx = MultimodalContextManager()
    resume_text = "Jane Smith - AI Engineer at NeuroAI. Expert in PyTorch, Whisper, and LLMs."
    ctx.set_resume(resume_text, "jane_resume.pdf")

    payload = ctx.format_context_for_llm("Tell me about your work at NeuroAI")
    assert "resume_context" in payload
    assert "NeuroAI" in payload["resume_context"]
    assert "jane_resume.pdf" in payload["resume_context"]

def test_llm_client_resume_grounded_mock():
    client = LLMClient(mock_mode=True)
    resume = "Candidate is an AI Engineer with extensive experience in real-time Whisper and FastAPI."
    ans = client.generate_response(
        screen_context="Normal desktop",
        audio_context="Can you introduce yourself?",
        conversation_history="",
        current_question="Can you give me a self introduction based on my resume?",
        resume_context=resume
    )
    assert len(ans) > 20
    assert "engineer" in ans.lower() or "ai" in ans.lower() or "full-stack" in ans.lower()

def test_hirematrix_project_challenges_mock():
    client = LLMClient(mock_mode=True)
    resume = (
        "AI Career Intelligence Platform (Python, FastAPI, React, NLP, LLMs, Neo4j)\n"
        "• Developed an AI-powered career intelligence platform using NLP, LLMs, and Neo4j."
    )
    ans = client.generate_response(
        screen_context="",
        audio_context="Can you explain the challenges you faced while developing HireMatrix?",
        conversation_history="",
        current_question="explain the challenges you are faced while developing an hirematrix project",
        resume_context=resume
    )
    # Must speak in first person
    assert "i " in ans.lower() or "my " in ans.lower()
    # Must cite authentic engineering challenges and tech stack
    assert "fastapi" in ans.lower() or "neo4j" in ans.lower() or "nlp" in ans.lower()
    # Must NOT have generic 3rd person textbook disconnects
    assert "challenges faced while developing a hirematrix project often include" not in ans.lower()
    assert "for instance, in my experience with" not in ans.lower()

def test_clear_resume(tmp_path):
    mgr = ResumeManager()
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Resume\nFull Stack Developer with 4 years experience.", encoding="utf-8")
    
    mgr.load_document(str(sample_file))
    assert mgr.get_resume_filename() == "test.md"
    
    mgr.clear_resume()
    assert mgr.get_resume_text() == ""
    assert mgr.get_resume_filename() == ""

def test_90s_elevator_pitch_generation():
    client = LLMClient(mock_mode=True)
    resume = "Candidate: Jayakrishna Gude. AI Career Intelligence Platform (HireMatrix), GitProof."
    ans = client.generate_response(
        screen_context="",
        audio_context="",
        conversation_history="",
        current_question="Give me a compelling 90-second elevator pitch based on my resume.",
        resume_context=resume
    )
    assert len(ans) > 100
    assert "pitch" in ans.lower()
    assert "jayakrishna" in ans.lower()
    assert "hirematrix" in ans.lower() or "career intelligence" in ans.lower()
    assert "gitproof" in ans.lower()

def test_context_manager_on_demand_screen_and_clear_transcripts():
    from audio.speech_to_text import TranscriptSegment
    import time
    ctx = MultimodalContextManager()
    ctx.update_screen("LeetCode Problem 1: Two Sum")
    
    seg = TranscriptSegment(
        text="What are the challenges in HireMatrix?",
        start_time=1.0,
        end_time=2.0,
        confidence=0.95,
        timestamp=time.time()
    )
    ctx.add_transcript(seg)

    # 1. Spoken question that does not ask for screen: screen context is suppressed
    payload = ctx.format_context_for_llm("What are the challenges in HireMatrix?", include_screen=False)
    assert "Two Sum" not in payload["screen_context"]
    assert "No screen context needed" in payload["screen_context"]

    # 2. Clearing transcripts removes consumed speech so it does not linger
    ctx.clear_transcripts()
    payload2 = ctx.format_context_for_llm("Next question", include_screen=False)
    assert "HireMatrix" not in payload2["speech_context"]
    assert "No recent speech audio" in payload2["speech_context"]


