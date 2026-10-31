#!/usr/bin/env python3
"""
CSC493 RAG Chatbot — Lecture Assistant
=======================================
Built from: CSC493 Lecture, October 8, 2026
Topic: Building a RAG Chatbot with Guardrails
Author: ZAW LIN (zaw.lin1@kmutt.ac.th)

Supports:
  💻 OFFLINE mode — Ollama (local LLM) or Mock response (no setup needed)
  🌐 ONLINE mode  — Gemini, OpenAI, or Anthropic API

Usage:
  python csc493_chatbot.py

No external pip packages required for offline/mock mode.
"""

import os
import sys
import json
import math
import textwrap
import getpass
from collections import Counter
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE — Chunked from CSC493 Lecture (Oct 8, 2026)
#  This is the RAG knowledge base the chatbot retrieves from.
# ══════════════════════════════════════════════════════════════════

LECTURE_CHUNKS = [
    {
        "id": 1,
        "title": "What is RAG (Retrieval-Augmented Generation)?",
        "content": (
            "RAG stands for Retrieval-Augmented Generation. You retrieve passages from a trusted library, "
            "then generate an answer grounded in that context. RAG provides domain-specific answers without "
            "retraining the model. Benefits: up-to-date information, domain specificity, verifiable citations, "
            "and no need to retrain. RAG supports grounding — answers are backed by retrieved documents "
            "rather than model memory alone. Citations are added for higher confidence."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 2,
        "title": "RAG Pipeline Stages",
        "content": (
            "The RAG pipeline has these stages: "
            "1. Data Collection — gather papers, articles, policies. "
            "2. Parsing & Cleaning — remove headers, footers, references; keep key text. "
            "3. Chunking — split text into manageable pieces. "
            "4. Indexing — store in vector/keyword database for retrieval. "
            "5. Retrieval — find relevant chunks for a query. "
            "6. Routing — determine if query is supported, has a domain gap, or is off-topic. "
            "7. Generation — use LLM to produce grounded response. "
            "8. Verification — check facts and citations. "
            "9. Response — deliver answer with citations."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 3,
        "title": "Guardrails in RAG Systems",
        "content": (
            "Guardrails must be placed at every stage: input, retrieval, and output. "
            "Input guardrails: injection guard, PII (personal info) protection, scope guard. "
            "Retrieval guardrails: only access approved/open-access documents, require approval for new sources. "
            "Output guardrails: AI validator, content moderation, fact verification. "
            "Tools: AWS Lambda Guard, NVIDIA NeMo Guardrails, LangChain. "
            "The safety layer should NEVER be switched off — it prevents personal information leaking to LLMs. "
            "Prompt injection can be minimized but not fully eliminated; risk remains."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 4,
        "title": "Specialist vs Generalist Chatbot Debate",
        "content": (
            "The debate: should a specialist RAG chatbot also answer general questions? "
            "Specialist bots: focused on domain, politely decline off-topic questions, more trustworthy, legally binding. "
            "Generalist bots: broader scope, higher risk of incorrect domain-specific info, more attack surface. "
            "Current trend: guardrailed specialists that can handle some adjacent queries (e.g., weather for smart home). "
            "A specialist bot is judged as an official source by law — the company deploying it is responsible for its answers. "
            "You must decide how strict the scope guard is based on your use case."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 5,
        "title": "Real-World Chatbot Failures (Case Studies)",
        "content": (
            "Case 1 — Air Canada (2024): Customer service chatbot told a passenger they could claim "
            "compensation after travel, contradicting the airline's own policy. Court ruled Air Canada was "
            "responsible for chatbot information on its website and had to pay compensation. "
            "Lesson: chatbot output can be legally binding as an official company statement. "
            "Case 2 — NYC Business Advice Bot: Told users they could take workers' tips and landlords "
            "could refuse housing vouchers — both illegal in NYC. The bot was removed as 'functionally unusable'. "
            "Lesson: incorrect specialist bot advice can be illegal and seriously damage an organisation's reputation."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 6,
        "title": "H-Well Smart Home Chatbot (Class Demo)",
        "content": (
            "H-Well is the RAG chatbot demo built for CSC493 based on the paper 'Evolution of Smart Homes'. "
            "Purpose: answers questions about smart home technology for aging populations "
            "(relevant to Thailand's aging society — 20%+ of population aged 60+ as of 2025). "
            "Supports English and Thai. Has offline (mock API, free) and online versions (API key, token costs). "
            "Built by Claude Opus 5.5. Guardrails: off-topic refusal, PII protection, prompt injection prevention. "
            "Demonstrates the specialist/generalist debate — weather data may be relevant in a smart home context."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 7,
        "title": "Offline vs Online RAG Modes",
        "content": (
            "Offline RAG: uses local model (e.g., Ollama with LLaMA/Mistral), free, no internet required, "
            "less powerful. Good for privacy, development, and cost-sensitive deployments. "
            "Online RAG: requires API key (Gemini, OpenAI, Anthropic), costs tokens, more capable. "
            "KMUTT/SIT students have Gemini AI Pro (can extend for 189 THB/year; downgrade to AI Plus also possible). "
            "Your choice depends on cost, privacy, power, and connectivity requirements. "
            "For the midterm: build a chatbot for your own class project topic."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 8,
        "title": "Chatbot Design Principles",
        "content": (
            "Key principles: "
            "1. Never trust model output blindly — always verify format and content. "
            "2. Keep prompts short for chatbots (unlike full LLM sessions). "
            "3. Include citations for all responses (higher confidence, verifiability). "
            "4. Place guardrails at input, retrieval, AND output stages. "
            "5. Never auto-download or execute code without user approval. "
            "6. Verify facts — check primary sources, not just marketing blogs. "
            "7. Personalise for users while maintaining safety rules. "
            "8. Build accountability — ground responses in retrieved documents. "
            "9. Handle domain gaps gracefully — tell users when information is missing. "
            "10. Measure refusal rate AND supported answer rate, not just accuracy."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 9,
        "title": "Retrieval and Routing Logic",
        "content": (
            "The retrieval system classifies each query as: "
            "Supported — information exists in RAG → answer with citations. "
            "Domain gap — information is missing → suggest what user can do or refer to a human. "
            "Off-topic — politely decline, explain scope. "
            "Language routing: H-Well supports English and Thai; other languages are declined. "
            "Context matters: weather may be on-topic for a smart home chatbot but off-topic for a legal one. "
            "The router uses the knowledge base similarity and predefined keywords to classify."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 10,
        "title": "AI Trust, Hallucination, and Fact-Checking",
        "content": (
            "Users tend to trust AI answers without checking sources (Time Magazine study). "
            "Publishers lose traffic as users get info directly from LLMs — fewer sources → more hallucination risk. "
            "Guardrails help prevent hallucination by grounding responses in retrieved documents. "
            "Best practice: read vendor/blog content for ideas, cite primary sources, verify, measure your system. "
            "Click through and check sources — do not accept AI output blindly. "
            "Prompt injection can be minimized but NOT eliminated — risk always remains."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 11,
        "title": "Midterm Requirements (CSC493)",
        "content": (
            "Midterm is ONLINE (work from home) due to the IMF meeting in Bangkok. "
            "Week 1 (Oct 15): Receive midterm, brief discussion, submit initial work. "
            "Week 2 (Oct 22): In-person follow-up, finish midterm + new work. "
            "Requirement: build a RAG chatbot based on YOUR class project topic, with guardrails. "
            "Evaluation metrics: refusal rate, supported answer rate, accuracy — not accuracy alone. "
            "To change project topic, contact the instructor with justification."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
    {
        "id": 12,
        "title": "Chatbot Evolution — Current Era",
        "content": (
            "Six eras of chatbot evolution (from previous lecture). "
            "Current era: Agentic chatbots — they can take actions autonomously. "
            "Previous bottleneck: natural language understanding. "
            "Current bottleneck: reliability — relevant answers, attack resistance, proper guardrails. "
            "Rules and retrieval have always existed but are now formalised as guardrails and RAG. "
            "Agentic bots act in the background — without guardrails, Claude (for example) can download files, "
            "access local data, and act without asking permission, which is why guardrails are critical."
        ),
        "source": "CSC493 Lecture · Oct 8, 2026",
    },
]


# ══════════════════════════════════════════════════════════════════
#  TF-IDF RETRIEVAL ENGINE  (pure stdlib — no external packages)
# ══════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list:
    """Lowercase word tokenizer."""
    return text.lower().split()


def build_index(chunks: list) -> tuple:
    """Build TF-IDF vectors for all chunks."""
    corpus = [f"{c['title']} {c['content']}" for c in chunks]
    N = len(corpus)

    # Term frequency per document
    tf_list = []
    for doc in corpus:
        tokens = _tokenize(doc)
        count = Counter(tokens)
        total = max(len(tokens), 1)
        tf_list.append({w: n / total for w, n in count.items()})

    # Document frequency
    df = Counter()
    for tf in tf_list:
        for word in tf:
            df[word] += 1

    # IDF
    idf = {w: math.log((N + 1) / (freq + 1)) for w, freq in df.items()}

    # TF-IDF vectors
    tfidf = [{w: f * idf.get(w, 0) for w, f in tf.items()} for tf in tf_list]
    return tfidf, idf


def _cosine(vec1: dict, vec2: dict) -> float:
    common = set(vec1) & set(vec2)
    dot = sum(vec1[w] * vec2[w] for w in common)
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0


def retrieve(query: str, tfidf_index: list, idf: dict, chunks: list,
             top_k: int = 3, threshold: float = 0.04) -> list:
    """Return the top-k most relevant chunks above threshold."""
    tokens = _tokenize(query)
    count = Counter(tokens)
    total = max(len(tokens), 1)
    q_tf = {w: n / total for w, n in count.items()}
    q_vec = {w: f * idf.get(w, 0) for w, f in q_tf.items()}

    scored = sorted(
        ((i, _cosine(q_vec, doc_vec)) for i, doc_vec in enumerate(tfidf_index)),
        key=lambda x: x[1], reverse=True
    )
    return [
        {"chunk": chunks[i], "score": s}
        for i, s in scored[:top_k] if s >= threshold
    ]


# ══════════════════════════════════════════════════════════════════
#  GUARDRAIL — Topic relevance check
# ══════════════════════════════════════════════════════════════════

TOPIC_KEYWORDS = {
    "rag", "retrieval", "augmented", "generation", "chatbot", "guardrail",
    "specialist", "generalist", "llm", "language model", "ai", "artificial intelligence",
    "smart home", "hwell", "h-well", "embedding", "chunk", "index", "vector",
    "prompt", "injection", "hallucination", "citation", "grounding", "pipeline",
    "offline", "online", "ollama", "api", "gemini", "openai", "anthropic",
    "midterm", "csc493", "lecture", "assignment", "project", "class",
    "aging", "thailand", "sit", "kmutt", "air canada", "nyc", "bot",
    "safety", "security", "trust", "verify", "accuracy", "evaluation",
    "model", "data", "training", "response", "query", "search", "document",
    "paper", "research", "knowledge", "output", "input", "routing", "agentic",
    "era", "nemo", "langchain", "llama", "mistral", "gpt", "claude",
    "tokenize", "tfidf", "similarity", "cosine", "pii", "personal",
    "refusal", "scope", "domain", "gap", "supported", "off-topic",
}


def is_on_topic(query: str) -> bool:
    """Return True if the query is relevant to the lecture / RAG domain."""
    lower = query.lower()
    # Direct keyword hit
    if any(kw in lower for kw in TOPIC_KEYWORDS):
        return True
    # Token-level overlap
    q_words = set(_tokenize(lower))
    all_kw_words = {w for kw in TOPIC_KEYWORDS for w in kw.split()}
    overlap = len(q_words & all_kw_words) / max(len(q_words), 1)
    return overlap >= 0.3


# ══════════════════════════════════════════════════════════════════
#  OFFLINE GENERATION — Ollama first, mock fallback
# ══════════════════════════════════════════════════════════════════

def _try_ollama(prompt: str, model: str = "llama3.2") -> Optional[str]:
    """Try to call a local Ollama instance. Returns None on failure."""
    try:
        import urllib.request, urllib.error
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode()).get("response", "").strip()
    except Exception:
        return None


def _mock_response(query: str, retrieved: list) -> str:
    """Template-based fallback when no LLM is available."""
    if not retrieved:
        return (
            "I don't have information about that in the CSC493 lecture knowledge base. "
            "Please ask about RAG, guardrails, the chatbot case studies, "
            "H-Well, or your midterm requirements."
        )
    top = retrieved[0]["chunk"]
    sentences = [s.strip() for s in top["content"].split(".") if len(s.strip()) > 20]
    q_words = set(_tokenize(query))
    ranked = sorted(sentences, key=lambda s: len(set(_tokenize(s)) & q_words), reverse=True)
    answer = ". ".join(ranked[:4]).strip()
    if not answer.endswith("."):
        answer += "."
    if len(retrieved) > 1:
        extra = retrieved[1]["chunk"]["content"][:180].rsplit(" ", 1)[0]
        answer += f"\n\nRelated: {extra}…"
    return answer


def generate_offline(query: str, context: str, retrieved: list,
                     ollama_model: str = "llama3.2") -> tuple:
    """Generate offline response. Returns (response_text, engine_used)."""
    prompt = (
        "You are a concise teaching assistant for CSC493 (AI & Chatbot Systems) at KMUTT.\n"
        "Answer ONLY from the provided lecture context. If context is insufficient, say so.\n\n"
        f"LECTURE CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Give a clear, concise answer (under 180 words). Mention key terms from the lecture."
    )
    result = _try_ollama(prompt, ollama_model)
    if result:
        return result, f"Ollama ({ollama_model})"
    return _mock_response(query, retrieved), "Mock (template)"


# ══════════════════════════════════════════════════════════════════
#  ONLINE GENERATION — Gemini / OpenAI / Anthropic
# ══════════════════════════════════════════════════════════════════

def _build_online_prompt(query: str, context: str) -> str:
    return (
        "You are a concise teaching assistant for CSC493 (AI & Chatbot Systems) at KMUTT/SIT.\n"
        "Answer using the provided lecture context. Be educational and reference specific concepts.\n\n"
        f"LECTURE CONTEXT (Oct 8, 2026 — RAG Chatbot lecture):\n{context}\n\n"
        f"STUDENT QUESTION: {query}\n\n"
        "Provide a clear answer (under 180 words) that: "
        "directly answers the question, references lecture concepts, and adds helpful clarification."
    )


def _call_gemini(prompt: str, api_key: str) -> str:
    import urllib.request
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-1.5-flash:generateContent?key={api_key}"
    )
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openai(prompt: str, api_key: str) -> str:
    import urllib.request
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt: str, api_key: str) -> str:
    import urllib.request
    payload = json.dumps({
        "model": "claude-haiku-20240307",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["content"][0]["text"].strip()


def generate_online(query: str, context: str, provider: str, api_key: str) -> tuple:
    """Generate online response. Returns (response_text, engine_used)."""
    prompt = _build_online_prompt(query, context)
    try:
        if provider == "gemini":
            return _call_gemini(prompt, api_key), "Gemini 1.5 Flash"
        elif provider == "openai":
            return _call_openai(prompt, api_key), "GPT-4o-mini"
        elif provider == "anthropic":
            return _call_anthropic(prompt, api_key), "Claude Haiku"
        else:
            return "Unknown provider.", "Error"
    except Exception as e:
        return f"[API Error — {provider}: {e}]", "Error"


# ══════════════════════════════════════════════════════════════════
#  TERMINAL UI HELPERS
# ══════════════════════════════════════════════════════════════════

W = 64  # display width


def hr(char="─", width=W):
    return char * width


def box_print(lines: list, title: str = ""):
    """Print a bordered box with optional title."""
    top = f"┌─ {title} " + "─" * (W - 4 - len(title)) if title else "┌" + hr()
    print(top)
    for line in lines:
        wrapped = textwrap.wrap(line, width=W - 4) or [""]
        for wl in wrapped:
            print(f"│  {wl}")
    print("└" + hr())


def print_banner(mode: str, engine: str = ""):
    os.system("clear" if os.name == "posix" else "cls")
    print(hr("═"))
    print("  CSC493 RAG Chatbot  ·  Lecture Assistant".center(W))
    print("  Built from Oct 8, 2026 Lecture  ·  KMUTT/SIT".center(W))
    print(hr("═"))
    icon = "🌐" if mode == "online" else "💻"
    mode_str = f"{icon} {mode.upper()} ({engine})" if engine else f"{icon} {mode.upper()}"
    print(f"  Mode    : {mode_str}")
    print(f"  Chunks  : {len(LECTURE_CHUNKS)} lecture segments indexed")
    print(hr())
    print("  Commands: 'switch' | 'topics' | 'help' | 'quit'")
    print(hr())
    print()


def print_answer(response: str, retrieved: list, engine: str):
    print()
    box_lines = [response, "", f"Engine: {engine}"]
    print(f"┌─ Answer {'─' * (W - 10)}")
    for line in textwrap.wrap(response, width=W - 4):
        print(f"│  {line}")
    print(f"│  {'─' * (W - 4)}")
    print(f"│  Engine: {engine}")
    print("├─ Sources " + "─" * (W - 10))
    for i, r in enumerate(retrieved, 1):
        c = r["chunk"]
        print(f"│  [{i}] {c['title']}")
        print(f"│      {c['source']}  ·  relevance {r['score']:.3f}")
    print("└" + hr())
    print()


# ══════════════════════════════════════════════════════════════════
#  SETUP HELPERS
# ══════════════════════════════════════════════════════════════════

def setup_online() -> dict:
    """Interactive online-mode setup."""
    print()
    print(hr())
    print("  📡  Online Mode — API Setup")
    print(hr())
    print("  Choose provider:")
    print("    1. Gemini  (Google)   — KMUTT students may have free access")
    print("    2. OpenAI  (GPT)      — gpt-4o-mini")
    print("    3. Anthropic (Claude) — claude-haiku")
    providers = {"1": "gemini", "2": "openai", "3": "anthropic"}
    env_vars = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}
    while True:
        choice = input("\n  Choice (1/2/3): ").strip()
        if choice in providers:
            provider = providers[choice]
            break
        print("  ⚠  Enter 1, 2, or 3.")

    key = os.environ.get(env_vars[provider], "")
    if key:
        print(f"  ✓ Using {env_vars[provider]} from environment.")
    else:
        key = getpass.getpass(f"  Enter {provider.capitalize()} API key: ").strip()

    return {"provider": provider, "api_key": key}


def choose_offline_model() -> str:
    """Let user pick an Ollama model or use Mock."""
    print()
    print(hr())
    print("  💻  Offline Mode — Model Setup")
    print(hr())
    print("  1. LLaMA 3.2   (ollama pull llama3.2)")
    print("  2. Mistral      (ollama pull mistral)")
    print("  3. LLaMA 3.1   (ollama pull llama3.1)")
    print("  4. Mock         (no Ollama needed — template responses)")
    choice = input("\n  Choice (1-4, default=1): ").strip() or "1"
    models = {"1": "llama3.2", "2": "mistral", "3": "llama3.1", "4": "__mock__"}
    return models.get(choice, "llama3.2")


# ══════════════════════════════════════════════════════════════════
#  MAIN CHAT LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    print(hr("═"))
    print("  CSC493 RAG Chatbot — Startup".center(W))
    print(hr("═"))
    print()
    print("  Choose initial mode:")
    print("    1. 💻  Offline  (Ollama / Mock — no API key needed)")
    print("    2. 🌐  Online   (Gemini / OpenAI / Anthropic API)")
    print()

    while True:
        choice = input("  Enter choice (1 / 2): ").strip()
        if choice == "1":
            mode = "offline"
            ollama_model = choose_offline_model()
            online_cfg = {}
            engine_label = f"Mock" if ollama_model == "__mock__" else f"Ollama/{ollama_model}"
            break
        elif choice == "2":
            mode = "online"
            online_cfg = setup_online()
            ollama_model = "llama3.2"
            engine_label = online_cfg["provider"].capitalize()
            break
        else:
            print("  ⚠  Please enter 1 or 2.")

    # Build index
    print("\n  📚 Indexing knowledge base…", end="", flush=True)
    tfidf_index, idf = build_index(LECTURE_CHUNKS)
    print(f"  ✓ {len(LECTURE_CHUNKS)} chunks ready.")

    print_banner(mode, engine_label)

    print("  💡 Example questions:")
    print("     • What is RAG?")
    print("     • What guardrails should I use?")
    print("     • Tell me about the Air Canada chatbot case")
    print("     • What do I need to submit for the midterm?")
    print("     • Should a specialist bot also answer general questions?")
    print()

    stats = {"total": 0, "supported": 0, "refused": 0}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Commands ──────────────────────────────────────────────
        if cmd == "quit":
            break

        if cmd == "help":
            print()
            box_print([
                "Ask about: RAG, guardrails, chatbot design, case studies, H-Well, midterm",
                "Commands:",
                "  switch  — toggle offline ↔ online mode",
                "  topics  — list all lecture topics in knowledge base",
                "  stats   — show session statistics",
                "  help    — show this message",
                "  quit    — exit",
            ], title="Help")
            continue

        if cmd == "topics":
            print()
            print(f"┌─ Lecture Topics {'─' * (W - 17)}")
            for c in LECTURE_CHUNKS:
                print(f"│  [{c['id']:2d}] {c['title']}")
            print("└" + hr())
            print()
            continue

        if cmd == "stats":
            print()
            supported_rate = (
                f"{stats['supported'] / stats['total'] * 100:.0f}%"
                if stats["total"] else "N/A"
            )
            refused_rate = (
                f"{stats['refused'] / stats['total'] * 100:.0f}%"
                if stats["total"] else "N/A"
            )
            box_print([
                f"Total queries    : {stats['total']}",
                f"Supported answers: {stats['supported']}  ({supported_rate})",
                f"Refused (off-topic): {stats['refused']}  ({refused_rate})",
                f"Current mode     : {mode.upper()} ({engine_label})",
            ], title="Session Statistics")
            continue

        if cmd == "switch":
            if mode == "offline":
                mode = "online"
                online_cfg = setup_online()
                engine_label = online_cfg["provider"].capitalize()
            else:
                mode = "offline"
                ollama_model = choose_offline_model()
                online_cfg = {}
                engine_label = "Mock" if ollama_model == "__mock__" else f"Ollama/{ollama_model}"
            print_banner(mode, engine_label)
            continue

        # ── Guardrail: topic relevance ────────────────────────────
        stats["total"] += 1

        if not is_on_topic(user_input):
            stats["refused"] += 1
            print()
            print(f"┌─ Guardrail ─ Off-Topic {'─' * (W - 24)}")
            print("│  ⚠  That question is outside my scope.")
            print("│     I'm a specialist assistant for the CSC493 lecture on")
            print("│     RAG chatbots and guardrails.")
            print("│     Try: RAG pipeline, guardrails, case studies, midterm.")
            print("└" + hr())
            print()
            continue

        # ── Retrieve ──────────────────────────────────────────────
        retrieved = retrieve(user_input, tfidf_index, idf, LECTURE_CHUNKS, top_k=3)

        if not retrieved:
            stats["refused"] += 1
            print()
            print(f"┌─ Domain Gap {'─' * (W - 13)}")
            print("│  ℹ  No relevant information found in the lecture knowledge base.")
            print("│     This may be a domain gap. Try rephrasing, or check the slides.")
            print("└" + hr())
            print()
            continue

        stats["supported"] += 1

        # Build context string
        context = "\n\n".join(
            f"[{r['chunk']['title']}]\n{r['chunk']['content']}" for r in retrieved
        )

        # ── Generate ──────────────────────────────────────────────
        print("  ⏳ Generating…", end="\r", flush=True)

        if mode == "offline":
            if ollama_model == "__mock__":
                response, engine_used = _mock_response(user_input, retrieved), "Mock (template)"
            else:
                response, engine_used = generate_offline(user_input, context, retrieved, ollama_model)
        else:
            response, engine_used = generate_online(
                user_input, context, online_cfg["provider"], online_cfg["api_key"]
            )

        print(" " * 20, end="\r")  # clear spinner
        print_answer(response, retrieved, engine_used)

    # ── Session summary ───────────────────────────────────────────
    print()
    if stats["total"] > 0:
        print(hr("─"))
        print(f"  Session: {stats['total']} queries | "
              f"{stats['supported']} supported | "
              f"{stats['refused']} refused")
        print(hr("─"))
    print("  Goodbye! Good luck with your midterm! 👋")
    print()


if __name__ == "__main__":
    main()