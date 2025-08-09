"""
Agent chatbot sederhana dengan LangGraph + LangChain
- Gaya respons: hangat, empatik, berbahasa alami (Indonesia)
- Tools contoh: baca_pesan, kirim_pesan (dummy mailbox)
- Backend LLM: OpenAI-compatible (router HuggingFace by default). Ganti sesuai kebutuhan.
- Fitur: multi-bubble + auto follow-up TANPA trigger, DIKENDALIKAN analisis topik
         + perbaikan tool-calling (policy + soft executor + intent autodial).

Cara pakai cepat:
1) pip install -U langgraph langchain-core langchain-openai pydantic typing_extensions python-dotenv
2) Ekspor token (contoh HF router):
   - Linux/macOS: export HF_TOKEN=sk-xxx
   - Windows (PowerShell): $Env:HF_TOKEN = "sk-xxx"
3) python agent_human_like.py
"""
from __future__ import annotations

import os
import json
import re
from typing import TypedDict, List, Dict, Tuple, Optional

import dotenv
dotenv.load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    AIMessage,
)
from langchain_core.tools import tool

# ===================== KONSTAN =====================
MSG_SPLIT = "<|split|>"           # pemisah gelembung/bubble
FOLLOW_YES = "<|followup|>yes"   # minta lanjut otomatis (opsional, dari model)
FOLLOW_NO = "<|followup|>no"
AUTO_FOLLOW_ROUNDS = 3           # pagar anti-spam: maksimal 3 follow-up otomatis

# Kata/frasal yang dianggap sinyal untuk TIDAK lanjut
STOP_PATTERNS = [
    r"\b(stop|berhenti|cukup|udah|sudah)\b",
    r"\b(nanti dulu|lain kali|nanti aja)\b",
    r"\b(ga|gak|nggak|enggak|tidak)\s+(dulu|usah|mau)\b",
    r"\b(bosan|bosen|capek)\b",
    r"\b(ga seru|gak seru|kurang seru)\b",
    r"\b(skip)\b",
]

# ===================== DUMMY MAILBOX =====================
INBOX = [
    {"from": "budi", "text": "Halo, ada update?"},
    {"from": "cici", "text": "Meeting jam 3 ya."},
    {"from": "andi", "text": "Tolong cek dokumen."},
]
OUTBOX: List[dict] = []

# ===================== TOOLS =====================
@tool
def baca_pesan(n: int = 5) -> str:
    """
    Ambil n pesan terakhir dari INBOX. Return JSON string.
    """
    n = max(0, int(n))
    return json.dumps(INBOX[-n:])

@tool
def kirim_pesan(to: str, content: str) -> str:
    """
    Kirim pesan ke penerima, simpan ke OUTBOX. Return konfirmasi JSON.
    """
    msg = {"to": to, "content": content}
    OUTBOX.append(msg)
    return json.dumps({"status": "sent", "message": msg})

TOOLS = [baca_pesan, kirim_pesan]
TOOLS_MAP = {"baca_pesan": baca_pesan, "kirim_pesan": kirim_pesan}

# ===================== LLM BACKEND =====================
HF_TOKEN = os.getenv("HF_TOKEN")
llm = ChatOpenAI(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct:novita",  # ganti model bila perlu
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    temperature=0.4,  # sedikit kreatif biar terasa alami
)
llm_with_tools = llm.bind_tools(TOOLS)

# ===================== STATE =====================
class State(TypedDict):
    messages: List[BaseMessage]
    # field lain bisa ditambah (profile, memory, dsb)

# ===================== NODES =====================
def agent_node(state: State):
    """Minta model merespons. Kalau butuh tool, dia akan memanggilnya (idealnya terstruktur)."""
    ai_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [ai_msg]}

def soft_tools_node(state: State):
    """
    Safety harness:
    1) Jika AI menulis tool call di TEKS (pseudo), jalankan manual.
    2) Jika intent user = 'baca pesan' tapi AI tidak memanggil tool, autodial baca_pesan.
    Hasil disuntik sebagai SystemMessage agar agent bisa merespons akurat.
    """
    new_msgs: List[BaseMessage] = []
    last_ai_msg: Optional[AIMessage] = next((m for m in reversed(state["messages"]) if m.type == "ai"), None)
    last_user_msg: Optional[HumanMessage] = next((m for m in reversed(state["messages"]) if m.type == "human"), None)

    def parse_pseudo_tool(content: str) -> Optional[Tuple[str, Dict[str, str]]]:
        # Contoh pola:
        # <tool_call>
        # <function=baca_pesan>
        # <parameter=n>
        # 5
        # </parameter>
        # </function>
        # </tool_call>
        if "<tool_call" not in content and "<function=" not in content:
            return None
        fn_match = re.search(r"<function=([a-zA-Z_][a-zA-Z0-9_]*)>", content)
        if not fn_match:
            return None
        fn_name = fn_match.group(1)
        params: Dict[str, str] = {}
        # Ambil blok parameter sederhana <parameter=key>value</parameter>
        for m in re.finditer(r"<parameter=([a-zA-Z_][a-zA-Z0-9_]*)>\s*([\s\S]*?)\s*</parameter>", content):
            params[m.group(1)] = m.group(2).strip()
        return fn_name, params

    ran_tool = False

    # 1) Eksekusi pseudo tool call jika ada
    if last_ai_msg and isinstance(last_ai_msg.content, str):
        parsed = parse_pseudo_tool(last_ai_msg.content)
        if parsed:
            fn_name, params = parsed
            tool_fn = TOOLS_MAP.get(fn_name)
            if tool_fn:
                try:
                    result = tool_fn.invoke(params) if hasattr(tool_fn, "invoke") else tool_fn.run(**{
                        k: (int(v) if v.isdigit() else v) for k, v in params.items()
                    })
                except TypeError:
                    # fallback untuk skema tool decorator
                    result = tool_fn.func(**{
                        k: (int(v) if v.isdigit() else v) for k, v in params.items()
                    })
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                new_msgs.append(SystemMessage(content=f"[HASIL {fn_name}] {result}"))
                ran_tool = True

    # 2) Intent autodial baca_pesan jika user minta tapi AI tidak panggil tool
    def user_asks_inbox(s: str) -> bool:
        s = s.lower()
        keys = ["pesan", "inbox", "ada pesan", "lihat pesan", "baca pesan", "masuk ga", "masuk gak", "masuk nggak"]
        return any(k in s for k in keys)

    if (not ran_tool) and last_user_msg and isinstance(last_user_msg.content, str) and user_asks_inbox(last_user_msg.content):
        try:
            result = baca_pesan.func(n=5)
        except Exception as e:
            result = json.dumps({"error": str(e)})
        new_msgs.append(SystemMessage(content=f"[HASIL baca_pesan] {result}"))
        ran_tool = True

    # Jika tidak ada yang dijalankan, tetap lanjut
    if not ran_tool:
        return {"messages": state["messages"]}

    # Setelah menyuntik hasil tool, panggil agent lagi
    ai_msg_2 = llm_with_tools.invoke(state["messages"] + new_msgs)
    return {"messages": state["messages"] + new_msgs + [ai_msg_2]}

tool_node = ToolNode(TOOLS)

def route(state: State):
    """
    Routing:
    - Jika AI terakhir punya tool_calls terstruktur -> 'tools'
    - Else, jika pseudo tool atau intent perlu tool -> 'soft_tools'
    - Else, 'end'
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"

    # cek pseudo tool atau intent
    last_ai_msg: Optional[AIMessage] = last if getattr(last, "type", "") == "ai" else None
    last_user_msg: Optional[HumanMessage] = last if getattr(last, "type", "") == "human" else None

    def looks_like_pseudo_tool(msg: Optional[AIMessage]) -> bool:
        if not msg or not isinstance(msg.content, str):
            return False
        c = msg.content
        return "<tool_call" in c or "<function=" in c

    def user_asks_inbox(s: Optional[str]) -> bool:
        if not s:
            return False
        s = s.lower()
        keys = ["pesan", "inbox", "ada pesan", "lihat pesan", "baca pesan", "masuk ga", "masuk gak", "masuk nggak"]
        return any(k in s for k in keys)

    if looks_like_pseudo_tool(last_ai_msg):
        return "soft_tools"
    if last_user_msg and user_asks_inbox(getattr(last_user_msg, "content", "")):
        return "soft_tools"

    return "end"

# ===================== GRAPH =====================
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("soft_tools", soft_tools_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", "soft_tools": "soft_tools", "end": END})
graph.add_edge("tools", "agent")
graph.add_edge("soft_tools", "agent")

app = graph.compile()

# ===================== SYSTEM STYLE PROMPT =====================
SYSTEM_HUMANLIKE = SystemMessage(
    content=f"""
Kamu adalah asisten percakapan berbahasa Indonesia yang empatik, hangat, dan jelas.
Prinsip: jawab ringkas dulu lalu tawarkan detail bila diminta, pakai bahasa sehari-hari,
hindari jargon, jangan terlalu formal, gunakan emoji seperlunya, dan tetap sopan.
Selalu gunakan bahasa yang sama dengan pengguna.

KEBIJAKAN TOOL (PENTING):
- Kalau pengguna menanyakan isi inbox/pesan/masuk atau serupa, WAJIB gunakan tool `baca_pesan`.
- Dilarang menebak isi pesan. Jika tool gagal, katakan gagal dan tampilkan errornya.
- Jika perlu mengirim pesan, gunakan tool `kirim_pesan`.

Format multi-bubble: Kamu boleh membalas 1–3 bubble pendek. Pisahkan tiap bubble dengan '{MSG_SPLIT}'.
Follow-up otomatis: jika percakapan layak dilanjutkan tanpa input pengguna, tambahkan tag '{FOLLOW_YES}'.
Jika tidak, pakai '{FOLLOW_NO}'. Gunakan bijak agar tidak terasa spam.
"""
)

# ===================== UTIL =====================
def last_ai(state: State) -> BaseMessage | None:
    for msg in reversed(state["messages"]):
        if msg.type == "ai":
            return msg
    return None

def parse_bubbles_and_follow(text: str) -> Tuple[List[str], bool]:
    t = text.strip()
    follow = False
    if FOLLOW_YES.lower() in t.lower():
        follow = True
        t = re.sub(re.escape(FOLLOW_YES), "", t, flags=re.IGNORECASE)
    if FOLLOW_NO.lower() in t.lower():
        t = re.sub(re.escape(FOLLOW_NO), "", t, flags=re.IGNORECASE)
    parts = [p.strip() for p in t.split(MSG_SPLIT)]
    parts = [p for p in parts if p]
    return parts, follow

def ends_with_question(s: str) -> bool:
    return s.strip().endswith("?")

def contains_stop_signals(s: str) -> bool:
    low = s.lower()
    return any(re.search(p, low) for p in STOP_PATTERNS)

def get_recent_context(state: State, k: int = 8) -> str:
    recent = state["messages"][-k:]
    lines = []
    for m in recent:
        role = "user" if m.type == "human" else ("assistant" if m.type == "ai" else m.type)
        lines.append(f"{role}: {getattr(m, 'content', '')}")
    return "\n".join(lines)

def extract_json_obj(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def llm_followup_controller(state: State) -> Tuple[bool, float, str]:
    """Putuskan apakah layak follow-up berdasarkan konteks terbaru. Return: (follow, score, reason)."""
    context = get_recent_context(state)
    sys = SystemMessage(
        content=(
            "Anda adalah kontroler keputusan follow-up percakapan.\n"
            "Tujuan: putuskan apakah asisten perlu mengirim 1 bubble follow-up TANPA menunggu user,\n"
            "berdasarkan konteks (ketertarikan, kejelasan topik, sopan santun).\n"
            "Kriteria FOLLOW: topik belum tuntas, tidak ada penolakan, ada nilai tambah singkat (fakta kecil, saran singkat),\n"
            "dan tidak menutup dengan pertanyaan dari asisten sebelumnya.\n"
            "Kriteria STOP: user menolak/menunda, topik selesai, atau asisten terakhir sudah bertanya.\n"
            "Format keluaran ketat JSON satu baris: {\"follow\": true|false, \"score\": 0..1, \"reason\": \"...\"}."
        )
    )
    hm = HumanMessage(content=f"Konteks terkini:\n{context}\n\nKeputusan?")
    resp = llm.invoke([sys, hm])
    data = extract_json_obj(resp.content)
    follow = bool(data.get("follow", False))
    score_raw = data.get("score", 0)
    try:
        score = float(score_raw)
    except Exception:
        score = 0.0
    reason = str(data.get("reason", ""))
    return follow, score, reason

def should_auto_follow(state: State) -> bool:
    """Keputusan gabungan: stop-signal, tanda tanya, tag model, controller JSON."""
    # 1) Jika user barusan berkata stop/menolak => jangan.
    last_user = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
    if last_user and contains_stop_signals(getattr(last_user, "content", "")):
        return False

    # 2) Jika balasan AI terakhir berakhir tanda tanya => beri ruang user.
    last = last_ai(state)
    if not last:
        return False
    bubbles, tag_follow = parse_bubbles_and_follow(last.content)
    if bubbles and ends_with_question(bubbles[-1]):
        return False

    # 3) Hormati tag eksplisit
    if tag_follow is True:
        return True
    if tag_follow is False:
        return False

    # 4) Controller
    follow, score, _ = llm_followup_controller(state)
    return bool(follow and score >= 0.55)

# ===================== CLI LOOP (DEMO) =====================
def chat_loop():
    print("\nChat siap. Bot bisa auto follow-up sampai 3 kali kalau topik dianggap layak. Ketik 'stop'/'exit' untuk berhenti.\n")
    state: State = {"messages": [SYSTEM_HUMANLIKE]}

    while True:
        user_inp = input("you: ").strip()
        if user_inp.lower() in {"exit", "quit", "keluar", "stop", "berhenti"}:
            print("bot: Sip. Sampai nanti ✌️")
            break

        state["messages"].append(HumanMessage(content=user_inp))

        # ====== RESPON UTAMA ======
        state = app.invoke(state)
        ai = last_ai(state)
        if not ai:
            print("bot: (tidak ada respons)")
            continue
        bubbles, _ = parse_bubbles_and_follow(ai.content)
        for b in bubbles:
            print(f"bot: {b}")

        # ====== AUTO FOLLOW-UP TERKENDALI (maks 3 putaran) ======
        for _ in range(AUTO_FOLLOW_ROUNDS):
            if not should_auto_follow(state):
                break
            # Instruksi follow-up: 1 bubble, tambah info baru, hindari pengulangan
            state["messages"].append(HumanMessage(content=(
                "Buat 1 bubble follow-up yang alami dari topik terakhir. "
                "Jangan mengulang kalimat sebelumnya. Tambahkan info baru atau ajakan singkat. "
                "Hindari bertanya jika barusan kamu sudah bertanya. "
                f"Gunakan format multi-bubble dengan maksimal 1 bubble. {FOLLOW_NO}"
            )))
            state = app.invoke(state)
            ai = last_ai(state)
            if not ai:
                break
            bubbles, _ = parse_bubbles_and_follow(ai.content)
            if not bubbles:
                break
            print(f"bot: {bubbles[0]}")

if __name__ == "__main__":
    chat_loop()