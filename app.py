"""Conferro AI backend. Run: uvicorn app:app --reload"""
import asyncio, hashlib, os, re, time
from collections import defaultdict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()
app = FastAPI(title="Conferro AI")

# ---------------------------------------------------------------- Lyzr agents
LYZR_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
AGENTS = {k: os.getenv(f"LYZR_{k.upper()}_AGENT_ID") for k in ("moderation", "teaching", "summary")}


async def lyzr(kind: str, message: str, session: str):
    """Call a Lyzr agent. Returns None if not configured or on error (callers fall back)."""
    key, agent_id = os.getenv("LYZR_API_KEY"), AGENTS.get(kind)
    if not (key and agent_id):
        return None
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(
                LYZR_URL,
                headers={"x-api-key": key, "Content-Type": "application/json"},
                json={"user_id": "conferro", "agent_id": agent_id,
                      "session_id": session, "message": message},
            )
            r.raise_for_status()
            return r.json().get("response")
    except Exception as e:  # network/auth/format problems must never break the meeting
        print(f"[lyzr:{kind}] {e}")
        return None


# --------------------------------------------------------------- Qdrant memory
DIM, COL = 256, "conferro_memory"


def embed(text: str):
    """Tiny hashed bag-of-words embedding so the demo needs no model download.
    Swap for a real embedding model (e.g. fastembed) for better recall."""
    v = [0.0] * DIM
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
    n = sum(x * x for x in v) ** 0.5 or 1
    return [x / n for x in v]


qd = (QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
      if os.getenv("QDRANT_URL") else QdrantClient(":memory:"))
if not qd.collection_exists(COL):
    qd.create_collection(COL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))


def remember(room: str, kind: str, text: str):
    try:
        qd.upsert(COL, [PointStruct(id=time.time_ns() % (2**63), vector=embed(text),
                                    payload={"room": room, "kind": kind, "text": text})])
    except Exception as e:
        print(f"[qdrant] {e}")


def recall(text: str, k: int = 3):
    try:
        hits = qd.query_points(COL, query=embed(text), limit=k).points
        return [f"[{h.payload['kind']}] {h.payload['text']}" for h in hits if h.score > 0.2]
    except Exception as e:
        print(f"[qdrant] {e}")
        return []


# ----------------------------------------------------------------- Moderation
BAD = {"idiot", "stupid", "shut up", "fuck", "shit", "bitch", "bastard", "asshole", "dumb"}


async def is_abusive(text: str, room: str) -> bool:
    reply = await lyzr(
        "moderation",
        "Reply with exactly ABUSIVE or OK. Is this meeting chat message abusive, "
        f"harassing or hateful?\n\n{text}", f"mod-{room}")
    if reply:
        return reply.strip().upper().startswith("ABUSIVE")
    t = text.lower()
    return any(b in t for b in BAD)


# ---------------------------------------------------------------------- Rooms
rooms: dict[str, dict[str, dict]] = defaultdict(dict)
banned: dict[str, set] = defaultdict(set)
transcripts: dict[str, list] = defaultdict(list)
questions: dict[str, list] = defaultdict(list)


async def send(p, msg):
    try:
        await p["ws"].send_json(msg)
    except Exception:
        pass


def host_of(room):
    return next((p for p in rooms[room].values() if p["role"] == "host"), None)


async def to_host(room, msg):
    h = host_of(room)
    if h:
        await send(h, msg)


async def broadcast(room, msg, skip=None):
    for n, p in list(rooms[room].items()):
        if n != skip:
            await send(p, msg)


async def push_roster(room):
    roster = [{k: p[k] for k in ("name", "strikes", "muted", "cam_mode", "engagement")}
              for p in rooms[room].values() if p["role"] == "listener"]
    await to_host(room, {"t": "roster", "roster": roster})


async def apply_strike(room, name, text):
    p = rooms[room][name]
    p["strikes"] += 1
    s = p["strikes"]
    if s == 1:
        note = "Warning 1 of 3: please stop using abusive language."
    elif s == 2:
        p["muted"] = True
        note = "Warning 2 of 3: your camera and microphone are now muted."
        await broadcast(room, {"t": "frame", "name": name, "img": None})
    else:
        note = "Warning 3 of 3: you have been removed from the meeting."
        banned[room].add(name)
    await send(p, {"t": "warning", "level": min(s, 3), "text": note, "muted": p["muted"]})
    await to_host(room, {"t": "alert", "text": f"{name} received warning {min(s, 3)}: \"{text[:50]}\""})
    if s >= 3:
        await p["ws"].close()
    await push_roster(room)


async def suggest(room):
    """Teaching assistant: turns the room's engagement mix into advice for the host."""
    ls = [p for p in rooms[room].values() if p["role"] == "listener"]
    if not ls:
        return
    n = len(ls)
    confused = sum(p["engagement"] == "confused" for p in ls)
    bored = sum(p["engagement"] == "bored" for p in ls)
    if (confused + bored) / n < 0.3:
        return
    context = " ".join(transcripts[room][-5:])[-600:] or "the current topic"
    past = recall(context)
    mood = "confused" if confused >= bored else "bored"
    prompt = (f"{confused} of {n} listeners look confused and {bored} look bored.\n"
              f"Recent lecture: {context}\nSimilar past moments: {past}\n"
              "Give the host ONE short, practical suggestion (real example, analogy or light joke).")
    tip = await lyzr("teaching", prompt, f"teach-{room}")
    if not tip:
        tip = ("Many listeners seem lost. Pause and give a real-life example, then ask a quick question."
               if mood == "confused" else
               "Energy is dropping. Try a short joke, a story, or ask the group a question.")
    await to_host(room, {"t": "suggest", "text": tip, "memory": past,
                         "stats": f"{confused} confused, {bored} bored of {n}"})


async def suggest_loop():
    while True:
        await asyncio.sleep(15)
        for room in list(rooms):
            await suggest(room)


@app.on_event("startup")
async def _start():
    asyncio.create_task(suggest_loop())


# ------------------------------------------------------------------ WebSocket
@app.websocket("/ws/{room}/{name}")
async def ws_room(ws: WebSocket, room: str, name: str, role: str = "listener"):
    await ws.accept()
    role = "host" if role == "host" else "listener"
    if name in banned[room] or name in rooms[room] or (role == "host" and host_of(room)):
        await ws.send_json({"t": "warning", "level": 3, "text": "You cannot join this room with that name or role."})
        return await ws.close()
    rooms[room][name] = dict(ws=ws, name=name, role=role, strikes=0, muted=False,
                             cam_mode="self_only", engagement="attentive")
    await push_roster(room)
    await to_host(room, {"t": "alert", "text": f"{name} joined"})
    try:
        while True:
            m = await ws.receive_json()
            p = rooms[room].get(name)
            if not p:
                break
            t = m.get("t")
            if t == "chat":
                text = str(m.get("text", ""))[:500]
                if role == "listener" and await is_abusive(text, room):
                    await apply_strike(room, name, text)
                    continue
                if text.strip().endswith("?"):
                    questions[room].append(text)
                    remember(room, "question", text)
                await broadcast(room, {"t": "chat", "name": name, "text": text})
            elif t == "transcript" and role == "host":
                text = str(m.get("text", ""))[:1000]
                transcripts[room].append(text)
                remember(room, "lecture", text)
            elif t == "engagement" and role == "listener" and not p["muted"]:
                p["engagement"] = m.get("state", "attentive")
                await push_roster(room)
            elif t == "cam_mode" and m.get("mode") in ("self_only", "host_only", "everyone"):
                p["cam_mode"] = m["mode"]
                await send_visibility_reset(room, p)
                await push_roster(room)
            elif t == "frame" and not p["muted"] and p["cam_mode"] != "self_only":
                msg = {"t": "frame", "name": name, "img": m.get("img")}
                if p["cam_mode"] == "everyone":
                    await broadcast(room, msg, skip=name)
                else:
                    await to_host(room, msg)
    except WebSocketDisconnect:
        pass
    finally:
        if rooms[room].get(name, {}).get("ws") is ws:
            rooms[room].pop(name, None)
            await broadcast(room, {"t": "frame", "name": name, "img": None})
            await push_roster(room)
            await to_host(room, {"t": "alert", "text": f"{name} left"})


async def send_visibility_reset(room, p):
    """When a listener narrows who can see them, clear their last frame for the others."""
    blank = {"t": "frame", "name": p["name"], "img": None}
    if p["cam_mode"] == "self_only":
        await broadcast(room, blank, skip=p["name"])
    elif p["cam_mode"] == "host_only":
        for n, o in rooms[room].items():
            if o["role"] != "host" and n != p["name"]:
                await send(o, blank)


# ------------------------------------------------------------------- REST API
@app.get("/api/summary/{room}")
async def summary(room: str):
    lecture, qs = " ".join(transcripts[room]), questions[room]
    reply = await lyzr(
        "summary",
        f"Summarize this class in 5 bullet points, then list open questions.\n\n"
        f"Lecture: {lecture[-3000:]}\n\nQuestions: {qs}", f"sum-{room}")
    if not reply:
        sents = re.split(r"(?<=[.!?])\s+", lecture)[:5]
        reply = "Key points:\n- " + "\n- ".join(sents or ["No transcript captured yet."])
        reply += "\n\nQuestions asked:\n- " + "\n- ".join(qs or ["None"])
    return {"summary": reply}


@app.post("/api/omi/{room}")
async def omi_webhook(room: str, request: Request):
    """Point an Omi memory/transcript webhook here; text goes into Qdrant and the transcript.
    Check Omi's current webhook docs for the exact payload; both common shapes are handled."""
    data = await request.json()
    segs = data.get("transcript_segments") or data.get("segments") or []
    text = " ".join(s.get("text", "") for s in segs) or data.get("text", "")
    if text.strip():
        transcripts[room].append(text)
        remember(room, "omi", text)
    return {"ok": True, "stored": bool(text.strip())}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
