"""
app.py
------
Flask REST API for the custom BPE tokenizer web app.
Serves the frontend and exposes tokenization + history endpoints.
"""

import json
import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from tokenizer_core import get_tokenizer
from models import init_db, SessionLocal, TokenizerSession

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# Initialise DB on startup
init_db()

# Load / train tokenizer once on startup
tokenizer = get_tokenizer(vocab_size=512)

AVAILABLE_MODELS = [
    {"id": "custom-bpe", "label": "Custom BPE (v1)"},
]

# Special token strings  (mirroring the ChatML format used by tiktokenizer)
SPECIAL_TOKENS = {
    "im_start": "<|im_start|>",
    "im_sep":   "<|im_sep|>",
    "im_end":   "<|im_end|>",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(messages: list[dict]) -> str:
    """
    Convert a list of {role, content} messages into a ChatML-style prompt
    that matches the format shown on tiktokenizer.vercel.app.
    """
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(
            f"{SPECIAL_TOKENS['im_start']}{role}"
            f"{SPECIAL_TOKENS['im_sep']}{content}"
            f"{SPECIAL_TOKENS['im_end']}"
        )
    # Trailing assistant prompt stub
    parts.append(
        f"{SPECIAL_TOKENS['im_start']}assistant{SPECIAL_TOKENS['im_sep']}"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Routes — Frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify({"models": AVAILABLE_MODELS})


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    """
    Accepts either:
      { "text": "<raw string to tokenize>" }          ← new simplified mode
      { "messages": [{role, content}, ...] }          ← legacy chat mode

    Response:
        {
            "token_count": int,
            "token_ids":   [int, ...],
            "segments":    [{"id": int, "text": str}, ...],
            "raw_text":    str
        }
    """
    data = request.get_json(force=True, silent=True) or {}

    # New simplified mode: direct raw text
    if "text" in data:
        raw_text = data["text"]
    else:
        # Legacy: build ChatML prompt from messages array
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            return jsonify({"error": "messages must be a list"}), 400
        raw_text = build_prompt(messages)

    ids, segments = tokenizer.encode_with_segments(raw_text)

    return jsonify({
        "token_count": len(ids),
        "token_ids":   ids,
        "segments":    segments,
        "raw_text":    raw_text,
    })


@app.route("/api/history", methods=["GET"])
def api_history_get():
    """Return the last 50 tokenisation sessions."""
    db = SessionLocal()
    try:
        sessions = (
            db.query(TokenizerSession)
            .order_by(TokenizerSession.created_at.desc())
            .limit(50)
            .all()
        )
        return jsonify({"history": [s.to_dict() for s in sessions]})
    finally:
        db.close()


@app.route("/api/history", methods=["POST"])
def api_history_post():
    """
    Save a tokenisation session.
    Body: { messages, token_count, token_ids, raw_text, model }
    """
    data = request.get_json(force=True, silent=True) or {}
    required = ("messages", "token_count", "token_ids")
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    db = SessionLocal()
    try:
        session = TokenizerSession(
            model       = data.get("model", "custom-bpe"),
            messages    = json.dumps(data["messages"]),
            token_count = int(data["token_count"]),
            token_ids   = json.dumps(data["token_ids"]),
            raw_text    = data.get("raw_text", ""),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return jsonify({"success": True, "id": session.id}), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/history/<int:session_id>", methods=["DELETE"])
def api_history_delete(session_id: int):
    db = SessionLocal()
    try:
        session = db.get(TokenizerSession, session_id)
        if not session:
            return jsonify({"error": "Not found"}), 404
        db.delete(session)
        db.commit()
        return jsonify({"success": True})
    finally:
        db.close()


@app.route("/api/vocab", methods=["GET"])
def api_vocab():
    """Return a sample of the trained vocabulary for debugging."""
    vocab_sample = {}
    for idx, token_bytes in list(tokenizer.vocab.items())[:300]:
        try:
            vocab_sample[idx] = token_bytes.decode("utf-8", errors="replace")
        except Exception:
            vocab_sample[idx] = repr(token_bytes)
    return jsonify({
        "vocab_size": tokenizer.vocab_size,
        "num_merges": len(tokenizer.merges),
        "sample":     vocab_sample,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Tokenizer Web App")
    print(f"  Vocab size : {tokenizer.vocab_size}")
    print(f"  Merges     : {len(tokenizer.merges)}")
    print("  Open       : http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
