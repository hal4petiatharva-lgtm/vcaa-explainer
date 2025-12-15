from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
import socket
import requests
from groq import Groq
from flask_cors import CORS
import importlib.metadata as importlib_metadata
import sys
import logging

load_dotenv()

# Load configuration before using it to initialize clients.
# This ensures environment values are available when constructing SDK clients.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Initialize Groq client only if the key is present.
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Create Flask app after configuration/client setup.
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
if client is None:
    logging.warning("GROQ_API_KEY not set; Groq client not initialized.")

system_message = """You are an expert VCE exam coach. Produce concise, actionable output a student can execute during timed assessments.

Output exactly two sections in Markdown:

📘 Command Term Analysis
- Single line: "Command Term: {TERM} - {concise definition based on VCAA expectations}".
- No action steps or planning language.

🎯 Exam Action Steps
- Provide 4 numbered steps the student can perform in the exam room to answer the question:
  1. Define & Judge (Intro): Define key concepts and state overall judgement upfront.
  2. Argue One Side (Body Paragraph 1): Present strongest evidence for one side.
  3. Argue the Other Side (Body Paragraph 2): Present strongest evidence for the other side.
  4. Justify Your Conclusion: Weigh evidence to defend initial judgement.
- Adapt steps to the actual command term(s) and question.

Rules:
- Do not include research prompts, external sources, case studies, outlines, or pre-written arguments/thesis.
- No factual examples. No meta-scaffolding language.
- Use second-person ("you"). Be concise, exam-focused, and tactically directive.
- If multiple command terms appear, tailor all sections to how they interact and what to do."""


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please paste a VCE question to analyze."}), 400

    if not GROQ_API_KEY:
        return jsonify({"error": "Missing Groq API key. Set GROQ_API_KEY in .env."}), 500

    try:
        chat = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            model=GROQ_MODEL,
            temperature=0.4,
            max_tokens=700,
            top_p=1,
            stream=False,
        )
        content = (chat.choices[0].message.content if chat.choices else "") or ""
        content = content.strip()
        if not content:
            return jsonify({"error": "Groq returned empty content."}), 502
        return jsonify({"analysis": content})
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            return jsonify({"error": "Rate limit reached. Please wait and try again."}), 429
        if "Unauthorized" in msg or "401" in msg:
            return jsonify({"error": "Invalid Groq API key."}), 401
        return jsonify({"error": f"Groq API error: {msg}"}), 502

@app.route("/debug", methods=["GET"])
def debug():
    packages = [f"{d.metadata.get('Name','unknown')}=={d.version}" for d in importlib_metadata.distributions()]
    try:
        importlib_metadata.version("groq")
        groq_installed = True
    except importlib_metadata.PackageNotFoundError:
        groq_installed = False
    try:
        importlib_metadata.version("google-generativeai")
        google_generativeai_installed = True
    except importlib_metadata.PackageNotFoundError:
        google_generativeai_installed = False
    return jsonify({
        "python_version": sys.version,
        "installed_packages": packages,
        "groq_installed": groq_installed,
        "google_generativeai_installed": google_generativeai_installed,
        "env_GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
        "env_AI_PROVIDER": os.environ.get("AI_PROVIDER"),
    })

@app.route("/test-groq", methods=["GET"])
def test_groq():
    try:
        import groq as groq_mod
        client_test = groq_mod.Groq(api_key=os.environ.get("GROQ_API_KEY", "test"))
        return jsonify({"status": "Groq import successful", "client": str(client_test)})
    except ImportError as e:
        return jsonify({"error": f"Groq not installed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Groq setup failed: {str(e)}"}), 500


def _is_port_in_use(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def _find_available_port(candidates=None) -> int:
    candidates = candidates or [5000, 5050, 8000, 8080, 3000, 5001]
    for p in candidates:
        if not _is_port_in_use(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Removed Gemini/OpenAI helper paths. Groq is now the sole provider.


if __name__ == "__main__":
    env_port = os.getenv("PORT")
    port = int(env_port) if env_port else _find_available_port()
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
