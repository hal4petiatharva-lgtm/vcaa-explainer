from flask import Flask, request, render_template, jsonify
import openai
import os
from dotenv import load_dotenv
import socket
from openai.error import AuthenticationError, RateLimitError, APIConnectionError, Timeout, ServiceUnavailableError, InvalidRequestError
import requests

load_dotenv()
app = Flask(__name__)
AI_PROVIDER = (os.getenv("AI_PROVIDER") or "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or OPENAI_API_KEY
if AI_PROVIDER == "openai":
    openai.api_key = OPENAI_API_KEY
elif AI_PROVIDER == "gemini":
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        import google.generativeai as genai  # type: ignore
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        genai = None

system_message = """You are an expert VCE consultant. Provide clear, professional, and actionable feedback on VCE exam questions.

Your response must follow this exact structure:

**What VCAA Expects:**
Explain the specific academic skills and response type this term requires in 2-3 concise sentences.

**Key Requirements:**
List 3-4 specific actions the student must demonstrate, using bullet points.

**Common Pitfalls to Avoid:**
List 3-4 frequent student errors, using bullet points.

**Sample Sentence Starter:**
Provide one example of how a high-quality response could begin.

**Important Formatting Rules:**
- The first line of your response must be the command term in bold, followed by a colon. Example: **Evaluate:**
- Use colons after each bold heading as shown above.
- Do not include a separate 'Command Term Identified' section.

Tone: Be direct, supportive, and professional. Avoid filler phrases or dramatic language."""


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please paste a VCE question to analyze."}), 400

    if AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            return jsonify({"error": "Missing OpenAI API key. Set OPENAI_API_KEY in .env."}), 500
    elif AI_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            return jsonify({"error": "Missing Google Gemini API key. Set GOOGLE_API_KEY in .env."}), 500
    else:
        return jsonify({"error": "Invalid AI provider. Set AI_PROVIDER to 'openai' or 'gemini'."}), 400

    try:
        if AI_PROVIDER == "openai":
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question},
                ],
                temperature=0.2,
                max_tokens=700,
                request_timeout=20,
            )
            content = resp["choices"][0]["message"]["content"].strip()
            return jsonify({"analysis": content})
        else:
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            try:
                import google.generativeai as genai  # type: ignore
                model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
                model = genai.GenerativeModel('gemini-pro')
                resp = model.generate_content([system_message, question])
                content = (getattr(resp, "text", "") or "").strip()
                if not content:
                    content, err = _gemini_rest_generate(system_message, question, model_name, GOOGLE_API_KEY)
                    if not content:
                        return jsonify({"error": f"Gemini generation failed: {err}"}), 502
                return jsonify({"analysis": content})
            except Exception:
                model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
                content, err = _gemini_rest_generate(system_message, question, model_name, GOOGLE_API_KEY)
                if not content:
                    return jsonify({"error": f"Gemini generation failed: {err}"}), 502
                return jsonify({"analysis": content})
    except AuthenticationError:
        return jsonify({"error": "Invalid OpenAI API key. Update OPENAI_API_KEY in .env and restart the server."}), 401
    except RateLimitError:
        return jsonify({"error": "Rate limit reached. Please wait a minute and try again."}), 429
    except Timeout:
        return jsonify({"error": "AI request timed out. Check your internet connection and try again."}), 504
    except APIConnectionError:
        return jsonify({"error": "Network error contacting OpenAI. Ensure you have internet access."}), 503
    except ServiceUnavailableError:
        return jsonify({"error": "OpenAI service is temporarily unavailable. Please try again shortly."}), 503
    except InvalidRequestError as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400
    except Exception:
        return jsonify({"error": "Failed to contact AI service."}), 502


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


def _gemini_list_models(api_key: str, timeout: int = 10):
    url = "https://generativelanguage.googleapis.com/v1/models"
    try:
        r = requests.get(url, headers={"x-goog-api-key": api_key}, timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
        models = data.get("models", [])
        names = [m.get("name") for m in models if m.get("name")]
        return names
    except Exception:
        return []


def _gemini_rest_generate(system_message: str, question: str, model_name: str, api_key: str, timeout: int = 20):
    headers = {"x-goog-api-key": api_key}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system_message}\n\n{question}"}]}
        ]
    }
    candidates = []
    if model_name:
        if model_name.startswith("models/"):
            candidates.append(model_name)
        else:
            candidates.append(f"models/{model_name}")
    discovered = _gemini_list_models(api_key)
    for n in discovered:
        if n not in candidates:
            candidates.append(n)
    last_error = None
    for name in candidates:
        url = f"https://generativelanguage.googleapis.com/v1/{name}:generateContent"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code != 200:
                try:
                    err = r.json().get("error", {}).get("message")
                except Exception:
                    err = r.text
                last_error = err or f"HTTP {r.status_code}"
                continue
            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for p in parts:
                t = p.get("text")
                if t:
                    return t.strip(), None
            last_error = "Empty response parts"
        except Exception as e:
            last_error = str(e)
    return None, last_error or "Unknown Gemini error"


if __name__ == "__main__":
    env_port = os.getenv("PORT")
    port = int(env_port) if env_port else _find_available_port()
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
