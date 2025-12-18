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
import re

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

try:
    from vce_knowledge_base import VCEDatabase
    vce_db = VCEDatabase()
    VCAA_AVAILABLE = bool(getattr(vce_db, "chunks", []))
except Exception:
    class DummyDB:
        def search(self, query, k=3):
            return []
    vce_db = DummyDB()
    VCAA_AVAILABLE = False

def detect_subject(query):
    """Detect probable VCE subject from query keywords."""
    query = query.lower()
    # Map keywords to VCAA subject names found in the database
    subjects = {
        'Psychology': ['psychology', 'neurotransmitter', 'nervous system', 'mental health', 'gaba', 'dopamine', 'stress', 'learning', 'memory', 'phobia', 'biopsychosocial', 'conditioning', 'plasticity'],
        'Legal Studies': ['legal studies', 'constitution', 'parliament', 'court', 'legislation', 'precedent', 'justice', 'rights', 'civil', 'criminal', 'sanction', 'remedy', 'high court'],
        'Business Management': ['business', 'management', 'stakeholder', 'operations', 'human resource', 'corporate', 'strategies', 'kpi', 'maslow', 'locke', 'motivation', 'change management'],
        'Health and Human Development': ['hhd', 'health', 'human development', 'sdg', 'who', 'morbidity', 'burden of disease', 'nutrition', 'life expectancy', 'daly', 'yzk'],
        'Economics': ['economics', 'demand', 'supply', 'market', 'inflation', 'budget', 'monetary', 'aggregate', 'fiscal', 'living standards', 'growth'],
        'Geography': ['geography', 'land use', 'population', 'tourism', 'environment', 'hazards', 'spatial', 'mega city', 'desertification', 'glacier', 'volcano'],
        'Global Politics': ['global politics', 'globalization', 'sovereignty', 'power', 'national interest', 'foreign policy', 'ngos', 'un', 'imf', 'icc', 'terrorism', 'asymmetric'],
        'History': ['history', 'revolution', 'colonies', 'ancient', 'modern', 'war', 'crisis', 'french', 'russian', 'chinese', 'american'],
        'Biology': ['biology', 'cell', 'enzyme', 'dna', 'protein', 'photosynthesis', 'respiration', 'pathogen', 'immunity', 'evolution', 'crispr', 'mitochondria'],
        'Chemistry': ['chemistry', 'molecule', 'reaction', 'equilibrium', 'stoichiometry', 'organic', 'analysis', 'bonding', 'titration', 'redox', 'nmr', 'hplc'],
        'Physics': ['physics', 'motion', 'energy', 'field', 'relativity', 'electricity', 'magnetism', 'wave', 'newton', 'einstein', 'projectile'],
        'English': ['english', 'language', 'text', 'author', 'theme', 'character', 'structure', 'argument', 'persuasive', 'contention'],
        'Literature': ['literature', 'literary', 'criticism', 'passage', 'perspective', 'view', 'interpretation'],
        'Media': ['media', 'audience', 'representation', 'codes', 'conventions', 'narrative', 'genre', 'ideology'],
    }
    
    detected = []
    for subj, keywords in subjects.items():
        if any(k in query for k in keywords):
            detected.append(subj)
            
    # Handle overlap for "globalization"
    if 'globalization' in query:
        if 'Geography' not in detected: detected.append('Geography')
        if 'Global Politics' not in detected: detected.append('Global Politics')
        if 'Economics' not in detected: detected.append('Economics')
    
    return list(set(detected)) if detected else None

# Debug route to check database status in detail
@app.route('/debug/db')
def debug_database():
    import os
    debug_info = {
        'database_loaded': VCAA_AVAILABLE,
        'chunks_in_memory': len(vce_db.chunks) if VCAA_AVAILABLE else 0,
        'embedding_file_exists': os.path.exists('vcaa_simple_embeddings.pkl'),
        'embedding_file_size': os.path.getsize('vcaa_simple_embeddings.pkl') if os.path.exists('vcaa_simple_embeddings.pkl') else 0,
        'current_working_directory': os.getcwd(),
        'files_in_directory': str(os.listdir('.'))
    }
    return jsonify(debug_info)

def clean_source_excerpt(chunk, query):
    """
    Cleans and extracts the most relevant excerpt from a source chunk.
    Removes noise (copyright, page numbers) and extracts 50-150 words around keywords.
    """
    if not chunk:
        return ""
    
    # 1. Pre-process: Remove noise lines
    lines = chunk.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip noise lines
        if re.search(r'© VCAA \d{4}', line): continue
        if re.search(r'Page \d+', line): continue
        if re.search(r'SECTION [A-Z]', line): continue
        if re.match(r'Question \d+', line, re.IGNORECASE): continue
        # Skip MCQ options if they start with A. B. C. D.
        if re.match(r'^[A-D]\.', line.strip()): continue 
        cleaned_lines.append(line)
        
    text = " ".join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return "Source content not available."

    # 2. Extract Relevant Sentences
    # Simple sentence splitter (heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        sentences = [text]
        
    query_terms = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 3] # Filter small words
    
    best_score = -1
    best_window = (0, 1) # start, end (exclusive)
    
    # Sliding window of 1-3 sentences to find best context
    for window_size in range(1, 4):
        if window_size > len(sentences): break
        for i in range(len(sentences) - window_size + 1):
            window_text = " ".join(sentences[i : i + window_size])
            # Score: count keyword matches
            score = sum(1 for term in query_terms if term in window_text.lower())
            
            # Prefer shorter windows if scores are equal to be concise
            if score > best_score:
                best_score = score
                best_window = (i, i + window_size)
    
    # If no matches found, just take the first few sentences
    if best_score <= 0:
        start, end = 0, min(3, len(sentences))
    else:
        start, end = best_window
        
    excerpt = " ".join(sentences[start:end])
    
    # 3. Enforce Length & Fluency
    words = excerpt.split()
    if len(words) > 120:
        excerpt = " ".join(words[:120]) + "..."
    
    # Add context indicators if we skipped text
    if start > 0:
        excerpt = "[...] " + excerpt
    if end < len(sentences):
        excerpt = excerpt + " [...]"
        
    return excerpt

system_message = """You are an expert VCE exam coach. Produce concise, actionable output a student can execute during timed assessments.

Output exactly two sections in Markdown:

📘 Command Term Analysis
First, detect ALL command terms in the question by scanning for standard VCAA verbs across the entire input. Common terms include: Analyse, Evaluate, Compare, Contrast, Discuss, Explain, Justify, Describe, Identify, Classify, Outline, Assess, Argue, To what extent.

Format:
- Start with: **Command Term:** {TERM}
- Then, provide 3-5 concise bullet points analyzing the term(s). Each bullet must focus on a key aspect:
  * Definition & Meaning
  * VCAA Assessment Criteria (what examiners look for)
  * Structural Requirements
  * Common Pitfalls or Key Differentiators
- Do NOT restate the section header "Command Term Analysis".
- Do NOT use a separate paragraph block. Use only bullet points.

For "Evaluate", ensure one bullet addresses the need for a clear judgement and weighing of evidence.
Do not include numbered steps, templates, or planning language in this section.

IMPORTANT: If "OFFICIAL VCAA DATABASE CONTEXT" is provided in the prompt, you MUST incorporate it into your analysis. Use it to clarify the specific VCAA interpretation of the command term or concept.

🎯 Exam Action Steps
- Provide 4 numbered steps the student can perform in the exam room to answer the question:
  1. Define & Judge (Intro): Define key concepts and state overall judgement upfront.
  2. Argue One Side (Body Paragraph 1): Present strongest evidence for one side.
  3. Argue the Other Side (Body Paragraph 2): Present strongest evidence for the other side.
  4. Justify Your Conclusion: Weigh evidence to defend initial judgement.
- Adapt steps to the actual command term(s) and question.
- If MULTIPLE terms are detected, integrate them into a single coherent sequence rather than concatenating separate term-specific steps. For example, for "Evaluate and Compare": frame evaluation criteria, evaluate context A, evaluate context B, then directly compare and conclude with a justified judgement.

Rules:
- Do not include research prompts, external sources, case studies, outlines, or pre-written arguments/thesis.
- No factual examples. No meta-scaffolding language.
- Use second-person ("you"). Be concise, exam-focused, and tactically directive.
- If multiple command terms appear, tailor all sections to how they interact and what to do.
- Keep sections strictly separate. The detailed paragraph belongs only in 📘 Command Term Analysis; the numbered steps belong only in 🎯 Exam Action Steps."""


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

    vcaa_results = []
    vcaa_context = ""
    detected_subjects = detect_subject(question)
    
    if VCAA_AVAILABLE:
        try:
            # Force a slightly lower k to ensure we get best matches, but handle empty results gracefully
            vcaa_results = vce_db.search(question, k=5, subject_filter=detected_subjects) or []
            lines = []
            for i, tup in enumerate(vcaa_results):
                try:
                    chunk, meta, score = tup
                except ValueError:
                    chunk = tup[0] if len(tup) > 0 else ""
                    meta = tup[1] if len(tup) > 1 else {}
                    score = tup[2] if len(tup) > 2 else 0.0
                subject = str(meta.get("subject", "Unknown"))
                year = str(meta.get("year", "Unknown"))
                typ = str(meta.get("type", "Unknown"))
                # Clean up chunk text
                snippet = (chunk or "")[:350].replace("\n", " ").strip()
                lines.append(f"SOURCE {i+1} [{subject} {year} {typ}]: {snippet}")
            if lines:
                vcaa_context = "\n".join(lines)
        except Exception as e:
            logging.warning(f"[VCEInsider] VCAA search error: {e}")
            vcaa_results = []
            vcaa_context = ""

    try:
        # Construct a very explicit prompt for the model
        subj_str = ", ".join(detected_subjects) if detected_subjects else "General VCE"
        
        user_content = (
            f"You are a VCE expert. You must strictly adhere to the current VCE Study Design (2023-2027) for {subj_str}.\n"
            f"Validate that all terms/concepts in your response are part of the current study design. If the user asks about an outdated term (e.g., 'spontaneous recovery' in Psychology), explicitly note this and refocus on current knowledge.\n\n"
            f"User Question: {question}\n\n"
        )
        
        if vcaa_context:
            user_content += (
                f"OFFICIAL VCAA DATABASE CONTEXT (MUST USE):\n"
                f"{vcaa_context}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Actively extract specific data from the context. Use phrases like:\n"
                f"   - \"The 2022 Exam Report highlights that approximately 40% of students...\"\n"
                f"   - \"A common pitfall noted by assessors was...\"\n"
                f"   - \"High-scoring responses consistently demonstrated...\"\n"
                f"2. Explain your subject focus: \"Analyzing this from a {subj_str} perspective...\"\n\n"
            )
        else:
            user_content += "No specific VCAA report excerpts available. Provide general expert VCE advice based on the current study design.\n\n"
            
        user_content += (
            f"Provide a response in two parts:\n"
            f"1. **Command Term Analysis**: Explain the term. Weave in the VCAA data/insights requested above.\n"
            f"2. **Exam Action Steps**: Give practical steps. Warn about common pitfalls found in the context.\n\n"
            f"Format: Markdown. Do NOT use inline brackets like [SOURCE X]. All source information will be displayed separately."
        )

        chat = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
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
        citations = []
        if vcaa_results:
            for (chunk, meta, score) in vcaa_results:
                cleaned_text = clean_source_excerpt(chunk, question)
                # Create a short preview (first ~25 words) from the cleaned text
                snippet_words = cleaned_text.split()[:25]
                snippet = " ".join(snippet_words) + ("..." if len(cleaned_text.split()) > 25 else "")
                
                citations.append({
                    "subject": str(meta.get("subject", "Unknown")),
                    "year": str(meta.get("year", "Unknown")),
                    "type": str(meta.get("type", "Unknown")),
                    "relevance": float(score),
                    "snippet": snippet,
                    "full_text": cleaned_text
                })
        return jsonify({"analysis": content, "citations": citations, "database_used": bool(vcaa_results), "vcaa_available": VCAA_AVAILABLE})
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            return jsonify({"error": "Rate limit reached. Please wait and try again."}), 429
        if "Unauthorized" in msg or "401" in msg:
            return jsonify({"error": "Invalid Groq API key."}), 401
        return jsonify({"error": f"Groq API error: {msg}"}), 502

def call_your_ai_function(prompt: str) -> str:
    if not GROQ_API_KEY or client is None:
        return "Analysis temporarily unavailable."
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.4,
            max_tokens=700,
            top_p=1,
            stream=False,
        )
        return (chat.choices[0].message.content or "").strip()
    except Exception:
        return "Analysis temporarily unavailable."

def analyze_with_vcaa(question: str):
    vcaa_results = vce_db.search(question, k=3) if VCAA_AVAILABLE else []
    parts = []
    parts.append(f"VCE QUESTION: {question}")
    if vcaa_results:
        parts.append("RELEVANT VCAA REFERENCES:")
        for i, (chunk, meta, score) in enumerate(vcaa_results):
            subj = str(meta.get("subject", "Unknown"))
            yr = str(meta.get("year", "Unknown"))
            snippet = (chunk or "")[:200].replace("\n", " ").strip()
            parts.append(f"{i+1}. {subj} {yr}: {snippet}")
    parts.append("ANALYSIS REQUIREMENTS:\n1. Identify command term(s)\n2. Explain VCAA expectations\n3. Provide step-by-step response plan\n4. Note common pitfalls\n5. Give example opening\n\nKeep response concise and exam-focused.")
    prompt = "\n".join(parts)
    analysis = call_your_ai_function(prompt)
    resp = {"analysis": analysis, "vcaa_available": VCAA_AVAILABLE, "citations": []}
    if vcaa_results:
        resp["citations"] = [
            {
                "subject": str(meta.get("subject", "Unknown")),
                "year": str(meta.get("year", "Unknown")),
                "type": str(meta.get("type", "Unknown")),
                "relevance": float(score),
            }
            for (_, meta, score) in vcaa_results
        ]
    return resp

@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        result = analyze_with_vcaa(question)
        return jsonify(result)
    except Exception:
        return jsonify({"error": "Analysis failed", "analysis": "Please try again later.", "vcaa_available": False}), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"vcaa_database": VCAA_AVAILABLE, "status": ("operational" if VCAA_AVAILABLE else "degraded")})
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
