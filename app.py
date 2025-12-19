from flask import Flask, request, render_template, jsonify, session, redirect, url_for
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
import random
import time
import json

load_dotenv()

# Load configuration before using it to initialize clients.
# This ensures environment values are available when constructing SDK clients.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Initialize Groq client only if the key is present.
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Create Flask app after configuration/client setup.
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key_123")
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

def clean_vcaa_chunk(raw_chunk, user_query):
    """
    Robust cleaning function for VCAA source chunks.
    Stages:
    1. Remove universal noise (copyright, headers, marks).
    2. Detect and remove gibberish/corruption (reversed text, fragments).
    3. Intelligent context extraction (keyword-based, sentence extraction).
    4. Final fluency and length check (15-50 words).
    """
    if not raw_chunk:
        return "Source content not available."

    # --- Stage 1: Remove Universal Noise Patterns ---
    # Split into lines to handle line-based noise first
    lines = raw_chunk.split('\n')
    cleaned_lines = []
    
    noise_patterns = [
        r'©\s*VCAA\s*\d{4}',            # Copyright
        r'Version\s*\d+\s*[–-]\s*\w+\s*\d{4}', # Version info
        r'Page\s*\d+',                   # Page numbers
        r'SECTION\s*[A-Z]',              # Section headers
        r'Question\s*\d+',               # Question labels
        r'Source\s*\d+:',                # Source labels
        r'TURN\s*OVER',                  # Turn over instruction
        r'^\s*\d+\s*marks?\s*$',         # Isolated mark allocations
        r'\(\d+\s*marks?\)',             # Parenthetical marks
        r'^\s*[A-D]\.\s',                # MCQ options start
        r'^\s*table\s*$',                # Common table artifact
        r'^\s*figure\s*\d+',             # Figure labels
    ]
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        is_noise = False
        for pat in noise_patterns:
            if re.search(pat, line, re.IGNORECASE):
                is_noise = True
                break
        if is_noise: continue
        
        cleaned_lines.append(line)
        
    text = " ".join(cleaned_lines)
    
    # --- Stage 2: Detect and Remove Gibberish/Corruption ---
    
    # Heuristic for reversed text or spaced text: "t h i s  i s  t e x t"
    # Remove sequences of single chars separated by spaces (min length 10 chars)
    # e.g. "a e r a s i h t"
    text = re.sub(r'(?:\b\w\s+){5,}\w', ' ', text)
    
    # Specific known reversed phrases or common corrupted headers
    text = re.sub(r'e t i r w t o n o d', '', text, flags=re.IGNORECASE) # "do not write" reversed
    text = re.sub(r'a e r a', '', text, flags=re.IGNORECASE) # "area" spaced
    
    # Remove any remaining long sequences of non-word patterns
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is too short or just punctuation/numbers after cleaning
    if len(text) < 10 or not re.search(r'[a-zA-Z]{3,}', text):
        return "Excerpt from VCAA exam paper."

    # --- Stage 3: Intelligent Context Extraction ---
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        sentences = [text]
        
    # Filter out sentences that look like MCQ options or fragments
    valid_sentences = []
    for s in sentences:
        s = s.strip()
        if not s: continue
        if re.match(r'^[A-D]\.', s): continue # MCQ option A. B. C. D.
        if len(s.split()) < 3: continue # Too short
        valid_sentences.append(s)
        
    if not valid_sentences:
        return "Excerpt from VCAA exam paper."
        
    # Keyword scoring
    query_terms = [w.lower() for w in re.findall(r'\w+', user_query) if len(w) > 3]
    
    best_score = -1
    best_excerpt = ""
    
    # Window size 1-2 sentences
    for i in range(len(valid_sentences)):
        # Single sentence check
        s1 = valid_sentences[i]
        score = sum(1 for term in query_terms if term in s1.lower())
        
        if score > best_score:
            best_score = score
            best_excerpt = s1
            
        # Two sentence check
        if i < len(valid_sentences) - 1:
            s2 = valid_sentences[i+1]
            combined = s1 + " " + s2
            score_combined = sum(1 for term in query_terms if term in combined.lower())
            
            # Boost score slightly for combined to prefer context if relevant
            if score_combined > best_score + 0.5:
                best_score = score_combined
                best_excerpt = combined

    # If no good match, take the first substantial sentence
    if best_score <= 0:
        best_excerpt = valid_sentences[0]
        if len(valid_sentences) > 1 and len(best_excerpt.split()) < 20:
             best_excerpt += " " + valid_sentences[1]

    # --- Stage 4: Final Fluency & Length Check ---
    
    words = best_excerpt.split()
    
    # Truncate to ~50 words max
    if len(words) > 50:
        best_excerpt = " ".join(words[:50]) + "..."
    elif len(words) < 3: # Too short fallback
        return "Excerpt from VCAA exam paper."
        
    # Capitalize first letter
    best_excerpt = best_excerpt[0].upper() + best_excerpt[1:]
    
    # Ensure it ends with punctuation if it's a full sentence (not truncated with ...)
    if not best_excerpt.endswith('.') and not best_excerpt.endswith('...') and not best_excerpt.endswith('?'):
        best_excerpt += '.'
        
    return best_excerpt

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
                cleaned_text = clean_vcaa_chunk(chunk, question)
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


# --- Mathematical Methods Question Engine (Sequential, LaTeX, MCQ) ---

def _normalize_answer(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"^\$|\\$|\\\\", "", s)  # strip latex wrappers
    s = s.replace(" ", "")
    s = re.sub(r"^(x|y|z)\s*=", "", s)  # drop leading variable assignments
    s = s.replace("\\,", "")  # remove thin spaces
    return s

def _is_equal(a: str, b: str) -> bool:
    na, nb = _normalize_answer(a), _normalize_answer(b)
    # Try numeric compare with tolerance
    try:
        va = float(re.sub(r"[^0-9\.\-]", "", na))
        vb = float(re.sub(r"[^0-9\.\-]", "", nb))
        return abs(va - vb) < 1e-6
    except Exception:
        return na == nb

methods_questions = [
    {
        "id": 1,
        "type": "short",
        "latex": r"Let\ f:\mathbb{R}\to\mathbb{R},\ f(x)=x^{3}-3x^{2}.\ \text{Find the coordinates of the local minimum.}",
        "correct_value": "(2,-4)",
        "marks": 2,
        "rubric": "1 mark for differentiating f'(x)=3x^2-6x and solving f'(x)=0. 1 mark for minimum coordinate (2, -4).",
        "answer_kind": "point"
    },
    {
        "id": 2,
        "type": "mcq",
        "latex": r"The\ pdf\ of\ a\ continuous\ random\ variable\ X\ is\ f(x)=kx\ \text{ for }\ 0\le x\le2,\ \text{and }0\ \text{ elsewhere. Find }k.",
        "options": {
            "A": r"k=\dfrac{1}{4}",
            "B": r"k=\dfrac{1}{2}",
            "C": r"k=1",
            "D": r"k=2"
        },
        "correct_option": "B",
        "marks": 1,
        "rubric": "1 mark for setting ∫_0^2 kx dx = 1 and solving k=1/2."
    },
    {
        "id": 3,
        "type": "short",
        "latex": r"\text{Solve for }x:\ 2\ln(x)-\ln(x+2)=\ln(3).",
        "correct_value": "6",
        "marks": 3,
        "rubric": "Use log laws to combine, exponentiate to form quadratic, reject negative, x=6.",
        "answer_kind": "numeric"
    },
    {
        "id": 4,
        "type": "mcq",
        "latex": r"\text{If } f(x)=e^{2x},\ \text{ then } f'(0)=\ ?",
        "options": { "A": r"0", "B": r"1", "C": r"2", "D": r"e" },
        "correct_option": "C",
        "marks": 1,
        "rubric": "Apply chain rule: f'(x)=2e^{2x}, evaluate at x=0 → 2."
    },
    {
        "id": 5,
        "type": "short",
        "latex": r"\text{Find the average rate of change of } g(x)=x^{2}+2x \text{ over }[1,3].",
        "correct_value": "6",
        "marks": 2,
        "rubric": "Compute g(3), g(1) and apply (g(3)-g(1))/(3-1).",
        "answer_kind": "numeric"
    },
]

def _next_question_id(sess):
    done = set(sess.get("correctly_answered", []))
    current = sess.get("current_q_id")
    ids = [q["id"] for q in methods_questions]
    ids.sort()
    for qid in ids:
        if qid != current and qid not in done:
            return qid
    # If all done, wrap to first
    return ids[0]

@app.route("/methods-practice", methods=["GET", "POST"])
def methods_practice():
    # Ensure session key exists
    if 'methods_session' not in session:
        session['methods_session'] = {
            "questions_asked": [],
            "correctly_answered": [],
            "timed_on": False,
            "time_limit": 0,
            "timer_expires_at": None
        }
    
    sess = session['methods_session']
    
    action = request.args.get('action')
    if action == 'toggle_timed':
        on = request.args.get('on', '0') == '1'
        sess['timed_on'] = on
        # Recompute timer for current question
        q = next((q for q in methods_questions if q['id'] == sess.get('current_q_id')), None)
        if q and on:
            limit = 300 if (q.get('type') == 'short') else 180
            sess['time_limit'] = limit
            sess['timer_expires_at'] = time.time() + limit
        else:
            sess['time_limit'] = 0
            sess['timer_expires_at'] = None
        session.modified = True
        return redirect(url_for('methods_practice'))
    if action == 'retry':
        sess['last_answer'] = ""
        sess['feedback'] = None
        sess['attempts'] = sess.get('attempts', 0) + 1
        # Reset timer if timed mode
        q = next((q for q in methods_questions if q['id'] == sess.get('current_q_id')), None)
        if q and sess.get('timed_on'):
            limit = 300 if (q.get('type') == 'short') else 180
            sess['time_limit'] = limit
            sess['timer_expires_at'] = time.time() + limit
        session.modified = True
        return redirect(url_for('methods_practice'))
    # Load new or initial question in sequential order
    if action == 'next' or 'current_q_id' not in sess:
        next_id = _next_question_id(sess)
        sess['current_q_id'] = next_id
        sess['start_time'] = time.time()
        sess['attempts'] = 0
        sess['feedback'] = None
        sess['last_answer'] = ""
        if sess.get('timed_on'):
            q = next((q for q in methods_questions if q['id'] == next_id), None)
            limit = 300 if (q and q.get('type') == 'short') else 180
            sess['time_limit'] = limit
            sess['timer_expires_at'] = time.time() + limit
        else:
            sess['time_limit'] = 0
            sess['timer_expires_at'] = None
        session.modified = True
        return redirect(url_for('methods_practice'))

    # Retrieve current question object
    q_id = sess.get('current_q_id')
    question = next((q for q in methods_questions if q['id'] == q_id), None)
    
    if not question:
        # Fallback reset
        sess.pop('current_q_id', None)
        return redirect(url_for('methods_practice', action='next'))

    user_answer = sess.get('last_answer', "")
    speed_warning = False
    time_remaining = 0
    if sess.get('timed_on') and sess.get('timer_expires_at'):
        time_remaining = max(0, int(sess['timer_expires_at'] - time.time()))

    if request.method == "POST":
        time_expired = request.form.get('time_expired') == '1'
        choice = request.form.get('choice', '').strip()
        user_answer = request.form.get('answer', '').strip()
        working = request.form.get('working', '').strip()
        
        # Calculate time taken
        start_t = sess.get('start_time', time.time())
        time_taken = time.time() - start_t
        
        # Speed Check logic
        if time_taken < 3.0 and not working and not sess.get('feedback') and not time_expired:
            speed_warning = True
            return render_template("methods_practice.html", question=question, speed_warning=True, user_answer=user_answer, feedback=None, attempts=sess.get('attempts', 0))

        # Compute correctness locally
        is_correct = False
        if question.get('type') == 'mcq':
            is_correct = (choice.upper() == question.get('correct_option'))
        else:
            is_correct = _is_equal(user_answer, question.get('correct_value', ''))

        computed_feedback = {
            "mark": (question.get('marks', 1) if is_correct else 0),
            "max_marks": question.get('marks', 1),
            "is_correct": is_correct,
            "feedback": ("Well done." if is_correct else "Revise the method and try again.")
        }

        # Optional AI feedback enrichment
        if client:
            try:
                prompt = (
                    "You are a VCAA Mathematical Methods assessor. Provide brief feedback only.\n"
                    f"Question (LaTeX): {question.get('latex')}\n"
                    f"Correct Answer: {question.get('correct_value') or question.get('correct_option')}\n"
                    f"Student Answer: {choice or user_answer}\n"
                    f"Rubric: {question.get('rubric')}\n"
                )
                chat = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_MODEL,
                    temperature=0.2,
                    max_tokens=120
                )
                ai_text = (chat.choices[0].message.content or "").strip()
                if ai_text:
                    computed_feedback["feedback"] = ai_text
            except Exception:
                pass

        sess['feedback'] = computed_feedback

        sess['attempts'] = sess.get('attempts', 0) + 1
        sess['last_answer'] = user_answer
        if is_correct:
            # Track correctly answered to avoid repeats in this session
            corr = set(sess.get('correctly_answered', []))
            corr.add(question['id'])
            sess['correctly_answered'] = list(corr)
        asked = list(sess.get('questions_asked', []))
        if question['id'] not in asked:
            asked.append(question['id'])
            sess['questions_asked'] = asked
        # Stop timer on submission
        sess['timer_expires_at'] = None
        session.modified = True
        
    return render_template(
        "methods_practice.html",
        question=question,
        feedback=sess.get('feedback'),
        attempts=sess.get('attempts', 0),
        user_answer=user_answer,
        speed_warning=speed_warning,
        timed_on=sess.get('timed_on', False),
        time_remaining=time_remaining
    )


if __name__ == "__main__":
    env_port = os.getenv("PORT")
    port = int(env_port) if env_port else _find_available_port()
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
