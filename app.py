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
import hashlib
import sqlite3
import uuid
from flask import g, make_response

def init_db():
    conn = sqlite3.connect('vce_progress.db')
    c = conn.cursor()
    
    # 1. Anonymous Sessions Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS anonymous_sessions (
            id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            data_migrated BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # 2. Question Attempts Table
    # Ensure session_id is a TEXT column to match anonymous_sessions.id
    c.execute('''
        CREATE TABLE IF NOT EXISTS question_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question_id TEXT,
            topic TEXT,
            exam_type TEXT,
            correct BOOLEAN,
            attempt_number INTEGER,
            time_spent_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_answer TEXT,
            feedback TEXT,
            question_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

def get_db():
    conn = sqlite3.connect('vce_progress.db')
    conn.row_factory = sqlite3.Row
    return conn

def cleanup_old_sessions(days=90):
    """
    Deletes anonymous sessions and their data inactive for > `days`.
    Call this periodically (e.g. via a cron job or admin endpoint).
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Calculate cutoff date
        cursor.execute("SELECT datetime('now', ?)", (f'-{days} days',))
        cutoff = cursor.fetchone()[0]
        
        # Delete attempts linked to old sessions
        cursor.execute('''
            DELETE FROM question_attempts 
            WHERE session_id IN (
                SELECT id FROM anonymous_sessions WHERE last_activity < ?
            )
        ''', (cutoff,))
        
        # Delete the sessions themselves
        cursor.execute("DELETE FROM anonymous_sessions WHERE last_activity < ?", (cutoff,))
        
        conn.commit()
        conn.close()
        logging.info(f"Cleaned up sessions inactive since {cutoff}")
    except Exception as e:
        logging.error(f"Cleanup failed: {e}")



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

@app.before_request
def manage_anonymous_session():
    """
    Middleware to handle persistent anonymous sessions via cookies.
    """
    # Skip for static files to save DB hits
    if request.path.startswith('/static'):
        return

    cookie_id = request.cookies.get('vce_tracker_id')
    valid_session = False
    
    if cookie_id:
        # Verify if this ID exists in our DB
        conn = get_db()
        row = conn.execute("SELECT id FROM anonymous_sessions WHERE id=?", (cookie_id,)).fetchone()
        if row:
            valid_session = True
            # Update last activity
            conn.execute("UPDATE anonymous_sessions SET last_activity=CURRENT_TIMESTAMP WHERE id=?", (cookie_id,))
            conn.commit()
            session['session_id'] = cookie_id
        conn.close()
    
    if not valid_session:
        # Create new persistent session
        new_id = str(uuid.uuid4())
        try:
            conn = get_db()
            conn.execute("INSERT INTO anonymous_sessions (id) VALUES (?)", (new_id,))
            conn.commit()
            conn.close()
            
            session['session_id'] = new_id
            # Flag to set cookie in after_request
            g.set_tracker_cookie = new_id
            g.is_new_user = True
        except Exception as e:
            logging.error(f"Failed to create anonymous session: {e}")
            # Fallback to a temporary session if DB fails
            if 'session_id' not in session:
                session['session_id'] = new_id

@app.after_request
def set_tracker_cookie(response):
    """
    Sets the long-lived cookie if a new session was created.
    """
    if hasattr(g, 'set_tracker_cookie'):
        # 90 days expiry
        max_age = 90 * 24 * 60 * 60 
        response.set_cookie(
            'vce_tracker_id', 
            g.set_tracker_cookie,
            max_age=max_age,
            httponly=True,
            samesite='Lax'
        )
    return response

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

def _normalize_answer(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"^\$|\\$|\\\\", "", s)  # strip inline latex $
    s = re.sub(r"^\\\[|\\\]$", "", s)  # strip display latex \[ \]
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

# --- Methods Practice Data ---
# Layer 2: Large, Categorized Backup Bank
# REQUIREMENT: Must be expanded to 80-100+ manually verified questions.
CURATED_QUESTION_BANK = [
    # --- CALCULUS (8) ---
    {
        "id": 1, "type": "short", "text": r"\[ \text{Find the antiderivative of } 3x^2. \]",
        "correct_answer": r"\[ x^3 + C \]", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "Award 1 mark for x^3 + C."
    },
    {
        "id": 2, "type": "mcq", "text": r"\[ \text{If } f(x) = e^{2x}, \text{ then } f'(0) \text{ is equal to:} \]",
        "options": {"A": r"\( 0 \)", "B": r"\( 1 \)", "C": r"\( 2 \)", "D": r"\( e \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "f'(x) = 2e^{2x}, f'(0) = 2."
    },
    {
        "id": 3, "type": "short", "text": r"\[ \text{Evaluate } \int_0^2 (x^3 - 3x^2) \, dx. \]",
        "correct_answer": r"\[ -4 \]", "marks": 2, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "[x^4/4 - x^3] from 0 to 2 = (4-8) - 0 = -4."
    },
    {
        "id": 4, "type": "mcq", "text": r"\[ \text{The average value of } y=\sin(x) \text{ over } [0, \pi] \text{ is:} \]",
        "options": {"A": r"\( 0 \)", "B": r"\( \frac{1}{\pi} \)", "C": r"\( \frac{2}{\pi} \)", "D": r"\( 2 \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "(1/pi)*[-cos x]0->pi = 2/pi."
    },
    {
        "id": 5, "type": "short", "text": r"\[ \text{Differentiate } y = \sin(x^2) \text{ with respect to } x. \]",
        "correct_answer": r"\[ 2x\cos(x^2) \]", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "Chain rule: cos(x^2) * 2x."
    },
    {
        "id": 6, "type": "mcq", "text": r"\[ \text{The gradient of the tangent to } y = \ln(x) \text{ at } x=2 \text{ is:} \]",
        "options": {"A": r"\( 0.5 \)", "B": r"\( 1 \)", "C": r"\( 2 \)", "D": r"\( \ln(2) \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "y' = 1/x. At x=2, y' = 0.5."
    },
    {
        "id": 7, "type": "mcq", "text": r"\[ \text{The function } f(x) = x^3 - 3x \text{ has a local minimum at } x = \]",
        "options": {"A": r"\( -1 \)", "B": r"\( 0 \)", "C": r"\( 1 \)", "D": r"\( \sqrt{3} \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "f'(x) = 3x^2 - 3 = 0 -> x = +/- 1. f''(1) > 0 so min."
    },
    {
        "id": 8, "type": "short", "text": r"\[ \text{A particle has velocity } v(t) = 2t - 4. \text{ Find its displacement at } t=3 \text{ if } x(0)=0. \]",
        "correct_answer": r"\[ -3 \]", "marks": 2, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "x(t) = t^2 - 4t. x(3) = 9 - 12 = -3."
    },

    # --- ALGEBRA (8) ---
    {
        "id": 9, "type": "short", "text": r"\[ \text{Solve for } x: \log_e(x) + \log_e(2) = \log_e(6). \]",
        "correct_answer": r"\[ x=3 \]", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "log(2x) = log(6) -> 2x = 6 -> x = 3."
    },
    {
        "id": 10, "type": "mcq", "text": r"\[ \text{Simplify } \frac{a^3 b^{-2}}{a^{-1} b^3}. \]",
        "options": {"A": r"\( a^2 b \)", "B": r"\( a^4 b^{-5} \)", "C": r"\( a^2 b^{-1} \)", "D": r"\( a^4 b^5 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "a^(3 - -1) b^(-2 - 3) = a^4 b^-5."
    },
    {
        "id": 11, "type": "short", "text": r"\[ \text{Solve the system: } 2x + y = 5, x - y = 4. \]",
        "correct_answer": r"\[ x=3, y=-1 \]", "marks": 2, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "Add equations: 3x = 9 -> x=3. 3-y=4 -> y=-1."
    },
    {
        "id": 12, "type": "mcq", "text": r"\[ \text{Which of these is a factor of } P(x) = x^3 - 2x^2 - x + 2? \]",
        "options": {"A": r"\( x+1 \)", "B": r"\( x+2 \)", "C": r"\( x-3 \)", "D": r"\( x \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "P(-1) = -1 - 2 + 1 + 2 = 0."
    },
    {
        "id": 13, "type": "short", "text": r"\[ \text{Find the discriminant of } 2x^2 + 4x + 5 = 0. \]",
        "correct_answer": r"\[ -24 \]", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "b^2 - 4ac = 16 - 4(2)(5) = 16 - 40 = -24."
    },
    {
        "id": 14, "type": "mcq", "text": r"\[ \text{The coefficient of } x^2 \text{ in } (2x+1)^4 \text{ is:} \]",
        "options": {"A": r"\( 4 \)", "B": r"\( 6 \)", "C": r"\( 12 \)", "D": r"\( 24 \)"},
        "correct_answer": "D", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "4C2 * (2x)^2 * (1)^2 = 6 * 4x^2 = 24x^2."
    },
    {
        "id": 15, "type": "mcq", "text": r"\[ \text{If } f(x) = 2x - 3, \text{ then } f^{-1}(x) \text{ is:} \]",
        "options": {"A": r"\( \frac{x+3}{2} \)", "B": r"\( \frac{x-3}{2} \)", "C": r"\( 2x+3 \)", "D": r"\( \frac{1}{2x-3} \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "y = 2x - 3 -> x = 2y - 3 -> y = (x+3)/2."
    },
    {
        "id": 16, "type": "short", "text": r"\[ \text{Make } a \text{ the subject of } v = u + at. \]",
        "correct_answer": r"\[ a = \frac{v-u}{t} \]", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "v-u = at -> a = (v-u)/t."
    },

    # --- PROBABILITY (7) ---
    {
        "id": 17, "type": "short", "text": r"\[ \text{If Var}(X) = 2, \text{ find Var}(3X - 1). \]",
        "correct_answer": r"\[ 18 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "3^2 * Var(X) = 9 * 2 = 18."
    },
    {
        "id": 18, "type": "mcq", "text": r"\[ \text{If } A \text{ and } B \text{ are independent, } P(A)=0.3, P(B)=0.4, \text{ then } P(A \cap B) = ? \]",
        "options": {"A": r"\( 0.7 \)", "B": r"\( 0.12 \)", "C": r"\( 0.1 \)", "D": r"\( 0 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "P(A)*P(B) = 0.3 * 0.4 = 0.12."
    },
    {
        "id": 19, "type": "short", "text": r"\[ X \sim N(10, 2^2). \text{ Find } Pr(8 < X < 12) \text{ approx.} \]",
        "correct_answer": r"\[ 0.68 \]", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "Within 1 SD of mean is approx 0.68."
    },
    {
        "id": 20, "type": "mcq", "text": r"\[ X \sim \text{Bi}(5, 0.2). \text{ The mean of } X \text{ is:} \]",
        "options": {"A": r"\( 0.2 \)", "B": r"\( 1 \)", "C": r"\( 2 \)", "D": r"\( 5 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "np = 5 * 0.2 = 1."
    },
    {
        "id": 21, "type": "short", "text": r"\[ \text{Given } P(A|B) = 0.5 \text{ and } P(B) = 0.4, \text{ find } P(A \cap B). \]",
        "correct_answer": r"\[ 0.2 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "P(A|B) = P(A n B)/P(B) -> 0.5 * 0.4 = 0.2."
    },
    {
        "id": 22, "type": "mcq", "text": r"\[ \text{For a sample proportion } \hat{p} \text{ with } n=100, p=0.1, \text{ the SD is:} \]",
        "options": {"A": r"\( 0.09 \)", "B": r"\( 0.03 \)", "C": r"\( 0.0009 \)", "D": r"\( 0.3 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "sqrt(p(1-p)/n) = sqrt(0.09/100) = 0.3/10 = 0.03."
    },
    {
        "id": 23, "type": "mcq", "text": r"\[ \text{If } f(x) = kx^2 \text{ is a PDF on } [0, 1], \text{ find } k. \]",
        "options": {"A": r"\( 1 \)", "B": r"\( 2 \)", "C": r"\( 3 \)", "D": r"\( 4 \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "Int kx^2 = [kx^3/3] = k/3 = 1 -> k=3."
    },

    # --- FUNCTIONS (7) ---
    {
        "id": 24, "type": "short", "text": r"\[ \text{State the maximal domain of } f(x) = \frac{1}{\sqrt{x-2}}. \]",
        "correct_answer": r"\[ (2, \infty) \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "x-2 > 0 -> x > 2."
    },
    {
        "id": 25, "type": "mcq", "text": r"\[ \text{The graph of } y = f(x-1) + 2 \text{ is } y=f(x) \text{ shifted:} \]",
        "options": {"A": r"\( \text{Right 1, Up 2} \)", "B": r"\( \text{Left 1, Up 2} \)", "C": r"\( \text{Right 1, Down 2} \)", "D": r"\( \text{Left 1, Down 2} \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "x-1 is Right 1, +2 is Up 2."
    },
    {
        "id": 26, "type": "short", "text": r"\[ \text{If } f(x) = x^2 \text{ and } g(x) = 2x+1, \text{ find } f(g(1)). \]",
        "correct_answer": r"\[ 9 \]", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "g(1)=3, f(3)=9."
    },
    {
        "id": 27, "type": "mcq", "text": r"\[ \text{The vertical asymptote of } y = \frac{3}{2x-4} + 1 \text{ is:} \]",
        "options": {"A": r"\( x=2 \)", "B": r"\( x=4 \)", "C": r"\( x=-2 \)", "D": r"\( y=1 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "2x-4=0 -> x=2."
    },
    {
        "id": 28, "type": "short", "text": r"\[ \text{Is } f(x) = x^3 + x \text{ odd, even, or neither?} \]",
        "correct_answer": r"\[ \text{Odd} \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "f(-x) = -x^3 - x = -f(x)."
    },
    {
        "id": 29, "type": "mcq", "text": r"\[ \text{Which function has an inverse function?} \]",
        "options": {"A": r"\( y=x^2 \)", "B": r"\( y=\sin(x) \)", "C": r"\( y=e^x \)", "D": r"\( y=|x| \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "One-to-one."
    },
    {
        "id": 30, "type": "mcq", "text": r"\[ \text{The period of } y = \cos(\frac{\pi x}{2}) \text{ is:} \]",
        "options": {"A": r"\( 1 \)", "B": r"\( 2 \)", "C": r"\( 4 \)", "D": r"\( \pi \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "2pi / (pi/2) = 4."
    },
    # --- NEW ROBUST QUESTIONS ---
    {
        "id": 31, "type": "short", "text": r"\[ \text{Find the derivative of } f(x) = x^2 e^{2x}. \]",
        "correct_answer": r"\[ 2xe^{2x}(1+x) \]", "marks": 2, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "Product rule: u=x^2, v=e^{2x}. f' = 2x e^{2x} + x^2 (2e^{2x}) = 2xe^{2x}(1+x)."
    },
    {
        "id": 32, "type": "mcq", "text": r"\[ \text{The average value of } f(x) = \sin(x) \text{ over } [0, \pi] \text{ is:} \]",
        "options": {"A": r"\( 0 \)", "B": r"\( \frac{2}{\pi} \)", "C": r"\( \frac{1}{\pi} \)", "D": r"\( 2 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "Avg = (1/pi) * Int(sin x) from 0 to pi = (1/pi) * [-cos x] = (1/pi) * (1 - (-1)) = 2/pi."
    },
    {
        "id": 33, "type": "short", "text": r"\[ \text{Solve } 2\cos(x) = -\sqrt{3} \text{ for } x \in [0, 2\pi]. \]",
        "correct_answer": r"\[ x = \frac{5\pi}{6}, \frac{7\pi}{6} \]", "marks": 2, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "cos(x) = -sqrt(3)/2. Ref angle pi/6. Quad 2 & 3. x = 5pi/6, 7pi/6."
    },
    {
        "id": 34, "type": "mcq", "text": r"\[ \text{If } X \sim N(20, 4) \text{ and } Z \sim N(0,1), \text{ then } P(X < 23) \text{ is equal to:} \]",
        "options": {"A": r"\( P(Z < 1.5) \)", "B": r"\( P(Z < 0.75) \)", "C": r"\( P(Z < 3) \)", "D": r"\( P(Z < -1.5) \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "Z = (23-20)/2 = 3/2 = 1.5."
    },
    {
        "id": 35, "type": "short", "text": r"\[ \text{Find the coordinates of the turning point of } y = 2(x-1)^2 + 3. \]",
        "correct_answer": r"\[ (1, 3) \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "Vertex form a(x-h)^2 + k. Vertex is (h, k) = (1, 3)."
    },

    # --- EXPANDED CALCULUS (36-50) ---
    {
        "id": 36, "type": "short", "text": r"\[ \text{Differentiate } y = \sqrt{3x+1} \text{ with respect to } x. \]",
        "correct_answer": r"\[ \frac{3}{2\sqrt{3x+1}} \]", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "Chain rule: 1/2(3x+1)^(-1/2) * 3."
    },
    {
        "id": 37, "type": "short", "text": r"\[ \text{Given } y = x^2 \sin(x), \text{ find } \frac{dy}{dx}. \]",
        "correct_answer": r"\[ 2x\sin(x) + x^2\cos(x) \]", "marks": 2, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "Product rule: u'v + uv'."
    },
    {
        "id": 38, "type": "short", "text": r"\[ \text{If } f(x) = \frac{e^x}{x}, \text{ find } f'(x). \]",
        "correct_answer": r"\[ \frac{e^x(x-1)}{x^2} \]", "marks": 2, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "Quotient rule: (x e^x - e^x * 1)/x^2."
    },
    {
        "id": 39, "type": "mcq", "text": r"\[ \text{The equation of the tangent to } y=x^2 \text{ at } x=1 \text{ is:} \]",
        "options": {"A": r"\( y=2x \)", "B": r"\( y=2x-1 \)", "C": r"\( y=x+1 \)", "D": r"\( y=2x+1 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "m=2(1)=2. Point (1,1). y-1 = 2(x-1) -> y=2x-1."
    },
    {
        "id": 40, "type": "short", "text": r"\[ \text{Find the gradient of the normal to } y=e^{2x} \text{ at } x=0. \]",
        "correct_answer": r"\[ -\frac{1}{2} \]", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "dy/dx = 2e^{2x}. At x=0, m_T = 2. m_N = -1/2."
    },
    {
        "id": 41, "type": "mcq", "text": r"\[ \text{The stationary points of } f(x) = x^3 - 12x \text{ occur at:} \]",
        "options": {"A": r"\( x=2 \)", "B": r"\( x=-2 \)", "C": r"\( x=\pm 2 \)", "D": r"\( x=0 \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "3x^2 - 12 = 0 -> x^2=4 -> x=2, -2."
    },
    {
        "id": 42, "type": "short", "text": r"\[ \text{Find the area bounded by } y=x^2 \text{ and the x-axis from } x=0 \text{ to } x=3. \]",
        "correct_answer": r"\[ 9 \]", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "Int x^2 dx = [x^3/3] from 0 to 3 = 27/3 = 9."
    },
    {
        "id": 43, "type": "short", "text": r"\[ \text{A particle has } a(t) = 6t. \text{ If } v(0)=2, \text{ find } v(t). \]",
        "correct_answer": r"\[ v(t) = 3t^2 + 2 \]", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "Int 6t dt = 3t^2 + C. C=2."
    },
    {
        "id": 44, "type": "mcq", "text": r"\[ \text{The position of a particle is } x(t) = t^2 - 4t. \text{ The distance travelled in the first 3 seconds is:} \]",
        "options": {"A": r"\( 3 \)", "B": r"\( 4 \)", "C": r"\( 5 \)", "D": r"\( -3 \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "Turns at t=2. x(0)=0, x(2)=-4, x(3)=-3. Dist = |0 to -4| + |-4 to -3| = 4 + 1 = 5."
    },
    {
        "id": 45, "type": "short", "text": r"\[ \text{Find the maximum value of } f(x) = -x^2 + 4x + 1. \]",
        "correct_answer": r"\[ 5 \]", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "Vertex at x = -4/-2 = 2. f(2) = -4 + 8 + 1 = 5."
    },
    {
        "id": 46, "type": "mcq", "text": r"\[ \text{The average rate of change of } f(x)=x^2 \text{ over } [1, 3] \text{ is:} \]",
        "options": {"A": r"\( 2 \)", "B": r"\( 3 \)", "C": r"\( 4 \)", "D": r"\( 8 \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "(f(3)-f(1))/(3-1) = (9-1)/2 = 4."
    },
    {
        "id": 47, "type": "short", "text": r"\[ \text{Find the instantaneous rate of change of } V = \frac{4}{3}\pi r^3 \text{ with respect to } r \text{ when } r=2. \]",
        "correct_answer": r"\[ 16\pi \]", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "dV/dr = 4pi r^2. At r=2, 16pi."
    },
    {
        "id": 48, "type": "mcq", "text": r"\[ \text{If } f'(x) > 0 \text{ and } f''(x) < 0, \text{ the graph is:} \]",
        "options": {"A": r"\( \text{Increasing and concave up} \)", "B": r"\( \text{Increasing and concave down} \)", "C": r"\( \text{Decreasing and concave up} \)", "D": r"\( \text{Decreasing and concave down} \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "f' > 0 -> inc. f'' < 0 -> concave down."
    },
    {
        "id": 49, "type": "short", "text": r"\[ \text{Evaluate } \int (2x+1)^3 \, dx. \]",
        "correct_answer": r"\[ \frac{1}{8}(2x+1)^4 + C \]", "marks": 1, "exam_type": "tech_free", "topic": "Calculus",
        "rubric": "Linear sub: 1/2 * (2x+1)^4 / 4."
    },
    {
        "id": 50, "type": "mcq", "text": r"\[ \text{If } F(x) = \int_1^x \frac{1}{t} \, dt, \text{ then } F'(x) \text{ is:} \]",
        "options": {"A": r"\( \ln(x) \)", "B": r"\( \frac{1}{x} \)", "C": r"\( -\frac{1}{x^2} \)", "D": r"\( e^x \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Calculus",
        "rubric": "FTC: d/dx Int_a^x f(t) dt = f(x)."
    },

    # --- EXPANDED ALGEBRA (51-65) ---
    {
        "id": 51, "type": "short", "text": r"\[ \text{Simplify } 2\log_e(3) - \log_e(9). \]",
        "correct_answer": r"\[ 0 \]", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "log(3^2) - log(9) = log(9) - log(9) = 0."
    },
    {
        "id": 52, "type": "mcq", "text": r"\[ \text{Solve } 2^{x+1} = 8^{x-1}. \]",
        "options": {"A": r"\( 1 \)", "B": r"\( 2 \)", "C": r"\( 3 \)", "D": r"\( 4 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "2^{x+1} = (2^3)^{x-1} -> x+1 = 3x-3 -> 2x=4 -> x=2."
    },
    {
        "id": 53, "type": "short", "text": r"\[ \text{Express } x^2 + 6x + 5 \text{ in the form } (x+h)^2 + k. \]",
        "correct_answer": r"\[ (x+3)^2 - 4 \]", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "(x+3)^2 - 9 + 5 = (x+3)^2 - 4."
    },
    {
        "id": 54, "type": "mcq", "text": r"\[ \text{The remainder when } P(x) = x^3 + 2x^2 - 5x + 1 \text{ is divided by } (x-1) \text{ is:} \]",
        "options": {"A": r"\( -1 \)", "B": r"\( 0 \)", "C": r"\( 1 \)", "D": r"\( -2 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "P(1) = 1 + 2 - 5 + 1 = -1."
    },
    {
        "id": 55, "type": "short", "text": r"\[ \text{Perform the division: } (x^3 - 1) \div (x-1). \]",
        "correct_answer": r"\[ x^2 + x + 1 \]", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "Standard factorization difference of cubes."
    },
    {
        "id": 56, "type": "short", "text": r"\[ \text{Solve for } x: y = x+2 \text{ and } y = x^2. \]",
        "correct_answer": r"\[ x=2, x=-1 \]", "marks": 2, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "x^2 = x+2 -> x^2 - x - 2 = 0 -> (x-2)(x+1)=0."
    },
    {
        "id": 57, "type": "mcq", "text": r"\[ \text{The determinant of } A = \begin{bmatrix} 2 & 3 \\ 1 & 4 \end{bmatrix} \text{ is:} \]",
        "options": {"A": r"\( 5 \)", "B": r"\( 11 \)", "C": r"\( 2 \)", "D": r"\( 8 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "ad - bc = 8 - 3 = 5."
    },
    {
        "id": 58, "type": "short", "text": r"\[ \text{Find the inverse of } A = \begin{bmatrix} 3 & 1 \\ 5 & 2 \end{bmatrix}. \]",
        "correct_answer": r"\[ \begin{bmatrix} 2 & -1 \\ -5 & 3 \end{bmatrix} \]", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "det = 6-5=1. Swap diag, negate off-diag."
    },
    {
        "id": 59, "type": "mcq", "text": r"\[ \text{The transformation defined by } \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} \text{ represents:} \]",
        "options": {"A": r"\( \text{Dilation factor 2 from x-axis} \)", "B": r"\( \text{Dilation factor 2 from y-axis} \)", "C": r"\( \text{Reflection in x-axis} \)", "D": r"\( \text{Translation up 2} \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "y' = 2y, x' = x. Dilation from x-axis."
    },
    {
        "id": 60, "type": "short", "text": r"\[ \text{Find the third term in the expansion of } (x+2)^4. \]",
        "correct_answer": r"\[ 24x^2 \]", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "4C2 * x^2 * 2^2 = 6 * x^2 * 4 = 24x^2."
    },
    {
        "id": 61, "type": "mcq", "text": r"\[ \text{Solve for } x: ax - 3 = 5. \]",
        "options": {"A": r"\( x=8/a \)", "B": r"\( x=2/a \)", "C": r"\( x=8a \)", "D": r"\( x=5+3a \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "ax = 8 -> x = 8/a."
    },
    {
        "id": 62, "type": "short", "text": r"\[ \text{If } f(x) = 2^x, \text{ simplify } f(x+2) - f(x). \]",
        "correct_answer": r"\[ 3 \cdot 2^x \]", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "2^{x+2} - 2^x = 4*2^x - 1*2^x = 3*2^x."
    },
    {
        "id": 63, "type": "mcq", "text": r"\[ \text{The solution to } x^2 - 9 < 0 \text{ is:} \]",
        "options": {"A": r"\( x < 3 \)", "B": r"\( x > -3 \)", "C": r"\( -3 < x < 3 \)", "D": r"\( x < -3 \cup x > 3 \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "Between the roots -3 and 3."
    },
    {
        "id": 64, "type": "short", "text": r"\[ \text{Simplify } \sqrt{75} - \sqrt{12}. \]",
        "correct_answer": r"\[ 3\sqrt{3} \]", "marks": 1, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "5sqrt(3) - 2sqrt(3) = 3sqrt(3)."
    },
    {
        "id": 65, "type": "mcq", "text": r"\[ \text{The system } kx + y = 3, 2x + y = 4 \text{ has no solution if } k = \]",
        "options": {"A": r"\( 1 \)", "B": r"\( 2 \)", "C": r"\( 3 \)", "D": r"\( -2 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "Parallel lines: gradients equal. -k = -2 -> k=2."
    },

    # --- EXPANDED PROBABILITY (66-80) ---
    {
        "id": 66, "type": "short", "text": r"\[ \text{A discrete RV } X \text{ has } P(X=0)=0.2, P(X=1)=0.8. \text{ Find } E(X). \]",
        "correct_answer": r"\[ 0.8 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "0*0.2 + 1*0.8 = 0.8."
    },
    {
        "id": 67, "type": "mcq", "text": r"\[ \text{If } X \sim \text{Bi}(10, 0.5), \text{ then } \text{Var}(X) \text{ is:} \]",
        "options": {"A": r"\( 2.5 \)", "B": r"\( 5 \)", "C": r"\( 25 \)", "D": r"\( 0.25 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "np(1-p) = 10 * 0.5 * 0.5 = 2.5."
    },
    {
        "id": 68, "type": "short", "text": r"\[ \text{If } X \sim \text{Bi}(3, 0.5), \text{ find } P(X=3). \]",
        "correct_answer": r"\[ 0.125 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "(0.5)^3 = 1/8 = 0.125."
    },
    {
        "id": 69, "type": "mcq", "text": r"\[ \text{If } Z \sim N(0,1), \text{ which is true?} \]",
        "options": {"A": r"\( P(Z<0) = 0.5 \)", "B": r"\( P(Z>1) = 0.5 \)", "C": r"\( P(Z=0) = 0.5 \)", "D": r"\( P(Z<0) = 0 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "Symmetry about 0."
    },
    {
        "id": 70, "type": "short", "text": r"\[ \text{Standardize } x=25 \text{ if } \mu=20, \sigma=2. \]",
        "correct_answer": r"\[ z=2.5 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "(25-20)/2 = 2.5."
    },
    {
        "id": 71, "type": "mcq", "text": r"\[ \text{For a 95\% confidence interval, the z-score used is approx:} \]",
        "options": {"A": r"\( 1.64 \)", "B": r"\( 1.96 \)", "C": r"\( 2.58 \)", "D": r"\( 1 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "Standard value for 95%."
    },
    {
        "id": 72, "type": "short", "text": r"\[ \text{Find the SD of } \hat{p} \text{ if } p=0.5, n=4. \]",
        "correct_answer": r"\[ 0.25 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "sqrt(0.25/4) = 0.5/2 = 0.25."
    },
    {
        "id": 73, "type": "mcq", "text": r"\[ \text{The width of a confidence interval decreases if:} \]",
        "options": {"A": r"\( n \text{ decreases} \)", "B": r"\( n \text{ increases} \)", "C": r"\( \text{confidence level increases} \)", "D": r"\( \hat{p} \text{ moves to 0.5} \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "Width prop to 1/sqrt(n)."
    },
    {
        "id": 74, "type": "short", "text": r"\[ \text{If } P(A)=0.4, P(B|A)=0.5, \text{ find } P(A \cap B). \]",
        "correct_answer": r"\[ 0.2 \]", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "P(A n B) = P(B|A)P(A) = 0.5 * 0.4 = 0.2."
    },
    {
        "id": 75, "type": "mcq", "text": r"\[ \text{Events } A, B \text{ are mutually exclusive if:} \]",
        "options": {"A": r"\( P(A \cap B) = P(A)P(B) \)", "B": r"\( P(A \cap B) = 0 \)", "C": r"\( P(A \cup B) = 1 \)", "D": r"\( P(A)=P(B) \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "Definition."
    },
    {
        "id": 76, "type": "short", "text": r"\[ \text{Find } k \text{ if } f(x)=k \text{ is a PDF on } [0, 5]. \]",
        "correct_answer": r"\[ 0.2 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "Area = 5k = 1 -> k=0.2."
    },
    {
        "id": 77, "type": "mcq", "text": r"\[ \text{The mean of the PDF } f(x)=2x \text{ on } [0, 1] \text{ is:} \]",
        "options": {"A": r"\( 1/2 \)", "B": r"\( 2/3 \)", "C": r"\( 1/3 \)", "D": r"\( 3/4 \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "Int x * 2x dx = [2x^3/3]0-1 = 2/3."
    },
    {
        "id": 78, "type": "short", "text": r"\[ \text{Find the median } m \text{ of } f(x)=2x \text{ on } [0, 1]. \]",
        "correct_answer": r"\[ \frac{1}{\sqrt{2}} \]", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "Int_0^m 2x dx = 0.5 -> m^2 = 0.5 -> m = 1/sqrt(2)."
    },
    {
        "id": 79, "type": "short", "text": r"\[ \text{A coin is tossed 3 times. Find the probability of exactly 2 heads. } \]",
        "correct_answer": r"\[ 0.375 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "3C2 (0.5)^3 = 3 * 0.125 = 0.375."
    },
    {
        "id": 80, "type": "mcq", "text": r"\[ \text{Which is NOT a property of a PDF } f(x)? \]",
        "options": {"A": r"\( f(x) \ge 0 \)", "B": r"\( \int f(x) dx = 1 \)", "C": r"\( f(x) \le 1 \)", "D": r"\( \text{Defined on domain} \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "PDF can be > 1 (e.g. uniform on [0, 0.5] has height 2)."
    },

    # --- EXPANDED FUNCTIONS (81-95) ---
    {
        "id": 81, "type": "short", "text": r"\[ \text{State the domain of } y = \log_e(x-3). \]",
        "correct_answer": r"\[ (3, \infty) \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "x-3 > 0."
    },
    {
        "id": 82, "type": "mcq", "text": r"\[ \text{The inverse of } y=x^3+1 \text{ is:} \]",
        "options": {"A": r"\( y=\sqrt[3]{x-1} \)", "B": r"\( y=\sqrt[3]{x+1} \)", "C": r"\( y=(x-1)^3 \)", "D": r"\( y=x^{1/3}-1 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "x = y^3+1 -> y^3=x-1 -> y=(x-1)^(1/3)."
    },
    {
        "id": 83, "type": "short", "text": r"\[ \text{If } f(x)=\sqrt{x} \text{ and } g(x)=x-1, \text{ find the domain of } f(g(x)). \]",
        "correct_answer": r"\[ [1, \infty) \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "x-1 >= 0 -> x >= 1."
    },
    {
        "id": 84, "type": "mcq", "text": r"\[ \text{The transformation } (x,y) \to (x+1, 2y) \text{ maps } y=x^2 \text{ to:} \]",
        "options": {"A": r"\( y=2(x-1)^2 \)", "B": r"\( y=2(x+1)^2 \)", "C": r"\( y=\frac{1}{2}(x-1)^2 \)", "D": r"\( y=\frac{1}{2}(x+1)^2 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "x' = x+1, y' = 2y. x=x'-1, y=y'/2. y'/2 = (x'-1)^2 -> y=2(x-1)^2."
    },
    {
        "id": 85, "type": "short", "text": r"\[ \text{State the amplitude of } y = -3\cos(2x) + 1. \]",
        "correct_answer": r"\[ 3 \]", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "|-3| = 3."
    },
    {
        "id": 86, "type": "mcq", "text": r"\[ \text{The range of } y = 2\sin(x) - 1 \text{ is:} \]",
        "options": {"A": r"\( [-1, 1] \)", "B": r"\( [-3, 1] \)", "C": r"\( [-2, 2] \)", "D": r"\( [-1, 3] \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "[-2-1, 2-1] = [-3, 1]."
    },
    {
        "id": 87, "type": "short", "text": r"\[ \text{State the equation of the horizontal asymptote of } y = 3e^{-x} + 2. \]",
        "correct_answer": r"\[ y=2 \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "As x -> inf, e^-x -> 0, y -> 2."
    },
    {
        "id": 88, "type": "mcq", "text": r"\[ \text{The asymptotes of } y = \frac{1}{x-1} + 2 \text{ are:} \]",
        "options": {"A": r"\( x=1, y=2 \)", "B": r"\( x=-1, y=2 \)", "C": r"\( x=1, y=-2 \)", "D": r"\( x=0, y=2 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "Denom zero at x=1. Vertical shift 2."
    },
    {
        "id": 89, "type": "short", "text": r"\[ \text{State the maximal domain of } y = \frac{1}{(x-2)^2}. \]",
        "correct_answer": r"\[ R \setminus \{2\} \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "All reals except where denom is zero."
    },
    {
        "id": 90, "type": "mcq", "text": r"\[ \text{For } f(x) = \begin{cases} x, x<0 \\ x^2, x \ge 0 \end{cases}, \text{ the function is:} \]",
        "options": {"A": r"\( \text{Continuous and differentiable everywhere} \)", "B": r"\( \text{Continuous but not differentiable at } x=0 \)", "C": r"\( \text{Discontinuous at } x=0 \)", "D": r"\( \text{Neither} \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "Cont: 0=0. Diff: LHD=1, RHD=0. Not diff."
    },
    {
        "id": 91, "type": "short", "text": r"\[ \text{For what values of } k \text{ is } f(x)=kx+3 \text{ strictly increasing?} \]",
        "correct_answer": r"\[ k > 0 \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "Gradient must be positive."
    },
    {
        "id": 92, "type": "mcq", "text": r"\[ \text{The function } f(x) = \cos(x) \text{ is:} \]",
        "options": {"A": r"\( \text{Odd} \)", "B": r"\( \text{Even} \)", "C": r"\( \text{Neither} \)", "D": r"\( \text{One-to-one} \)"},
        "correct_answer": "B", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "cos(-x) = cos(x)."
    },
    {
        "id": 93, "type": "short", "text": r"\[ \text{Find the x-intercepts of } y = x^2 - 5x + 6. \]",
        "correct_answer": r"\[ x=2, x=3 \]", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "(x-2)(x-3)=0."
    },
    {
        "id": 94, "type": "mcq", "text": r"\[ \text{The period of } y = \tan(2x) \text{ is:} \]",
        "options": {"A": r"\( \pi \)", "B": r"\( 2\pi \)", "C": r"\( \pi/2 \)", "D": r"\( \pi/4 \)"},
        "correct_answer": "C", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "pi/n = pi/2."
    },
    {
        "id": 95, "type": "short", "text": r"\[ \text{Sketch } y = |x-1|. \text{ The minimum value is:} \]",
        "correct_answer": r"\[ 0 \]", "marks": 1, "exam_type": "tech_free", "topic": "Functions",
        "rubric": "Vertex at (1,0)."
    },

    # --- FINAL ADDITIONS (96-100) ---
    {
        "id": 96, "type": "short", "text": r"\[ \text{Solve for } x: 3^{2x} - 10 \cdot 3^x + 9 = 0. \]",
        "correct_answer": r"\[ x=0, x=2 \]", "marks": 2, "exam_type": "tech_free", "topic": "Algebra",
        "rubric": "Let u=3^x. u^2-10u+9=0 -> (u-9)(u-1)=0. 3^x=9 or 1."
    },
    {
        "id": 97, "type": "mcq", "text": r"\[ \text{Using Newton's method for } x^3 + x - 1 = 0 \text{ with } x_0 = 1, \text{ the next estimate } x_1 \text{ is:} \]",
        "options": {"A": r"\( 0.75 \)", "B": r"\( 0.68 \)", "C": r"\( 0.5 \)", "D": r"\( 1.2 \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Algebra",
        "rubric": "f(1)=1, f'(1)=4. x1 = 1 - 1/4 = 0.75."
    },
    {
        "id": 98, "type": "short", "text": r"\[ \text{For PDF } f(x) = 3x^2 \text{ on } [0,1], \text{ find } Pr(X > 0.5). \]",
        "correct_answer": r"\[ 0.875 \]", "marks": 1, "exam_type": "tech_free", "topic": "Probability",
        "rubric": "Integral from 0.5 to 1 of 3x^2 is [x^3] = 1 - 0.125 = 0.875."
    },
    {
        "id": 99, "type": "mcq", "text": r"\[ \text{If } X \sim \text{Bi}(10, 0.3), \text{ then } Pr(X \ge 1) \text{ is equal to:} \]",
        "options": {"A": r"\( 1 - 0.7^{10} \)", "B": r"\( 1 - 0.3^{10} \)", "C": r"\( 0.3 \)", "D": r"\( 0.7^{10} \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Probability",
        "rubric": "1 - P(X=0) = 1 - (0.7)^10."
    },
    {
        "id": 100, "type": "mcq", "text": r"\[ \text{The range of } f(x) = e^{-(x-1)^2} + 2 \text{ is:} \]",
        "options": {"A": r"\( (2, 3] \)", "B": r"\( [2, 3] \)", "C": r"\( (2, 3) \)", "D": r"\( [2, \infty) \)"},
        "correct_answer": "A", "marks": 1, "exam_type": "tech_active", "topic": "Functions",
        "rubric": "Max value at x=1 is e^0+2=3. As x->inf, approaches 2. Range (2, 3]."
    },

]

UNIVERSAL_DEFAULT_QUESTION = {
    "id": "universal_backup",
    "type": "short",
    "text": r"\[ \text{Find the derivative of } f(x) = x^2. \]",
    "correct_answer": r"\[ 2x \]",
    "marks": 1,
    "topic": "Calculus",
    "exam_type": "tech_free",
    "rubric": "Power rule: nx^{n-1}."
}

def is_question_sane(text):
    """
    Aggressive validation to catch broken AI outputs like '( cdot' or empty math.
    """
    if not text:
        return False
        
    # List of patterns that indicate a nonsensical/incomplete question
    invalid_patterns = [
        r'\(\s*\\cdot',      # Catches '( cdot' and '( \cdot'
        r'Consider the function \(\.', # Catches 'Consider the function (.'
        r'\\\(.*?\\\)\s*\)',  # Catches nested '\( ... \)' (e.g. \(\(x\)\)) which breaks rendering
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, text):
            return False
            
    # Additionally, ensure there is at least one complete LaTeX expression
    # Must contain \( ... \) with some content inside
    if not re.search(r'\\\(.*?[a-zA-Z0-9].*?\\\)', text) and not re.search(r'\\\[.*?[a-zA-Z0-9].*?\\\]', text):
        return False
        
    return True

def clean_vcaa_question_chunk(raw_chunk):
    """
    Extracts a complete question with context from a raw VCAA chunk.
    Refined to include preceding context and stop at delimiters.
    """
    if not raw_chunk:
        return None
        
    text = raw_chunk.strip()
    
    # 1. Remove Metadata tags
    text = re.sub(r'\[.*?\]', '', text)
    
    # 2. Identify Start Point (Prioritize complete questions)
    # Look for "Question X", "Part a", or just the start
    start_index = 0
    
    # Regex for explicit headers: "Question 1", "Part A", "SECTION B"
    header_match = re.search(r'(?:Question\s+\d+|Part\s+[A-Za-z]|SECTION\s+[A-Z])', text, re.IGNORECASE)
    
    # Regex for sub-parts: "a.", "ii."
    subpart_match = re.search(r'(?:^|\n)\s*(?:[a-z]\.|[ivx]+\.)\s+', text)
    
    if header_match:
        # Start from the header
        start_index = header_match.start()
    elif subpart_match:
        # Found a sub-part. Scan BACKWARDS for context (up to 200 chars or 3 lines)
        # Look for keywords: "Let", "Consider", "Given", "The function"
        idx = subpart_match.start()
        lookback_limit = max(0, idx - 300)
        context_window = text[lookback_limit:idx]
        
        # Search for context keywords in the window
        context_start = -1
        keywords = ["Let ", "Consider ", "Given ", "The function ", "Suppose "]
        
        for kw in keywords:
            # Find last occurrence of keyword before the subpart
            kw_idx = context_window.rfind(kw)
            if kw_idx != -1:
                # If found, set context_start relative to text
                found_abs = lookback_limit + kw_idx
                if found_abs > context_start:
                    context_start = found_abs
        
        if context_start != -1:
            start_index = context_start
        else:
            # Fallback: if near start, take from 0. Else just take the subpart.
            if idx < 100:
                start_index = 0
            else:
                start_index = idx

    # 3. Extract Candidate Block
    candidate = text[start_index:]
    
    # 4. Find End Point (Stop at next delimiter)
    # Delimiters: Next "Question", "SECTION", sub-part "b.", "iii.", or "marks"
    
    # We want to capture the CURRENT question/part.
    # If we started at "Question 1", we stop at "Question 2".
    # If we started at "a.", we stop at "b.".
    
    # Common delimiters indicating NEW question/part
    delimiters = [
        r'\n\s*Question\s+\d+',
        r'\n\s*SECTION\s+[A-Z]',
        r'\n\s*[a-z]\.\s',
        r'\n\s*[ivx]+\.\s',
        r'\(\d+\s*marks?\)' # Stop AFTER marks
    ]
    
    cutoff_index = len(candidate)
    
    for delim in delimiters:
        # Find first match of this delimiter
        # We skip the very start of candidate if it matches the delimiter (e.g. we started with "a.")
        match = re.search(delim, candidate[5:], re.IGNORECASE) # offset 5 to skip self
        if match:
            # If delimiter is "marks", we include it. Others we exclude.
            if "marks" in delim:
                end_pos = match.end() + 5 # +5 offset
                if end_pos < cutoff_index:
                    cutoff_index = end_pos
            else:
                start_pos = match.start() + 5
                if start_pos < cutoff_index:
                    cutoff_index = start_pos

    final_text = candidate[:cutoff_index].strip()
    
    # Post-cleaning: Remove "Question X" header if it's just a label at the start?
    # User requested bold text. We'll leave the header as part of the text but clean up spacing.
    final_text = re.sub(r'\s+', ' ', final_text)
    
    if len(final_text) < 15:
        return None
        
    return final_text

def normalize_latex_delimiters(text):
    """
    Replaces $...$ with \(...\) or \[...\] to ensure MathJax compatibility.
    - Uses \[...\] for blocks that look like display math (contain newlines or double $$).
    - Uses \(...\) for inline math.
    """
    if not text:
        return ""
    
    # 1. Replace double dollars $$...$$ with \[...\]
    text = re.sub(r'\$\$(.*?)\$\$', r'\\[ \1 \\]', text, flags=re.DOTALL)
    
    # 2. Replace single dollars $...$ with \(...\)
    # We use a non-greedy match for content inside dollars.
    # Note: This is a heuristic. It might fail on edge cases like "\$100", 
    # but for AI-generated math questions it's usually sufficient.
    text = re.sub(r'\$(.*?)\$', r'\\( \1 \\)', text, flags=re.DOTALL)
    
    return text

MATH_FORMATTING_RULES = r"""
    MATHEMATICAL NOTATION RULES - YOU MUST OBEY FOR ALL TOPICS: 
    1.  **Scope:** ALL mathematical objects—variables, functions, equations, inequalities, expressions—MUST be placed within LaTeX delimiters: \( \) for inline, \[ \] for display. 
    2.  **Correct Examples:** You must write: 
        *   "Consider the function \(f(x) = x^2 - 3\)."  [NOT `f(x) = x^2 - 3`] 
        *   "Solve \(x^2 + 2x - 8 = 0\)."                [NOT `x^2 + 2x - 8 = 0`] 
        *   "Find the derivative \(\frac{d}{dx}(e^{2x})\)." [NOT `d/dx(e^(2x))`] 
    3.  **Forbidden Informal Notation:** NEVER output plain text containing: 
        *   `x^2` or `x**2` outside of LaTeX. 
        *   `f(x)=` without delimiters. 
        *   `a/b` for fractions (use `\frac{a}{b}`). 
        *   `*` for multiplication (use `\cdot` if needed). 
    4.  **Goal:** The output should read like a formal VCE textbook or exam paper. 
    5.  Your entire response must be valid JSON.
"""

def sanitize_latex(text):
    """
    Sanitizes LaTeX input by fixing common AI formatting errors.
    """
    if not text:
        return ""
    import re
    
    # Rule 0: Fix Python-style powers (x**2 -> x^2)
    text = text.replace("**", "^")

    # Rule 1: Fix unescaped functions. Ensure 'sin', 'cos', 'log', 'frac' etc. have a backslash.
    text = re.sub(r'(?<!\\)\b(sin|cos|tan|log|ln|frac|sqrt|pi|theta)(?![a-zA-Z])', r'\\\1', text)
    
    # Rule 2: Auto-convert standalone x^2 to \(x^2\) (simple heuristic)
    # Matches single letter, caret, digits, not preceded by backslash or letter
    text = re.sub(r'(?<!\\|\w)([a-zA-Z]\^\d+)', r'\(\1\)', text)

    # Rule 3: Enforce delimiters. Replace standalone f(x)=... with \(f(x)=...\).
    # Heuristic: Capture "f(x)=stuff" until punctuation or newline
    text = re.sub(r'(?<!\\)(\b[a-zA-Z]\([a-zA-Z0-9]+\)\s*=[^.,\n$]+)', r'\(\1\)', text)
    
    return text

def get_backup_question(topic, exam_type):
    """
    Returns a reliable backup question from the hardcoded list.
    Guarantees a return value by falling back to a universal default.
    """
    # 1. Filter candidates
    candidates = [q for q in CURATED_QUESTION_BANK if q.get('topic') == topic and q.get('exam_type') == exam_type]
    
    # 2. Relax constraints if needed
    if not candidates and topic != 'All':
         # Relax exam type
         candidates = [q for q in CURATED_QUESTION_BANK if q.get('topic') == topic]
    
    if not candidates:
         # Fallback to any
         candidates = CURATED_QUESTION_BANK
    
    # 3. Final Check & Universal Default
    if not candidates:
        logging.error("ERROR: Backup bank empty. Using universal default.")
        return UNIVERSAL_DEFAULT_QUESTION.copy()

    # Return a copy to avoid mutation issues
    return random.choice(candidates).copy()

def validate_latex(text):
    """
    Validates that the text contains proper LaTeX delimiters for math.
    Returns True if valid, False if it likely contains unformatted math.
    """
    if not text:
        return True # Empty is valid in a trivial sense, or handled elsewhere
        
    # If text contains math-y symbols but no LaTeX delimiters, it's suspect.
    # Common math symbols: =, +, *, /, ^, _, \, <, >
    # We look for at least one pair of delimiters if these symbols are present.
    
    has_delimiters = r'\(' in text or r'\[' in text
    
    # If we have delimiters, we assume it's okay (we can't parse perfectly).
    if has_delimiters:
        return True
        
    # If no delimiters, check for suspicious math patterns
    # e.g. "x=3", "f(x)", "3x^2"
    suspicious_pattern = r'[a-zA-Z]\(x\)|[0-9]+[\+\-\=\^][0-9a-z]+|\b[xy]=\d+'
    if re.search(suspicious_pattern, text):
        return False
        
    return True

def validate_and_correct_latex(text):
    """
    Fixes malformed LaTeX (e.g. missing operands) and double-escaped backslashes.
    Returns None if the LaTeX structure is fundamentally broken (mismatched delimiters).
    """
    if not text:
        return text

    import re

    # FIX 1: Add a placeholder before a lone '\cdot'
    # Turns '( \cdot e^{3x})' into '\(k \cdot e^{3x}\)' 
    text = re.sub(r'\(\s*\\cdot', r'\(k \\cdot', text)

    # FIX 2: Ensure all '\(' and '\)' have single backslashes.
    # Replace any occurrence of '\\(' with '\(' and '\\)' with '\)' 
    text = re.sub(r'\\\\\(', r'\(', text)
    text = re.sub(r'\\\\\)', r'\)', text)
    text = re.sub(r'\\\\\[', r'\[', text)
    text = re.sub(r'\\\\\]', r'\]', text)

    # FIX 3: Final sanity check - the text must be valid LaTeX.
    # Count that '(' and ')' are paired when preceded by a backslash.
    # Note: We are counting \( and \) specifically.
    if text.count(r'\(') != text.count(r'\)'):
        # This LaTeX is broken. Return None to trigger a fallback.
        return None
    return text

def fix_latex_corruption(raw_text):
    """
    Fixes specific LaTeX corruption issues like form feed characters and nested delimiters.
    """
    if not raw_text:
        return None
        
    # 1. Fix the Form Feed (\x0crac -> \frac)
    if '\x0crac' in raw_text:
        raw_text = raw_text.replace('\x0crac', r'\frac')
        
    # 2. Enforce Consistent Delimiters (Remove Nested Parentheses)
    # Fix patterns like '(\( ... ))' or '(\( ... )' to '\( ... \)'
    import re
    raw_text = re.sub(r'\(\s*\\\(', r'\\(', raw_text)   # Remove '(' before '\('
    raw_text = re.sub(r'\\\)\s*\)', r'\\)', raw_text)   # Remove ')' after '\)'
    
    # 3. Final Validation
    # Ensure the string now has matching \( and \) counts.
    if raw_text.count(r'\(') != raw_text.count(r'\)'):
        return None
        
    return raw_text

def generate_with_retry(topic, exam_type, max_attempts=2, difficulty="medium"):
    """
    Layer 1: AI Generation with Retry.
    Generates a new, polished question using AI + VCAA source material.
    Replaces the old extraction logic with an infinite generation engine.
    """
    if not VCAA_AVAILABLE or not client:
        return None

    # 1. Fetches Source Material
    try:
        # Search for relevant content
        query = f"{topic} {exam_type} Mathematical Methods"
        results = vce_db.search(query, k=5)
    except Exception as e:
        logging.error(f"VCAA DB Search Error: {e}")
        return None

    if not results:
        return None

    # 2. Select Best Chunk
    selected_chunk_text = ""
    # Prefer chunks that look like they have math content
    for res in results:
        raw_text = ""
        if isinstance(res, tuple):
            item = res[0]
            if hasattr(item, 'page_content'):
                raw_text = item.page_content
            else:
                raw_text = str(item)
        elif hasattr(res, 'page_content'):
            raw_text = res.page_content
        else:
            raw_text = str(res)
            
        # Basic check for math context
        if len(raw_text) > 40 and any(c in raw_text for c in ['=', 'Find', 'Solve', 'Calculate', 'Given', 'Let']):
            selected_chunk_text = raw_text
            break
            
    if not selected_chunk_text:
        # Fallback to first result if no "perfect" math chunk found
        res = results[0]
        if isinstance(res, tuple):
            item = res[0]
            selected_chunk_text = item.page_content if hasattr(item, 'page_content') else str(item)
        else:
            selected_chunk_text = res.page_content if hasattr(res, 'page_content') else str(res)

    # 3. AI Generation Prompt
    # Constraint for VCE Study Design
    vce_constraint = (
        "CRITICAL CONSTRAINT: You are writing for the VCE Mathematical Methods (2023-2027) study design. "
        "You MUST ONLY use concepts, techniques, and terminology from this syllabus. "
        "You MUST NEVER reference or use: integration by parts, partial fractions, complex numbers, "
        "or any topic not explicitly listed in the official VCAA study design.\n"
        "Key areas: differentiation, integration (polynomials, exp, trig), probability (discrete/continuous RVs, normal, binomial), "
        "functions (transformations, polynomials, exp, log, circ), algebra."
    )

    prompt = f"""
    ROLE: You are a VCE Mathematical Methods exam writer.
    TASK: Using the provided VCAA exam snippet ONLY as inspiration, generate ONE new, complete, and self-contained practice question.
    {vce_constraint}
    TOPIC: {topic}
    EXAM TYPE: {exam_type}
    DIFFICULTY: {difficulty}
    
    {MATH_FORMATTING_RULES}
    
    REQUIREMENTS:
    1.  The question must be clear, solvable, and include ALL necessary context and definitions.
    2.  Do NOT copy the snippet verbatim. Create a new variant.
    3.  Output your response in this exact JSON format:
    {{
        "question_text": "The full question text here, with LaTeX.",
        "correct_answer": "The LaTeX-formatted answer (e.g., \(x=3\)).",
        "marks": 2,
        "question_type": "short" // or "mcq"
    }}
    IMPORTANT: When writing LaTeX in the JSON string, you MUST double-escape backslashes (e.g., use "\\\\frac" for \\frac, "\\\\[" for \\[).
    VCAA SNIPPET FOR INSPIRATION: {selected_chunk_text}
    """

    try:
        # Retry loop for validation
        for attempt in range(max_attempts):
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=600
            )
            
            response_content = completion.choices[0].message.content
            data = json.loads(response_content)
            
            # 4. Process AI Response & Validate
            if "question_text" not in data or "correct_answer" not in data:
                logging.error("AI response missing required fields")
                continue # Retry

            # Step 1: Strict Sane Check (User Requirement)
            # If the AI output is fundamentally broken, do not retry—fail fast to the backup.
            if not is_question_sane(data["question_text"]):
                logging.warning(f"AI Output failed Sane Check (Attempt {attempt+1}). IMMEDIATELY using backup.")
                return get_backup_question(topic, exam_type)

            # Fix LaTeX Corruption
            cleaned_q = fix_latex_corruption(data["question_text"])
            if cleaned_q is None:
                 logging.warning(f"AI Output failed Corruption Check (Attempt {attempt+1}).")
                 if attempt < max_attempts - 1:
                     continue
                 return None # Trigger fallback

            q_text = cleaned_q
            a_text = data["correct_answer"]
            
            # Stage 2: Post-Processing Sanitization
            q_text = sanitize_latex(q_text)
            a_text = sanitize_latex(a_text)
            
            # Stage 2b: Correction & Structural Validation
            q_text = validate_and_correct_latex(q_text)
            a_text = validate_and_correct_latex(a_text)
            
            if q_text is None or a_text is None:
                logging.warning(f"AI Output failed Structural LaTeX validation (Attempt {attempt+1}). Regenerating...")
                if attempt < max_attempts - 1:
                     prompt += "\n\nSYSTEM NOTE: Your previous response had mismatched LaTeX delimiters. Please check your \\( and \\) pairs."
                     continue
                else:
                     return None # Let the caller handle fallback

            # Stage 3: Final Validation & Fallback
            if not validate_latex(q_text) or not validate_latex(a_text):
                logging.warning(f"AI Output failed LaTeX validation (Attempt {attempt+1}). Regenerating...")
                if attempt < max_attempts - 1:
                     prompt += "\n\n**IMPORTANT:** Ensure the LaTeX is syntactically perfect. Every `\\(` must be followed by a valid variable, number, or function, not an operator like `+` or `\\cdot` alone."
                     continue
                else:
                    return None # Let the caller handle fallback
                
            generated_id = hashlib.md5(q_text.encode('utf-8')).hexdigest()
            
            # Post-process to fix any stray dollar signs
            final_text = normalize_latex_delimiters(q_text)
            final_answer = normalize_latex_delimiters(a_text)
            
            # Verification Log
            logging.info(f"Final Question Text Repr: {repr(final_text)}")
            
            # 5. Return Question Dictionary
            return {
                "id": generated_id,
                "text": final_text,
                "type": data.get("question_type", "short"),
                "correct_answer": final_answer,
                "marks": data.get("marks", 2),
                "topic": topic,
                "exam_type": exam_type,
                "rubric": "Mark according to VCAA standards."
            }
        
        return None # Failed after retries

    except Exception as e:
        logging.error(f"AI Generation Error: {e}")
        return None

def get_next_question(topic, exam_type, used_ids):
    """
    Layer 3: Intelligent Fallback Logic.
    Decides the next question source based on availability.
    Returns: (question_obj, is_fallback)
    """
    # Step 1: Try AI with retry
    ai_question = generate_with_retry(topic, exam_type)
    if ai_question and ai_question['id'] not in used_ids:
        return ai_question, False

    # Step 2: Filter CURATED_QUESTION_BANK for topic & exam_type
    eligible = [q for q in CURATED_QUESTION_BANK 
                if q.get('topic') == topic and q.get('exam_type') == exam_type 
                and q['id'] not in used_ids and str(q['id']) not in used_ids]
    if eligible:
        return random.choice(eligible), True

    # Step 3: Expand to same exam_type, different topic
    expanded = [q for q in CURATED_QUESTION_BANK 
                if q.get('exam_type') == exam_type 
                and q['id'] not in used_ids and str(q['id']) not in used_ids]
    if expanded:
        # Pass a flag to show a UI note: "Expanding to related topics."
        return random.choice(expanded), True

    # Step 4: Ultimate fallback (should rarely hit)
    # Just grab any valid question, even if topic/type mismatch, prioritizing unused
    final_pool = [q for q in CURATED_QUESTION_BANK if q['id'] not in used_ids and str(q['id']) not in used_ids]
    if final_pool:
         return random.choice(final_pool), True
         
    # If absolutely everything is used, recycle a random one
    if CURATED_QUESTION_BANK:
         return random.choice(CURATED_QUESTION_BANK), True
         
    return None, False

def _find_available_port(start_port=5000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return start_port

@app.route("/methods-setup", methods=["GET", "POST"])
def methods_setup():
    if request.method == "POST":
        topic = request.form.get("topic")
        exam_type = request.form.get("exam_type")
        timed_mode = request.form.get("timed") == "on"
        try:
            total_questions = int(request.form.get("total_questions", 5))
        except ValueError:
            total_questions = 5
        
        # Validation
        if not topic or not exam_type:
             return render_template("methods_setup.html", error="Please select both Topic and Exam Type.")

        # Session ID is already handled by middleware
 
        # Set flat keys as requested
        session['methods_topic'] = topic
        session['methods_exam_type'] = exam_type
        session['methods_timed'] = timed_mode
        session['total_questions'] = total_questions

        session['methods_session'] = {
            "questions_asked": [],
            "correctly_answered": [],
            "timed_on": timed_mode,
            "timer_expires_at": None,
            "topic": topic,
            "exam_type": exam_type,
            "total_questions": total_questions,
            "config_set": True,
            "fallback_mode": False
        }
        return redirect(url_for('methods_practice'))
    
    return render_template("methods_setup.html")

@app.route("/methods-exit")
def methods_exit():
    session.pop('methods_session', None)
    session.pop('methods_topic', None)
    session.pop('methods_exam_type', None)
    session.pop('methods_timed', None)
    session.pop('total_questions', None)
    return redirect(url_for('methods_setup'))

@app.route("/methods-practice", methods=["GET", "POST"])
def methods_practice():
    # Ensure session is configured
    if not session.get('methods_exam_type'):
        return redirect(url_for('methods_setup'))
    
    # Ensure state dict exists
    if 'methods_session' not in session:
        return redirect(url_for('methods_setup'))
    
    sess = session['methods_session']

    action = request.args.get('action')
    
    if action == 'retry':
        sess['last_answer'] = ""
        sess['feedback'] = None
        sess['attempts'] = sess.get('attempts', 0) + 1
        # Reset timer if timed mode
        if sess.get('timed_on'):
            q_id = sess.get('current_q_id')
            # Try to get from session data first, then backup list
            question = sess.get('current_question_data')
            if not question or question.get('id') != q_id:
                question = next((q for q in CURATED_QUESTION_BANK if q['id'] == q_id), None)
            
            limit = (question.get('marks', 1) * 90) if question else 300
            sess['timer_expires_at'] = time.time() + limit
        session.modified = True
        return redirect(url_for('methods_practice'))

    # Load new or initial question in sequential order
    if action == 'next' or 'current_q_id' not in sess:
        # Check if quiz limit reached
        asked_count = len(sess.get('questions_asked', []))
        total_q = sess.get('total_questions', 5)
        
        if asked_count >= total_q:
            # Quiz Complete
            return render_template("methods_practice.html", quiz_complete=True, score=len(sess.get('correctly_answered', [])), total=total_q)

        next_q, is_fallback = get_next_question(sess.get('topic'), sess.get('exam_type'), sess.get('questions_asked', []))
        
        if next_q is None:
             # Check if this was the very first attempt (no questions asked yet)
             if asked_count == 0:
                 return render_template("methods_setup.html", error=f"No questions found for Topic: {sess.get('topic')} ({sess.get('exam_type')}). Please try another combination.")

             # No more valid questions available (or exhausted bank)
             return render_template("methods_practice.html", quiz_complete=True, score=len(sess.get('correctly_answered', [])), total=total_q)

        sess['current_q_id'] = next_q['id']
        sess['current_question_data'] = next_q
        sess['fallback_mode'] = is_fallback
        sess['start_time'] = time.time()
        sess['attempts'] = 0
        sess['feedback'] = None
        sess['last_answer'] = ""
        
        if sess.get('timed_on'):
            limit = (next_q.get('marks', 1) * 90) if next_q else 300
            sess['timer_expires_at'] = time.time() + limit
        else:
            sess['timer_expires_at'] = None
            
        session.modified = True
        return redirect(url_for('methods_practice'))

    # Retrieve current question object
    q_id = sess.get('current_q_id')
    question = sess.get('current_question_data')
    if not question or question.get('id') != q_id:
        question = next((q for q in CURATED_QUESTION_BANK if q['id'] == q_id), None)
    
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
            return render_template("methods_practice.html", question=question, speed_warning=True, user_answer=user_answer, feedback=None, attempts=sess.get('attempts', 0), current_q_number=len(sess.get('questions_asked', [])) + 1, total_questions=sess.get('total_questions', 5))

        # Compute correctness locally
        is_correct = False
        if question.get('type') == 'mcq':
            is_correct = (choice.upper() == question.get('correct_answer'))
        else:
            is_correct = _is_equal(user_answer, question.get('correct_answer', ''))

        computed_feedback = {
            "mark": (question.get('marks', 1) if is_correct else 0),
            "max_marks": question.get('marks', 1),
            "is_correct": is_correct,
            "feedback": ("Well done." if is_correct else "Revise the method and try again.")
        }

        # Optional AI feedback enrichment
        if client:
            try:
                # Determine correct/student values for prompt clarity
                correct_val = question.get('correct_answer')
                student_val = choice if question.get('type') == 'mcq' else user_answer

                # Constraint for VCE Study Design
                vce_constraint = (
                    "CRITICAL CONSTRAINT: You are writing for the VCE Mathematical Methods (2023-2027) study design. "
                    "You MUST ONLY use concepts, techniques, and terminology from this syllabus. "
                    "You MUST NEVER reference or use: integration by parts, partial fractions, complex numbers, "
                    "or any topic not explicitly listed in the official VCAA study design."
                )

                prompt = (
                    "You are a VCAA Mathematical Methods assessor. Provide brief feedback only.\n"
                    f"{vce_constraint}\n"
                    f"Question (LaTeX): {question.get('text')}\n"
                    f"Correct Answer: {correct_val}\n"
                    f"Student Answer: {student_val}\n"
                    f"Rubric: {question.get('rubric')}\n"
                    f"Marks Available: {question.get('marks', 1)}\n"
                    "Note: If correct, say 'Correct. [Reasoning]'. If incorrect, say 'Incorrect. [Hint/Reasoning]'. Explicitly mention marks awarded (e.g. 1/2).\n"
                    f"{MATH_FORMATTING_RULES.replace('5.  Your entire response must be valid JSON.', '')}"
                )
                
                chat = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_MODEL,
                    temperature=0.2,
                    max_tokens=500
                )
                ai_text = (chat.choices[0].message.content or "").strip()
                
                # Apply LaTeX validation/correction
                corrected_text = validate_and_correct_latex(ai_text)
                if corrected_text:
                    ai_text = corrected_text
                
                # Check for truncation (completeness check)
                # If it doesn't end with typical punctuation, it might be truncated.
                if ai_text and not ai_text.strip().endswith(('.', '!', '?', ']', ')', '}')):
                     # Retry with nudge
                    prompt += "\n\nSYSTEM NOTE: Your previous response was truncated. Please provide a complete, concise sentence."
                    chat = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=GROQ_MODEL,
                        temperature=0.2,
                        max_tokens=500
                    )
                    ai_text = (chat.choices[0].message.content or "").strip()
                    
                    # Apply LaTeX validation/correction again
                    corrected_text = validate_and_correct_latex(ai_text)
                    if corrected_text:
                        ai_text = corrected_text

                if ai_text:
                    computed_feedback["feedback"] = ai_text
            except Exception:
                pass

        sess['feedback'] = computed_feedback

        # Log attempt to DB
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO question_attempts 
                (session_id, question_id, topic, exam_type, correct, attempt_number, time_spent_seconds, user_answer, feedback, question_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.get('session_id'),
                question['id'],
                question.get('topic', 'Unknown'),
                question.get('exam_type', 'Unknown'),
                is_correct,
                sess.get('attempts', 0) + 1,
                time_taken,
                user_answer,
                computed_feedback.get('feedback', ''),
                question.get('text', '')
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Failed to log attempt: {e}")

        sess['attempts'] = sess.get('attempts', 0) + 1
        sess['last_answer'] = user_answer
        if is_correct:
            # Track correctly answered to avoid repeats in this session
            corr = set(sess.get('correctly_answered', []))
            corr.add(question['id'])
            sess['correctly_answered'] = list(corr)
        
        # Add to questions_asked if not already there (should be there only once)
        asked = list(sess.get('questions_asked', []))
        if question['id'] not in asked:
            asked.append(question['id'])
            sess['questions_asked'] = asked
            
        # Stop timer on submission
        sess['timer_expires_at'] = None
        session.modified = True
        
    show_try_again = bool(sess.get('feedback') and not sess['feedback'].get('is_correct') and sess.get('attempts', 0) < 3)
    
    asked_ids = sess.get('questions_asked', [])
    current_q_number = len(asked_ids)
    if q_id not in asked_ids:
        # It's a new question being viewed, so it's the (len + 1)th question
        current_q_number += 1
        
    return render_template(
        "methods_practice.html",
        question=question,
        feedback=sess.get('feedback'),
        attempts=sess.get('attempts', 0),
        user_answer=user_answer,
        speed_warning=speed_warning,
        timed_on=sess.get('timed_on', False),
        time_remaining=time_remaining,
        show_try_again=show_try_again,
        current_q_number=current_q_number,
        total_questions=sess.get('total_questions', 5),
        fallback_warning=sess.get('fallback_mode', False)
    )

@app.route("/my-progress")
def my_progress():
    # Session is handled by middleware
    
    conn = get_db()
    
    # Check if table exists (in case init_db wasn't run or DB was deleted)
    try:
        # Chart Data: Accuracy over time
        chart_query = '''
            SELECT DATE(created_at) as day, AVG(correct)*100 as accuracy 
            FROM question_attempts 
            WHERE session_id=? 
            GROUP BY day 
            ORDER BY day ASC
        '''
        chart_data = conn.execute(chart_query, (session.get('session_id'),)).fetchall()
        
        # Topic Breakdown
        topic_query = '''
            SELECT topic, AVG(correct)*100 as avg_score 
            FROM question_attempts 
            WHERE session_id=? 
            GROUP BY topic
        '''
        topic_data = conn.execute(topic_query, (session.get('session_id'),)).fetchall()
    except sqlite3.OperationalError:
        # Table might not exist yet
        chart_data = []
        topic_data = []
        
    conn.close()
    
    # Format for Chart.js
    dates = [row['day'] for row in chart_data]
    accuracies = [round(row['accuracy'], 1) for row in chart_data]
    
    topics = []
    for row in topic_data:
        topics.append({
            'name': row['topic'],
            'avg_score': round(row['avg_score'], 1),
            'is_weak': row['avg_score'] < 60
        })
        
    return render_template("my_progress.html", dates=dates, accuracies=accuracies, topics=topics)

@app.route("/my-questions")
def my_questions():
    # Session handled by middleware
        
    conn = get_db()
    try:
        query = '''
            SELECT * FROM question_attempts 
            WHERE session_id=? 
            ORDER BY created_at DESC
        '''
        attempts = conn.execute(query, (session.get('session_id'),)).fetchall()
    except sqlite3.OperationalError:
        attempts = []
        
    conn.close()
    
    return render_template("my_questions.html", attempts=attempts)

@app.route('/admin')
def admin_dashboard():
    # 1. Access Control
    admin_key = os.environ.get('ADMIN_KEY')
    request_key = request.args.get('key')
    
    # If key is missing/incorrect, return 401
    # Correct validation: check if keys are missing OR don't match 
    if not admin_key or not request_key or request_key != admin_key: 
        return "Unauthorized", 401

    headline = {
        'total_sessions': 0,
        'questions_24h': 0,
        'active_users_7d': 0
    }
    topic_performance = []
    recent_activity = []
    error = None

    conn = get_db()
    try:
        # Check if tables exist (sanity check)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('anonymous_sessions', 'question_attempts')").fetchall()
        if len(tables) < 2:
             error = "Required database tables not found."
        else:
            # 2. Headline Metrics
            # Total Quiz Sessions
            res = conn.execute("SELECT COUNT(DISTINCT session_id) FROM anonymous_sessions").fetchone()
            if res: headline['total_sessions'] = res[0]

            # Questions Answered (Last 24h)
            res = conn.execute("SELECT COUNT(*) FROM question_attempts WHERE created_at >= datetime('now', '-1 day')").fetchone()
            if res: headline['questions_24h'] = res[0]

            # Active Users (7d)
            res = conn.execute("SELECT COUNT(DISTINCT session_id) FROM anonymous_sessions WHERE last_activity >= datetime('now', '-7 days')").fetchone()
            if res: headline['active_users_7d'] = res[0]

            # 3. Topic Performance
            # Topic, Total Attempts, Average Accuracy (%)
            tp_query = '''
                SELECT topic, COUNT(*) as total_attempts, ROUND(AVG(correct) * 100, 1) as avg_accuracy 
                FROM question_attempts 
                GROUP BY topic 
                ORDER BY total_attempts DESC
            '''
            topic_performance = [dict(row) for row in conn.execute(tp_query).fetchall()]

            # 4. Recent Activity Log
            # Last 50 question_attempts
            ra_query = '''
                SELECT created_at, session_id, topic, exam_type, attempt_number, correct, time_spent_seconds 
                FROM question_attempts 
                ORDER BY created_at DESC 
                LIMIT 50
            '''
            rows = conn.execute(ra_query).fetchall()
            recent_activity = []
            for row in rows:
                d = dict(row)
                # Format session_id to first 6 chars for readability
                if d['session_id']:
                    d['session_id'] = d['session_id'][:6]
                recent_activity.append(d)

    except Exception as e:
        error = f"Database error: {str(e)}"
        logging.error(f"Admin dashboard error: {e}")
    finally:
        conn.close()

    return render_template('admin.html', headline=headline, topic_performance=topic_performance, recent_activity=recent_activity, error=error)

@app.route("/start-topic-quiz")
def start_topic_quiz():
    topic = request.args.get('topic')
    if not topic:
        return redirect(url_for('my_progress'))
        
    # Session is handled by middleware
        
    # Initialize a new practice session for this topic
    session['methods_topic'] = topic
    session['methods_exam_type'] = 'tech_active' # Default
    session['methods_timed'] = False
    session['total_questions'] = 5
    
    session['methods_session'] = {
        "questions_asked": [],
        "correctly_answered": [],
        "timed_on": False,
        "timer_expires_at": None,
        "topic": topic,
        "exam_type": 'tech_active',
        "total_questions": 5,
        "config_set": True,
        "fallback_mode": False
    }
    return redirect(url_for('methods_practice'))

@app.route("/review-question/<question_id>", methods=["GET", "POST"])
def review_question(question_id):
    # Find question
    question = next((q for q in CURATED_QUESTION_BANK if str(q['id']) == str(question_id)), None)
    if not question:
        question = UNIVERSAL_DEFAULT_QUESTION.copy()
        question['id'] = question_id 
    
    feedback = None
    user_answer = ""
    
    if request.method == "POST":
        user_answer = request.form.get('answer', '')
        choice = request.form.get('choice', '')
        
        # Check correctness
        is_correct = False
        if question.get('type') == 'mcq':
            is_correct = (choice.upper() == question.get('correct_answer'))
        else:
            is_correct = _is_equal(user_answer, question.get('correct_answer', ''))
            
        # Simplified Feedback for Review
        computed_feedback = {
            "mark": (question.get('marks', 1) if is_correct else 0),
            "max_marks": question.get('marks', 1),
            "is_correct": is_correct,
            "feedback": ("Correct. Well done." if is_correct else "Incorrect. Review the method.")
        }
        
        # Try to use AI if available
        if client:
             try:
                correct_val = question.get('correct_answer')
                student_val = choice if question.get('type') == 'mcq' else user_answer
                prompt = (
                    "You are a VCAA Mathematical Methods assessor. Provide brief feedback only.\n"
                    f"Question: {question.get('text')}\n"
                    f"Correct Answer: {correct_val}\n"
                    f"Student Answer: {student_val}\n"
                    f"Rubric: {question.get('rubric')}\n"
                    "Provide 1 sentence of constructive feedback."
                )
                chat = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_MODEL,
                    temperature=0.2,
                    max_tokens=200
                )
                ai_text = (chat.choices[0].message.content or "").strip()
                if ai_text:
                    computed_feedback["feedback"] = ai_text
             except:
                pass

        # Log attempt
        try:
            conn = get_db()
            cursor = conn.cursor()
            # Get max attempt number
            prev = cursor.execute("SELECT MAX(attempt_number) FROM question_attempts WHERE session_id=? AND question_id=?", (session.get('session_id'), question_id)).fetchone()
            next_attempt = (prev[0] if prev[0] else 0) + 1
            
            cursor.execute('''
                INSERT INTO question_attempts 
                (session_id, question_id, topic, exam_type, correct, attempt_number, time_spent_seconds, user_answer, feedback, question_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.get('session_id'),
                question_id,
                question.get('topic', 'Unknown'),
                question.get('exam_type', 'Unknown'),
                is_correct,
                next_attempt,
                0,
                user_answer,
                computed_feedback.get('feedback', ''),
                question.get('text', '')
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(e)
            
        feedback = computed_feedback
        
    return render_template("review_question.html", question=question, feedback=feedback, user_answer=user_answer)


if __name__ == "__main__":
    env_port = os.getenv("PORT")
    port = int(env_port) if env_port else _find_available_port()
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
