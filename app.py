from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
import sqlite3

# ===== VCAADECODE HANDWRITING SYSTEM UPDATE ===== 
# TIMESTAMP: 2025-12-23_15:00:00_UTC 
# UPDATE_ID: HW_QUIZ_78901 
# PURPOSE: Add /math-quiz route for handwriting practice 
# ================================================ 
import os
import random
import uuid
import json
import time

app = Flask(__name__)
app.secret_key = 'dev'
DATABASE = 'vce_progress.db'

# --- Database Helpers ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        # Unified Quiz Tables
        db.execute('''
            CREATE TABLE IF NOT EXISTS unified_quiz_sessions (
                session_id TEXT PRIMARY KEY,
                topic TEXT,
                exam_type TEXT,
                is_timed BOOLEAN,
                question_count INTEGER,
                current_question INTEGER DEFAULT 1,
                score INTEGER DEFAULT 0,
                time_started TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS unified_quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question_id TEXT,
                answer_type TEXT,
                strokes_json TEXT,
                answer_latex TEXT,
                ai_feedback TEXT,
                marks_awarded INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

# --- Mock Data ---
CURATED_QUESTION_BANK = [
    {
        'id': '1',
        'topic': 'Calculus',
        'exam_type': 'tech_free',
        'text': r'Find the derivative of \( f(x) = x^3 - 2x^2 + 5 \).',
        'correct_answer': r'3x^2 - 4x',
        'marks': 2
    },
    {
        'id': '2',
        'topic': 'Functions',
        'exam_type': 'tech_free',
        'text': r'Solve for x: \( 2x + 5 = 13 \).',
        'correct_answer': r'4',
        'marks': 1
    },
    {
        'id': '3',
        'topic': 'Probability',
        'exam_type': 'tech_active',
        'text': r'If \( X \sim \mathcal{N}(0, 1) \), find \( \Pr(X > 1) \) correct to 2 decimal places.',
        'correct_answer': r'0.16',
        'marks': 1
    },
    {
        'id': '4',
        'topic': 'Algebra',
        'exam_type': 'tech_free',
        'text': r'Simplify \( \frac{x^2 - 9}{x - 3} \).',
        'correct_answer': r'x + 3',
        'marks': 1
    },
    {
        'id': '5',
        'topic': 'Calculus',
        'exam_type': 'tech_active',
        'text': r'Evaluate \( \int_0^1 x^2 \, dx \).',
        'correct_answer': r'1/3',
        'marks': 2
    }
]

def get_next_question_logic(topic, exam_type, used_ids):
    candidates = [q for q in CURATED_QUESTION_BANK 
                  if q.get('topic') == topic and q.get('exam_type') == exam_type
                  and str(q['id']) not in used_ids]
    
    if not candidates:
        candidates = [q for q in CURATED_QUESTION_BANK 
                      if q.get('topic') == topic
                      and str(q['id']) not in used_ids]
                      
    if not candidates:
         candidates = [q for q in CURATED_QUESTION_BANK if str(q['id']) not in used_ids]
         
    if not candidates:
        candidates = CURATED_QUESTION_BANK
        
    if candidates:
        return random.choice(candidates)
    return None

# --- Routes ---

@app.route('/')
def index():
    return render_template('base.html') # Placeholder if index.html missing, or just base

@app.route('/my-progress')
def my_progress():
    return "My Progress Placeholder" # Placeholder

# ===== UNIFIED MATH QUIZ SYSTEM =====

@app.route('/math-quiz')
def math_quiz():
    """Main Math Methods handwriting quiz"""
    return render_template('math_quiz.html')

@app.route('/test-route')
def test_route():
    return "Flask is running"

@app.route('/api/quiz/start', methods=['POST'])
def start_quiz():
    """Start new quiz session with selected settings"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data received'}), 400
    
    # Create session ID
    session_id = f"quiz_{int(time.time())}"
    
    topic = data.get('topic', 'Calculus')
    exam_type = data.get('exam_type', 'tech_free')
    is_timed = data.get('is_timed', False)
    question_count = int(data.get('question_count', 5))
    
    # Store session in database
    conn = get_db()
    
    # Create table if doesn't exist (safety check)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS unified_quiz_sessions (
            session_id TEXT PRIMARY KEY,
            topic TEXT,
            exam_type TEXT,
            is_timed BOOLEAN,
            question_count INTEGER,
            current_question INTEGER DEFAULT 1,
            score INTEGER DEFAULT 0,
            time_started TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
    """)
    
    conn.execute("""
        INSERT INTO unified_quiz_sessions 
        (session_id, topic, exam_type, is_timed, question_count)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session_id, 
        topic, 
        exam_type, 
        1 if is_timed else 0, 
        question_count
    ))
    
    conn.commit()
    conn.close()
    
    # Get first question
    question = get_next_question_logic(topic, exam_type, [])
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': 'Quiz session started',
        'question': question,
        'total_questions': question_count,
        'current_question_num': 1
    })

@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz_answer():
    """Submit handwriting/typing answer for AI analysis"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No answer data'}), 400
    
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    answer_type = data.get('answer_type')
    strokes_json = data.get('strokes_json')
    answer_latex = data.get('answer_latex')
    
    # Find the question for basic validation
    question = next((q for q in CURATED_QUESTION_BANK if str(q['id']) == str(question_id)), None)
    
    correct = False
    feedback = "✓ Correct solution. You would receive full marks on a VCAA exam."
    marks = 2
    
    if question:
        correct_answer = question.get('correct_answer', '').replace(' ', '')
        user_answer = answer_latex.replace(' ', '') if answer_latex else ''
        if user_answer and (user_answer in correct_answer or correct_answer in user_answer):
            correct = True
            marks = question.get('marks', 1)
        else:
            correct = False
            marks = 0
            feedback = f"Incorrect. The expected answer was {question['correct_answer']}."

    # Record attempt
    conn = get_db()
    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS unified_quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question_id TEXT,
            answer_type TEXT,
            strokes_json TEXT,
            answer_latex TEXT,
            ai_feedback TEXT,
            marks_awarded INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute('''
        INSERT INTO unified_quiz_attempts 
        (session_id, question_id, answer_type, strokes_json, answer_latex, ai_feedback, marks_awarded)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, question_id, answer_type, strokes_json, answer_latex, feedback, marks))
    
    if marks > 0:
        conn.execute('UPDATE unified_quiz_sessions SET score = score + ? WHERE session_id = ?', (marks, session_id))
        
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'feedback': feedback,
        'mark': marks,
        'correct': correct,
        'next_question': data.get('question_number', 0) + 1
    })

@app.route('/api/quiz/next', methods=['POST'])
def next_unified_question():
    data = request.json
    session_id = data.get('session_id')
    
    conn = get_db()
    session_row = conn.execute('SELECT * FROM unified_quiz_sessions WHERE session_id = ?', (session_id,)).fetchone()
    
    if not session_row:
        return jsonify({'error': 'Session not found'}), 404
        
    current_q = session_row['current_question']
    total_q = session_row['question_count']
    
    if current_q >= total_q:
        return jsonify({'finished': True})
    
    # Increment question number
    next_q_num = current_q + 1
    conn.execute('UPDATE unified_quiz_sessions SET current_question = ? WHERE session_id = ?', (next_q_num, session_id))
    conn.commit()
    
    # Get used question IDs
    attempts = conn.execute('SELECT question_id FROM unified_quiz_attempts WHERE session_id = ?', (session_id,)).fetchall()
    used_ids = [str(a['question_id']) for a in attempts]
    
    # Get next question
    question = get_next_question_logic(session_row['topic'], session_row['exam_type'], used_ids)
    
    conn.close()
    
    return jsonify({
        'question': question,
        'current_question_num': next_q_num,
        'total_questions': total_q
    })

@app.route('/api/quiz/end', methods=['POST'])
@app.route('/api/quiz/end/<session_id>', methods=['POST']) # Support both
def end_quiz_session(session_id=None):
    """End quiz and return final results"""
    if not session_id:
        data = request.get_json()
        session_id = data.get('session_id')
        
    conn = get_db()
    session_row = conn.execute('SELECT * FROM unified_quiz_sessions WHERE session_id = ?', (session_id,)).fetchone()
    if session_row:
        conn.execute("UPDATE unified_quiz_sessions SET status = 'completed' WHERE session_id = ?", (session_id,))
        conn.commit()
    
    attempts = conn.execute('SELECT * FROM unified_quiz_attempts WHERE session_id = ?', (session_id,)).fetchall()
    conn.close()
    
    score = session_row['score'] if session_row else 0
    total = session_row['question_count'] if session_row else 0
    
    return jsonify({
        'success': True,
        'score': score,
        'total': total,
        'total_questions': total,
        'accuracy': (score/total*100) if total > 0 else 0,
        'weak_topics': ['Chain Rule', 'Integration Techniques'], # Mock data
        'session_id': session_id,
        'attempts': [dict(a) for a in attempts] # For frontend list
    })

@app.route('/debug-quiz-routes')
def debug_quiz_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        if 'quiz' in str(rule):
            routes.append(f"{rule.endpoint}: {rule.rule}")
    return "<br>".join(routes) if routes else "No quiz routes found"

# ===== END MATH QUIZ SYSTEM =====

@app.route('/debug-quiz-status')
def debug_quiz_status():
    """Debug endpoint to verify /math-quiz route exists"""
    import os
    routes = []
    for rule in app.url_map.iter_rules():
        if 'quiz' in str(rule) or 'math' in str(rule):
            routes.append(f"{rule.endpoint}: {rule.rule}")
    
    return f"""
    <h1>Quiz System Status</h1>
    <p>Routes found: {len(routes)}</p>
    <ul>
        {"".join(f'<li>{r}</li>' for r in routes)}
    </ul>
    <p>Template exists: {os.path.exists('templates/math_quiz.html')}</p>
    """

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
