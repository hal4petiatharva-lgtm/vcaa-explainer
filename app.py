
# --- Unified Quiz Routes ---

@app.route('/math-quiz')
def math_quiz():
    return render_template('math_quiz.html')

@app.route('/api/quiz/start', methods=['POST'])
def start_unified_quiz():
    data = request.json
    topic = data.get('topic', 'Calculus')
    exam_type = data.get('exam_type', 'tech_free')
    is_timed = data.get('is_timed', False)
    question_count = int(data.get('question_count', 5))
    
    session_id = str(uuid.uuid4())
    
    conn = get_db()
    conn.execute('''
        INSERT INTO unified_quiz_sessions 
        (session_id, topic, exam_type, is_timed, question_count, status)
        VALUES (?, ?, ?, ?, ?, 'active')
    ''', (session_id, topic, exam_type, is_timed, question_count))
    conn.commit()
    conn.close()
    
    # Get first question
    question = get_next_question_logic(topic, exam_type, [])
    
    return jsonify({
        'session_id': session_id,
        'question': question,
        'total_questions': question_count,
        'current_question_num': 1
    })

@app.route('/api/quiz/submit', methods=['POST'])
def submit_unified_answer():
    data = request.json
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    answer_type = data.get('answer_type') # 'handwriting' or 'typing'
    strokes_json = data.get('strokes_json')
    answer_latex = data.get('answer_latex')
    
    # Find the question
    question = next((q for q in CURATED_QUESTION_BANK if str(q['id']) == str(question_id)), None)
    
    correct = False
    feedback = "Could not analyze answer."
    marks = 0
    
    if question:
        # Simplified checking logic
        correct_answer = question.get('correct_answer', '').replace(' ', '')
        user_answer = answer_latex.replace(' ', '') if answer_latex else ''
        
        # In a real scenario, use AI here. For now, simple string check + some leniency
        if user_answer and (user_answer in correct_answer or correct_answer in user_answer):
            correct = True
            marks = question.get('marks', 1)
            feedback = f"Correct! The answer is {question['correct_answer']}."
        else:
            feedback = f"Incorrect. The expected answer was {question['correct_answer']}."
    
    conn = get_db()
    
    # Record attempt
    conn.execute('''
        INSERT INTO unified_quiz_attempts 
        (session_id, question_id, answer_type, strokes_json, answer_latex, ai_feedback, marks_awarded)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, question_id, answer_type, strokes_json, answer_latex, feedback, marks))
    
    # Update score in session
    if marks > 0:
        conn.execute('''
            UPDATE unified_quiz_sessions 
            SET score = score + ? 
            WHERE session_id = ?
        ''', (marks, session_id))
        
    conn.commit()
    conn.close()
    
    return jsonify({
        'correct': correct,
        'feedback': feedback,
        'marks': marks
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
    
    # Get used question IDs to avoid repeats
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

@app.route('/api/quiz/end/<session_id>', methods=['POST'])
def end_unified_quiz(session_id):
    conn = get_db()
    session_row = conn.execute('SELECT * FROM unified_quiz_sessions WHERE session_id = ?', (session_id,)).fetchone()
    conn.execute("UPDATE unified_quiz_sessions SET status = 'completed' WHERE session_id = ?", (session_id,))
    
    attempts = conn.execute('SELECT * FROM unified_quiz_attempts WHERE session_id = ?', (session_id,)).fetchall()
    conn.commit()
    conn.close()
    
    if not session_row:
        return jsonify({'error': 'Session not found'}), 404
        
    # Analyze weak areas (mock logic)
    weak_areas = []
    # Real logic would aggregate by sub-topic from attempts
    
    return jsonify({
        'score': session_row['score'],
        'total_questions': session_row['question_count'],
        'attempts': [dict(a) for a in attempts],
        'weak_areas': weak_areas
    })

def get_next_question_logic(topic, exam_type, used_ids):
    # Reuse the logic from get_backup_question or get_next_question
    # This is a helper for the API routes
    
    # Filter candidates
    candidates = [q for q in CURATED_QUESTION_BANK 
                  if q.get('topic') == topic and q.get('exam_type') == exam_type
                  and str(q['id']) not in used_ids]
    
    if not candidates:
        # Relax exam type
        candidates = [q for q in CURATED_QUESTION_BANK 
                      if q.get('topic') == topic
                      and str(q['id']) not in used_ids]
                      
    if not candidates:
         # Fallback to any unused
         candidates = [q for q in CURATED_QUESTION_BANK if str(q['id']) not in used_ids]
         
    if not candidates:
        # If all used, just pick random
        candidates = CURATED_QUESTION_BANK
        
    if candidates:
        return random.choice(candidates)
    return None
