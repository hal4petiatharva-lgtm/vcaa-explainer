document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    const state = {
        sessionId: null,
        currentQuestion: null,
        totalQuestions: 0,
        currentQuestionNum: 0,
        isTimed: false,
        startTime: null,
        timerInterval: null,
        inputMode: 'handwriting' // 'handwriting' or 'typing'
    };

    // --- DOM Elements ---
    const views = {
        setup: document.getElementById('setup-view'),
        quiz: document.getElementById('quiz-view'),
        results: document.getElementById('results-view')
    };

    const myscriptElement = document.getElementById('quiz-canvas');
    const latexPreview = document.getElementById('latex-preview');
    const typingInput = document.getElementById('typing-input-field');
    
    // --- Setup View Logic ---
    window.startQuiz = async (e) => {
        if(e) e.preventDefault();
        
        const form = document.getElementById('quiz-setup-form');
        const formData = new FormData(form);
        
        const payload = {
            topic: formData.get('topic'),
            exam_type: formData.get('exam'),
            is_timed: formData.get('timed') === 'on',
            question_count: formData.get('question_count')
        };
        
        state.isTimed = payload.is_timed;

        try {
            const res = await fetch('/api/quiz/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            state.sessionId = data.session_id;
            state.totalQuestions = data.total_questions;
            
            loadQuestion(data.question, data.current_question_num);
            switchView('quiz');
            
            if (state.isTimed) startTimer();
            
        } catch (error) {
            console.error("Failed to start quiz:", error);
            alert("Could not start quiz. Please try again.");
        }
    };

    // --- Quiz View Logic ---
    function loadQuestion(question, num) {
        state.currentQuestion = question;
        state.currentQuestionNum = num;
        
        // Update UI
        document.getElementById('progress-text').textContent = `Question ${num} / ${state.totalQuestions}`;
        
        // Render MathJax in question text
        const qDisplay = document.getElementById('question-display');
        qDisplay.innerHTML = question.text;
        if (window.MathJax) MathJax.typesetPromise([qDisplay]);
        
        // Reset Inputs
        if (myscriptElement && myscriptElement.clear) myscriptElement.clear();
        latexPreview.innerHTML = '';
        typingInput.value = '';
        document.getElementById('feedback-area').style.display = 'none';
        
        // Show Check Button, Hide Next Button
        document.getElementById('btn-check').style.display = 'inline-block';
        document.getElementById('btn-next').style.display = 'none';
        
        // Reset Input Mode UI
        setInputMode(state.inputMode);
    }

    window.setInputMode = (mode) => {
        state.inputMode = mode;
        
        // Update Buttons
        document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById(`btn-mode-${mode}`).classList.add('active');
        
        // Update Visibility
        document.getElementById('handwriting-input').classList.toggle('active', mode === 'handwriting');
        document.getElementById('typing-input').classList.toggle('active', mode === 'typing');
    };

    // MyScript Integration
    let currentStrokes = null;
    let currentLatex = "";

    if (myscriptElement) {
        myscriptElement.addEventListener('exported', (event) => {
            if (event.detail && event.detail.exports) {
                currentLatex = event.detail.exports['application/x-latex'];
                currentStrokes = event.detail.exports['application/vnd.myscript.jiix'];
                
                latexPreview.innerHTML = `$$${currentLatex}$$`;
                if (window.MathJax) MathJax.typesetPromise([latexPreview]);
            }
        });
    }

    window.submitAnswer = async () => {
        const btnCheck = document.getElementById('btn-check');
        btnCheck.disabled = true;
        btnCheck.textContent = "Checking...";
        
        let answerLatex = "";
        let strokes = null;
        
        if (state.inputMode === 'handwriting') {
            answerLatex = currentLatex;
            strokes = JSON.stringify(currentStrokes);
        } else {
            answerLatex = typingInput.value;
        }

        try {
            const res = await fetch('/api/quiz/submit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: state.sessionId,
                    question_id: state.currentQuestion.id,
                    answer_type: state.inputMode,
                    strokes_json: strokes,
                    answer_latex: answerLatex
                })
            });
            const result = await res.json();
            
            // Show Feedback
            const feedbackArea = document.getElementById('feedback-area');
            feedbackArea.style.display = 'block';
            feedbackArea.className = `feedback-area ${result.correct ? 'correct' : 'incorrect'}`;
            feedbackArea.innerHTML = result.feedback;
            if (window.MathJax) MathJax.typesetPromise([feedbackArea]);
            
            // Switch Buttons
            btnCheck.style.display = 'none';
            document.getElementById('btn-next').style.display = 'inline-block';
            
        } catch (error) {
            console.error("Submission failed:", error);
            alert("Error checking answer.");
        } finally {
            btnCheck.disabled = false;
            btnCheck.textContent = "Check Answer";
        }
    };

    window.nextQuestion = async () => {
        try {
            const res = await fetch('/api/quiz/next', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ session_id: state.sessionId })
            });
            const data = await res.json();
            
            if (data.finished) {
                endQuiz();
            } else {
                loadQuestion(data.question, data.current_question_num);
            }
        } catch (error) {
            console.error("Next question failed:", error);
        }
    };

    // --- Results View Logic ---
    async function endQuiz() {
        if (state.timerInterval) clearInterval(state.timerInterval);
        
        try {
            const res = await fetch(`/api/quiz/end/${state.sessionId}`, { method: 'POST' });
            const data = await res.json();
            
            document.getElementById('final-score-num').textContent = `${data.score}/${data.total_questions}`;
            
            // Populate attempt list
            const attemptsList = document.getElementById('attempts-list');
            attemptsList.innerHTML = '';
            data.attempts.forEach((att, index) => {
                const li = document.createElement('div');
                li.className = 'stat-item';
                li.innerHTML = `
                    <span>Question ${index + 1}</span>
                    <span>${att.marks_awarded > 0 ? '✅' : '❌'}</span>
                `;
                attemptsList.appendChild(li);
            });
            
            switchView('results');
        } catch (error) {
            console.error("End quiz failed:", error);
        }
    }

    window.newQuiz = () => {
        switchView('setup');
        // Reset form or state if needed
    };

    // --- Helpers ---
    function switchView(viewName) {
        Object.values(views).forEach(el => el.classList.remove('active'));
        views[viewName].classList.add('active');
    }
    
    function startTimer() {
        let seconds = 0;
        const timerDisplay = document.getElementById('timer-text');
        state.timerInterval = setInterval(() => {
            seconds++;
            const m = Math.floor(seconds / 60).toString().padStart(2, '0');
            const s = (seconds % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `${m}:${s}`;
        }, 1000);
    }
});
