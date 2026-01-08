const el = document.getElementById('math-canvas')
const preview = document.getElementById('latex-preview')
const confMsg = document.getElementById('confidence-msg')
const fallbackArea = document.getElementById('fallback-area')
const btnAnalyze = document.getElementById('btn-analyze')
let lastLatex = ''
let lastConfidence = 0
let strokesBuffer = []
function captureStrokes(evt) {
  if (evt && evt.detail && evt.detail.strokes) return evt.detail.strokes
  return strokesBuffer
}
function showFallback() {
  if (fallbackArea) fallbackArea.style.display = 'block'
}
function updatePreview(latex) {
  preview.innerHTML = `$$${latex}$$`
  if (window.MathJax && window.MathJax.typeset) window.MathJax.typeset()
}
function sendToBackend(payload) {
  return fetch('/api/myscript', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  }).catch(() => {
    try {
      const queue = JSON.parse(localStorage.getItem('pendingStrokes') || '[]')
      queue.push(payload)
      localStorage.setItem('pendingStrokes', JSON.stringify(queue))
    } catch (e) {}
  })
}
if (el) {
  el.addEventListener('exported', (event) => {
    const latex = (event.detail && event.detail.exports && event.detail.exports['application/x-latex']) || ''
    const confidence = event.detail && typeof event.detail.confidence === 'number' ? event.detail.confidence : 0
    const strokes = captureStrokes(event)
    lastLatex = latex
    lastConfidence = confidence
    updatePreview(latex)
    confMsg.textContent = confidence < 0.7 ? 'Please write clearer' : ''
    const payload = {
      latex,
      confidence,
      strokes,
      session_id: window.PRACTICE_SESSION_ID,
      question_id: window.PRACTICE_QUESTION_ID
    }
    sendToBackend(payload)
  })
} else {
  showFallback()
}
if (btnAnalyze) {
  btnAnalyze.addEventListener('click', () => {
    const latex = lastLatex || document.getElementById('fallback-input')?.value || ''
    const payload = {
      session_id: window.PRACTICE_SESSION_ID,
      final_latex: latex,
      question_id: window.PRACTICE_QUESTION_ID
    }
    fetch('/practice/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    }).then(r => r.json()).then(res => {
      alert(`${res.feedback} | Mark: ${res.mark}`)
    }).catch(() => {
      alert('Analysis failed')
    })
  })
}
function flushQueue() {
  try {
    const queue = JSON.parse(localStorage.getItem('pendingStrokes') || '[]')
    if (!Array.isArray(queue) || queue.length === 0) return
    const next = queue.shift()
    localStorage.setItem('pendingStrokes', JSON.stringify(queue))
    sendToBackend(next)
  } catch (e) {}
}
window.addEventListener('online', flushQueue)
function adjustCanvas() {
  if (window.innerWidth < 768) {
    el?.setAttribute('width','100%')
    el?.setAttribute('height','300')
  }
}
adjustCanvas()
