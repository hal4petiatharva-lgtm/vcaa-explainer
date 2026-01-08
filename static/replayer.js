const canvas = document.getElementById('replayCanvas')
const ctx = canvas ? canvas.getContext('2d') : null
const speedSel = document.getElementById('replaySpeed')
const btnPlay = document.getElementById('btnPlay')
const btnPause = document.getElementById('btnPause')
let playing = false
let speed = 1
let timer = null
function drawStroke(s, scale) {
  if (!ctx) return
  ctx.beginPath()
  for (let i = 0; i < s.x.length; i++) {
    const x = s.x[i]
    const y = s.y[i]
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.strokeStyle = '#111'
  ctx.lineWidth = 2
  ctx.stroke()
}
function clearCanvas() {
  if (!ctx) return
  ctx.clearRect(0,0,canvas.width,canvas.height)
}
function animateStrokes(strokes) {
  if (!Array.isArray(strokes)) return
  clearCanvas()
  let idx = 0
  playing = true
  function step() {
    if (!playing) return
    if (idx >= strokes.length) { playing = false; return }
    drawStroke(strokes[idx], speed)
    idx++
    timer = setTimeout(step, 200 / speed)
  }
  step()
}
if (speedSel) {
  speedSel.addEventListener('change', () => {
    speed = parseFloat(speedSel.value || '1')
  })
}
if (btnPlay) {
  btnPlay.addEventListener('click', () => {
    try {
      if (timer) clearTimeout(timer)
      animateStrokes(window.STROKES || [])
    } catch (e) {}
  })
}
if (btnPause) {
  btnPause.addEventListener('click', () => {
    playing = false
    if (timer) clearTimeout(timer)
  })
}
