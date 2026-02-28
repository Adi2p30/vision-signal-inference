const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const logEntries = document.getElementById('logEntries');
const tsDisplay = document.getElementById('tsDisplay');
const SCALE_W = 480; // downscale width for speed

// State
let autoRunning = false;
let autoRafId = null;
let lastCaptureTime = -1;
let inflight = 0;
let doneTotal = 0;
let scoreA = 0, scoreB = 0;
let lastScoreA_ts = 0, lastScoreB_ts = 0;
const allRows = [];
let fpsTimestamps = [];

// Play-by-play score -> wallclock mapping
// Key: "awayScore-homeScore", Value: wallclock ISO string (last play with that score)
let scoreToWallclock = {};

fetch('/play-by-play')
  .then(r => r.json())
  .then(plays => {
    // Build lookup: for each unique score state, store the wallclock of the
    // first play that reached that score (the scoring play itself)
    const seen = new Set();
    for (const p of plays) {
      const key = p.awayScore + '-' + p.homeScore;
      if (!seen.has(key)) {
        seen.add(key);
        scoreToWallclock[key] = p.wallclock;
      }
    }
    console.log('Play-by-play loaded:', Object.keys(scoreToWallclock).length, 'score states');
  })
  .catch(err => console.warn('Failed to load play-by-play:', err));

video.addEventListener('timeupdate', () => {
  const t = video.currentTime;
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1);
  tsDisplay.textContent = m + ':' + s.padStart(4, '0');
});

function captureFrame() {
  const scale = Math.min(1, SCALE_W / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return {
    timestamp: video.currentTime,
    base64: canvas.toDataURL('image/jpeg', 0.5).split(',')[1]
  };
}

function updateStats() {
  document.getElementById('inflightCount').textContent = inflight;
  document.getElementById('doneCount').textContent = doneTotal;
  // fps over last 10s
  const now = performance.now();
  fpsTimestamps = fpsTimestamps.filter(t => now - t < 10000);
  const fps = fpsTimestamps.length > 0 ? (fpsTimestamps.length / ((now - fpsTimestamps[0]) / 1000)).toFixed(1) : '0';
  document.getElementById('inferFps').textContent = fps;
}

// --- Auto: capture every frame, fire concurrently ---
function toggleAuto() {
  const on = document.getElementById('autoCheck').checked;
  document.getElementById('autoLabel').classList.toggle('active', on);
  if (on) {
    autoRunning = true;
    lastCaptureTime = -1;
    autoLoop();
  } else {
    autoRunning = false;
    if (autoRafId) cancelAnimationFrame(autoRafId);
  }
}

function autoLoop() {
  if (!autoRunning) return;
  autoRafId = requestAnimationFrame(autoLoop);
  if (video.paused || video.ended) return;
  const t = video.currentTime;
  // Capture every unique second of video
  const tRounded = Math.floor(t);
  if (tRounded === lastCaptureTime) return;
  lastCaptureTime = tRounded;
  fireInference(captureFrame());
}

function manualCapture() {
  fireInference(captureFrame());
}

// --- Fire & forget inference ---
function fireInference(frame) {
  inflight++;
  updateStats();

  const ts = frame.timestamp.toFixed(1) + 's';
  const entryId = 'e' + Date.now() + Math.random().toString(36).slice(2, 6);

  logEntries.insertAdjacentHTML('afterbegin',
    '<div class="entry pending" id="' + entryId + '">' +
    '<div class="at"><span class="spin"></span>' + ts + '</div>' +
    '<div class="out">...</div></div>'
  );

  const prompt = document.getElementById('promptInput').value || '';

  fetch('/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timestamp: frame.timestamp, base64: frame.base64, prompt: prompt })
  })
  .then(r => r.json())
  .then(data => {
    const el = document.getElementById(entryId);
    if (!el) return;
    el.classList.remove('pending');
    if (data.error) {
      el.classList.add('err');
      el.querySelector('.out').textContent = data.error;
    } else {
      el.querySelector('.at').textContent = ts;
      el.querySelector('.out').textContent = data.response;
      updateScore(data.response, frame.timestamp);
    }
  })
  .catch(err => {
    const el = document.getElementById(entryId);
    if (!el) return;
    el.classList.remove('pending');
    el.classList.add('err');
    el.querySelector('.out').textContent = err.message;
  })
  .finally(() => {
    inflight--;
    doneTotal++;
    fpsTimestamps.push(performance.now());
    updateStats();
  });
}

// --- Score & drought ---
function updateScore(csvText, currentTs) {
  const lines = csvText.trim().split('\n');
  for (const line of lines) {
    const cols = line.split(',');
    if (cols.length >= 7) {
      const a = parseInt(cols[2]);
      const b = parseInt(cols[3]);
      if (!isNaN(a) && !isNaN(b)) {
        if (a > scoreA) lastScoreA_ts = currentTs;
        if (b > scoreB) lastScoreB_ts = currentTs;
        scoreA = a;
        scoreB = b;
      }
      const mTeam = cols[4]?.trim() || 'neutral';
      const mScore = cols[5]?.trim() || '0';
      document.getElementById('sbMomentum').textContent = mTeam + ' ' + mScore;
      allRows.push(line.trim());

      // Map VLM-extracted score to real wallclock via play-by-play
      const scoreKey = a + '-' + b;
      const wallclock = scoreToWallclock[scoreKey];
      if (wallclock && window.setPlayheadByWallclock) {
        window.setPlayheadByWallclock(wallclock);
        window._lastScorePlayheadUpdate = Date.now();
      }
    }
  }
  document.getElementById('sbScore').textContent = scoreA + ' - ' + scoreB;
  const dA = currentTs - lastScoreA_ts;
  const dB = currentTs - lastScoreB_ts;
  document.getElementById('droughtA').textContent = formatDrought(dA);
  document.getElementById('droughtB').textContent = formatDrought(dB);
}

function formatDrought(sec) {
  if (sec < 60) return sec.toFixed(0) + 's';
  return Math.floor(sec / 60) + 'm ' + (sec % 60).toFixed(0) + 's';
}

function exportCSV() {
  const header = 'timestamp,action,team_a_score,team_b_score,momentum_team,momentum_score,momentum_reason';
  const csvData = header + '\n' + allRows.join('\n');
  const blob = new Blob([csvData], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'inference_log.csv';
  a.click();
}

function clearLog() {
  logEntries.innerHTML = '';
  allRows.length = 0;
  scoreA = 0; scoreB = 0;
  lastScoreA_ts = 0; lastScoreB_ts = 0;
  doneTotal = 0;
  document.getElementById('sbScore').textContent = '0 - 0';
  document.getElementById('sbMomentum').textContent = 'neutral 0';
  document.getElementById('droughtA').textContent = '0s';
  document.getElementById('droughtB').textContent = '0s';
  updateStats();
}
