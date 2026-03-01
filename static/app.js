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
let dualVlm = false; // set from /config on load
let teamAbbr = { a: 'AWAY', b: 'HOME' };

// Play-by-play lookups for score+clock -> wallclock mapping
let scoreClockToWallclock = {};
let scoreToWallclock = {};

// Detect dual VLM mode + team names from server config
fetch('/config')
  .then(r => r.json())
  .then(cfg => {
    dualVlm = cfg.dual_vlm;
    console.log('Mode:', dualVlm ? 'dual VLM' : 'legacy single VLM');
    // Apply team names to scoreboard
    if (cfg.team_a) {
      teamAbbr.a = cfg.team_a.abbr;
      teamAbbr.b = cfg.team_b.abbr;
      document.getElementById('teamALabel').textContent = cfg.team_a.abbr;
      document.getElementById('teamBLabel').textContent = cfg.team_b.abbr;
      document.getElementById('droughtALabel').textContent = cfg.team_a.abbr;
      document.getElementById('droughtBLabel').textContent = cfg.team_b.abbr;
      console.log('Teams:', cfg.team_a.name, '(away) vs', cfg.team_b.name, '(home)');
    }
  })
  .catch(() => { dualVlm = false; });

fetch('/play-by-play')
  .then(r => r.json())
  .then(plays => {
    const seenScore = new Set();
    for (const p of plays) {
      const scoreKey = p.awayScore + '-' + p.homeScore;
      const clockKey = scoreKey + '@' + p.clock;
      scoreClockToWallclock[clockKey] = p.wallclock;
      if (!seenScore.has(scoreKey)) {
        seenScore.add(scoreKey);
        scoreToWallclock[scoreKey] = p.wallclock;
      }
    }
    console.log('Play-by-play loaded:', Object.keys(scoreClockToWallclock).length, 'score+clock states,', Object.keys(scoreToWallclock).length, 'score states');
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
    } else if (data.momentum !== undefined) {
      // Dual VLM mode
      handleDualResponse(data, el, ts, frame.timestamp);
    } else {
      // Legacy single VLM mode
      handleLegacyResponse(data, el, ts, frame.timestamp);
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

// --- Dual VLM response handling ---
function handleDualResponse(data, el, ts, currentTs) {
  el.querySelector('.at').textContent = ts;
  const momentum = (data.momentum || '').trim();
  const score = (data.score || '').trim();

  // Build display text
  let lines = [];
  if (momentum && momentum !== 'NONE') lines.push(momentum);
  if (score && score !== 'NONE') lines.push('score: ' + score);

  if (lines.length === 0) {
    el.querySelector('.out').textContent = '(no data)';
    el.style.opacity = '0.4';
  } else {
    el.querySelector('.out').textContent = lines.join('\n');
  }

  // Update scoreboard from score VLM
  if (score && score !== 'NONE') {
    updateScoreFromVLM(score, currentTs);
  }

  // Update momentum from momentum VLM
  if (momentum && momentum !== 'NONE') {
    updateMomentumFromVLM(momentum);
  }

  // Apply clock correction from backend (overrides raw VLM clock)
  if (data.clock_correction) {
    applyClockCorrection(data.clock_correction);
  }

  // Display flags from backend
  displayFlags(data.flags);

  // Combine for CSV export
  if (momentum && momentum !== 'NONE') {
    const mCols = momentum.split(',');
    const sCols = (score && score !== 'NONE') ? score.split(',') : ['', '', '', ''];
    if (mCols.length >= 5) {
      const combined = [
        mCols[0], mCols[1],
        sCols[0] || '', sCols[1] || '', sCols[2] || '', sCols[3] || '',
        mCols[2], mCols[3], mCols[4]
      ].join(',');
      allRows.push(combined);
    }
  }
}

function updateScoreFromVLM(scoreText, currentTs) {
  const cols = scoreText.split(',');
  if (cols.length >= 3) {
    const a = parseInt(cols[0]);
    const b = parseInt(cols[1]);
    const gameClock = cols[2]?.trim() || '';
    const period = cols[3] ? parseInt(cols[3].trim()) : null;

    if (!isNaN(a) && !isNaN(b)) {
      if (a > scoreA) lastScoreA_ts = currentTs;
      if (b > scoreB) lastScoreB_ts = currentTs;
      scoreA = a;
      scoreB = b;
    }

    document.getElementById('sbScore').textContent = scoreA + ' - ' + scoreB;
    if (gameClock) document.getElementById('sbClock').textContent = gameClock;
    if (period && !isNaN(period)) document.getElementById('sbPeriod').textContent = period + 'H';

    const dA = currentTs - lastScoreA_ts;
    const dB = currentTs - lastScoreB_ts;
    document.getElementById('droughtA').textContent = formatDrought(dA);
    document.getElementById('droughtB').textContent = formatDrought(dB);

    // Map to wallclock for chart
    if (!isNaN(a) && !isNaN(b)) {
      const scoreKey = a + '-' + b;
      const clockKey = scoreKey + '@' + gameClock;
      const wallclock = scoreClockToWallclock[clockKey] || scoreToWallclock[scoreKey];
      if (wallclock && window.setPlayheadByWallclock) {
        window.setPlayheadByWallclock(wallclock);
        window._lastScorePlayheadUpdate = Date.now();
      }
    }
  }
}

function updateMomentumFromVLM(momentumText) {
  const cols = momentumText.split(',');
  // Format: timestamp,action,momentum_team,momentum_score,momentum_reason
  if (cols.length >= 5) {
    const mTeam = cols[2]?.trim() || 'neutral';
    const mScore = cols[3]?.trim() || '0';
    document.getElementById('sbMomentum').textContent = mTeam + ' ' + mScore;
  }
}

// --- Legacy single VLM response handling ---
function handleLegacyResponse(data, el, ts, currentTs) {
  if (data.response.trim() === 'NONE') {
    el.querySelector('.at').textContent = ts;
    el.querySelector('.out').textContent = '(no score visible)';
    el.style.opacity = '0.4';
  } else {
    el.querySelector('.at').textContent = ts;
    el.querySelector('.out').textContent = data.response;
    updateScoreLegacy(data.response, currentTs);
  }
  displayFlags(data.flags);
}

// --- Legacy score update (9-column CSV format with period) ---
function updateScoreLegacy(csvText, currentTs) {
  const trimmed = csvText.trim();
  if (trimmed === 'NONE' || trimmed === '') return;

  const lines = trimmed.split('\n');
  for (const line of lines) {
    const cols = line.split(',');
    // Format: timestamp,action,score_a,score_b,clock,period,mom_team,mom_score,mom_reason
    if (cols.length >= 9) {
      const a = parseInt(cols[2]);
      const b = parseInt(cols[3]);
      const gameClock = cols[4]?.trim() || '';
      const period = parseInt(cols[5]?.trim());
      if (!isNaN(a) && !isNaN(b)) {
        if (a > scoreA) lastScoreA_ts = currentTs;
        if (b > scoreB) lastScoreB_ts = currentTs;
        scoreA = a;
        scoreB = b;
      }
      if (gameClock) document.getElementById('sbClock').textContent = gameClock;
      if (!isNaN(period)) document.getElementById('sbPeriod').textContent = period + 'H';
      const mTeam = cols[6]?.trim() || 'neutral';
      const mScore = cols[7]?.trim() || '0';
      document.getElementById('sbMomentum').textContent = mTeam + ' ' + mScore;
      allRows.push(line.trim());

      const scoreKey = a + '-' + b;
      const clockKey = scoreKey + '@' + gameClock;
      const wallclock = scoreClockToWallclock[clockKey] || scoreToWallclock[scoreKey];
      if (wallclock && window.setPlayheadByWallclock) {
        window.setPlayheadByWallclock(wallclock);
        window._lastScorePlayheadUpdate = Date.now();
      }
    } else if (cols.length >= 8) {
      // Fallback: old 8-column format without period
      const a = parseInt(cols[2]);
      const b = parseInt(cols[3]);
      const gameClock = cols[4]?.trim() || '';
      if (!isNaN(a) && !isNaN(b)) {
        if (a > scoreA) lastScoreA_ts = currentTs;
        if (b > scoreB) lastScoreB_ts = currentTs;
        scoreA = a;
        scoreB = b;
      }
      if (gameClock) document.getElementById('sbClock').textContent = gameClock;
      const mTeam = cols[5]?.trim() || 'neutral';
      const mScore = cols[6]?.trim() || '0';
      document.getElementById('sbMomentum').textContent = mTeam + ' ' + mScore;
      allRows.push(line.trim());
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

// --- Clock self-correction display ---
function applyClockCorrection(cc) {
  const el = document.getElementById('sbCorrection');
  if (!cc) { el.textContent = ''; return; }

  if (cc.accepted) {
    el.textContent = '';
    el.className = 'sb-correction';
  } else {
    // Clock was rejected -> show corrected value and streak
    if (cc.corrected_display) {
      document.getElementById('sbClock').textContent = cc.corrected_display;
    }
    el.textContent = 'CORRECTING (' + cc.outlier_streak + '/30)';
    el.className = 'sb-correction active';
  }
}

// --- Flag display ---
function displayFlags(flags) {
  const container = document.getElementById('flagsContainer');
  if (!flags || flags.length === 0) {
    container.innerHTML = '';
    container.style.display = 'none';
    return;
  }
  container.style.display = 'block';
  container.innerHTML = flags.map(f =>
    '<div class="flag flag-' + f.severity + '">' +
    '<span class="flag-label">' + f.label + '</span> ' +
    '<span class="flag-msg">' + f.message + '</span>' +
    '</div>'
  ).join('');
}

function exportCSV() {
  const header = 'timestamp,action,team_a_score,team_b_score,game_clock,momentum_team,momentum_score,momentum_reason';
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
  document.getElementById('sbClock').textContent = '20:00';
  document.getElementById('sbPeriod').textContent = '1H';
  document.getElementById('sbCorrection').textContent = '';
  document.getElementById('sbMomentum').textContent = 'neutral 0';
  document.getElementById('droughtA').textContent = '0s';
  document.getElementById('droughtB').textContent = '0s';
  document.getElementById('flagsContainer').innerHTML = '';
  document.getElementById('flagsContainer').style.display = 'none';
  updateStats();
}
