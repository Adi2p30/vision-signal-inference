const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const logEntries = document.getElementById('logEntries');
const tsDisplay = document.getElementById('tsDisplay');
const annotatedFrame = document.getElementById('annotatedFrame');
const summaryPanel = document.getElementById('summaryPanel');
const SCALE_W = 480;

let autoRunning = false;
let autoRafId = null;
let lastCaptureTime = -1;
let inflight = 0;
let doneTotal = 0;
let fpsTimestamps = [];

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
  const fps = fpsTimestamps.length > 1
    ? (fpsTimestamps.length / ((now - fpsTimestamps[0]) / 1000)).toFixed(1)
    : '0';
  document.getElementById('inferFps').textContent = fps;
}

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

  fetch('/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timestamp: frame.timestamp, base64: frame.base64 })
  })
  .then(r => r.json())
  .then(data => {
    const el = document.getElementById(entryId);
    if (!el) return;
    el.classList.remove('pending');

    if (data.error) {
      el.classList.add('err');
      el.querySelector('.out').textContent = data.error;
      return;
    }

    // Update annotated frame
    if (data.annotated_frame) {
      annotatedFrame.src = 'data:image/jpeg;base64,' + data.annotated_frame;
    }

    // Update summary
    if (data.summary) {
      updateSummary(data.summary);
    }

    // Format detection log entry
    const dets = data.detections || [];
    if (dets.length === 0) {
      el.querySelector('.at').textContent = ts;
      el.querySelector('.out').textContent = '(no detections)';
      el.style.opacity = '0.4';
    } else {
      el.querySelector('.at').textContent = ts + ' — ' + dets.length + ' objects';
      const lines = dets.map(d => {
        const tid = d.track_id !== null ? '#' + d.track_id + ' ' : '';
        return tid + d.class + ' ' + (d.confidence * 100).toFixed(0) + '%';
      });
      el.querySelector('.out').textContent = lines.join('\n');
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

function updateSummary(summary) {
  const entries = Object.entries(summary).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    summaryPanel.innerHTML = '<span class="summary-empty">No objects</span>';
    return;
  }
  summaryPanel.innerHTML = entries
    .map(([cls, count]) => '<span class="summary-item">' + cls + ': <span class="summary-count">' + count + '</span></span>')
    .join('');
}

function clearLog() {
  logEntries.innerHTML = '';
  summaryPanel.innerHTML = '';
  annotatedFrame.removeAttribute('src');
  doneTotal = 0;
  updateStats();
}
