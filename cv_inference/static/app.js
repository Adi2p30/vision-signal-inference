const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('overlayCanvas');
const overlayCtx = overlay.getContext('2d');
const courtCanvas = document.getElementById('courtCanvas');
const courtCtx = courtCanvas.getContext('2d');
const logEntries = document.getElementById('logEntries');
const tsDisplay = document.getElementById('tsDisplay');
const summaryPanel = document.getElementById('summaryPanel');
const courtInfo = document.getElementById('courtInfo');
const movementPanel = document.getElementById('movementPanel');
const SCALE_W = 480;
const TIMEOUT_MS = 4000;
const MAX_CONCURRENT = 6;

let autoRunning = false;
let autoRafId = null;
let inflight = 0;
let doneTotal = 0;
let skippedTotal = 0;
let fpsTimestamps = [];
let showBoxes = true;
let showTrails = true;
let showBall = true;
let showMasks = true;
let currentModel = 'sam2';
let latestData = null;

/* Court / calibration state */
let homography = null;     // 3×3 matrix
let calibrating = false;
let calibPoints = [];
let calibFrameSize = null;

/* ── Color palette for track IDs ── */
const TRACK_COLORS = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
  '#469990', '#dcbeff', '#9a6324', '#fffac8', '#800000',
  '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
];
function trackColor(tid) {
  return TRACK_COLORS[tid % TRACK_COLORS.length];
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

function setModel(mode) {
  currentModel = mode;
  document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('btn-' + mode);
  if (btn) btn.classList.add('active');
  const maskLabel = document.getElementById('maskLabel');
  if (maskLabel) maskLabel.style.display = mode === 'sam2' ? '' : 'none';
  const ml = document.getElementById('modelLabel');
  if (ml) ml.textContent = mode;
}

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
    base64: canvas.toDataURL('image/jpeg', 0.5).split(',')[1],
    frameW: canvas.width,
    frameH: canvas.height,
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
  document.getElementById('skippedCount').textContent = skippedTotal;
}

function toggleAuto() {
  const on = document.getElementById('autoCheck').checked;
  document.getElementById('autoLabel').classList.toggle('active', on);
  if (on) {
    autoRunning = true;
    autoLoop();
  } else {
    autoRunning = false;
    if (autoRafId) cancelAnimationFrame(autoRafId);
  }
}

function toggleBoxes() {
  showBoxes = document.getElementById('boxCheck').checked;
  document.getElementById('boxLabel').classList.toggle('active', showBoxes);
  redrawOverlay();
}

function toggleTrails() {
  showTrails = document.getElementById('trailCheck').checked;
  document.getElementById('trailLabel').classList.toggle('active', showTrails);
  redrawOverlay();
}

function toggleBall() {
  showBall = document.getElementById('ballCheck').checked;
  document.getElementById('ballLabel').classList.toggle('active', showBall);
  redrawOverlay();
}

function toggleMasks() {
  showMasks = document.getElementById('maskCheck').checked;
  document.getElementById('maskLabel').classList.toggle('active', showMasks);
  redrawOverlay();
}

/* ─────────────────── HOMOGRAPHY MATH ─────────────────── */

const COURT_FT_W = 94, COURT_FT_H = 50;

function gaussianElim(A, b) {
  const n = A.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let mr = c, mv = Math.abs(M[c][c]);
    for (let r = c + 1; r < n; r++) {
      if (Math.abs(M[r][c]) > mv) { mv = Math.abs(M[r][c]); mr = r; }
    }
    [M[c], M[mr]] = [M[mr], M[c]];
    if (Math.abs(M[c][c]) < 1e-10) return null;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  const x = Array(n);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

function computeHomography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [X, Y] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.push(X);
    A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.push(Y);
  }
  const h = gaussianElim(A, b);
  if (!h) return null;
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

function projectPt(H, pt) {
  const [x, y] = pt;
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (Math.abs(w) < 1e-10) return null;
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

/* ─────────────────── CALIBRATION ─────────────────── */

const CORNER_NAMES = ['TOP-LEFT', 'TOP-RIGHT', 'BOTTOM-RIGHT', 'BOTTOM-LEFT'];

function startCalibration() {
  if (calibrating) { cancelCalibration(); return; }
  calibrating = true;
  calibPoints = [];
  overlay.style.pointerEvents = 'auto';
  overlay.style.cursor = 'crosshair';
  overlay.addEventListener('click', onCalibClick);
  courtInfo.textContent = 'Click corner 1/4: ' + CORNER_NAMES[0];
  courtInfo.style.color = '#ffe119';
  document.getElementById('calibBtn').textContent = 'Cancel';
}

function cancelCalibration() {
  calibrating = false;
  calibPoints = [];
  overlay.style.pointerEvents = 'none';
  overlay.style.cursor = '';
  overlay.removeEventListener('click', onCalibClick);
  courtInfo.textContent = homography ? 'Court calibrated' : 'Set court corners to enable minimap';
  courtInfo.style.color = homography ? '#3cb44b' : '#333';
  document.getElementById('calibBtn').textContent = 'Calibrate';
  redrawOverlay();
}

function onCalibClick(e) {
  if (!calibrating) return;
  const rect = overlay.getBoundingClientRect();
  const frame = captureFrame();
  const sx = frame.frameW / rect.width;
  const sy = frame.frameH / rect.height;
  calibPoints.push([
    Math.round((e.clientX - rect.left) * sx * 10) / 10,
    Math.round((e.clientY - rect.top) * sy * 10) / 10,
  ]);
  calibFrameSize = [frame.frameW, frame.frameH];
  drawCalibMarkers();
  if (calibPoints.length < 4) {
    courtInfo.textContent = 'Click corner ' + (calibPoints.length + 1) + '/4: ' + CORNER_NAMES[calibPoints.length];
  } else {
    applyCourt(calibPoints);
    cancelCalibration();
  }
}

function drawCalibMarkers() {
  redrawOverlay();
  if (!calibFrameSize) return;
  const dw = overlay.clientWidth, dh = overlay.clientHeight;
  const [sw, sh] = calibFrameSize;
  overlayCtx.save();
  for (let i = 0; i < calibPoints.length; i++) {
    const px = calibPoints[i][0] * (dw / sw);
    const py = calibPoints[i][1] * (dh / sh);
    overlayCtx.strokeStyle = '#ffe119';
    overlayCtx.lineWidth = 1.5;
    overlayCtx.beginPath();
    overlayCtx.moveTo(px - 10, py); overlayCtx.lineTo(px + 10, py);
    overlayCtx.moveTo(px, py - 10); overlayCtx.lineTo(px, py + 10);
    overlayCtx.stroke();
    overlayCtx.fillStyle = '#ffe119';
    overlayCtx.font = 'bold 12px monospace';
    overlayCtx.fillText((i + 1) + '', px + 8, py - 6);
  }
  overlayCtx.restore();
}

function applyCourt(corners) {
  const dst = [[0, 0], [COURT_FT_W, 0], [COURT_FT_W, COURT_FT_H], [0, COURT_FT_H]];
  homography = computeHomography(corners, dst);
  if (homography) {
    courtInfo.textContent = 'Court calibrated — minimap active';
    courtInfo.style.color = '#3cb44b';
    drawCourtBase();
    if (latestData) drawPlayersOnCourt(latestData.detections, latestData.trails, latestData.frame_size, latestData.ball, latestData.ball_trail);
  } else {
    courtInfo.textContent = 'Homography failed — try again';
    courtInfo.style.color = '#c44';
  }
}

function autoDetectCourt() {
  const frame = captureFrame();
  courtInfo.textContent = 'Detecting court…';
  courtInfo.style.color = '#ffe119';
  fetch('/court-detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ base64: frame.base64 }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error || !data.corners) {
      courtInfo.textContent = data.error || 'No court detected — try Calibrate';
      courtInfo.style.color = '#c44';
      return;
    }
    calibFrameSize = data.frame_size || [frame.frameW, frame.frameH];
    calibPoints = data.corners;
    drawCalibMarkers();
    applyCourt(data.corners);

    // Show partial info
    if (data.partial) {
      const edges = (data.edges_touching || []).join(', ');
      courtInfo.textContent = 'Court calibrated (partial — touches: ' + edges + ')';
      courtInfo.style.color = '#f58231';
    }
    calibPoints = [];
  })
  .catch(err => {
    courtInfo.textContent = err.message;
    courtInfo.style.color = '#c44';
  });
}

function resetCourt() {
  homography = null;
  calibPoints = [];
  calibFrameSize = null;
  courtInfo.textContent = 'Set court corners to enable minimap';
  courtInfo.style.color = '#333';
  initCourtCanvas();
}

/* ─────────────────── COURT MINIMAP ─────────────────── */

function initCourtCanvas() {
  const rect = courtCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = rect.width * dpr;
  const h = Math.round(w * (COURT_FT_H / COURT_FT_W));
  courtCanvas.width = w;
  courtCanvas.height = h;
  courtCanvas.style.height = Math.round(rect.width * (COURT_FT_H / COURT_FT_W)) + 'px';
  drawCourtBase();
}

function drawCourtBase() {
  const cw = courtCanvas.width, ch = courtCanvas.height;
  courtCtx.clearRect(0, 0, cw, ch);
  courtCtx.fillStyle = '#12100a';
  courtCtx.fillRect(0, 0, cw, ch);

  const sx = cw / COURT_FT_W, sy = ch / COURT_FT_H;
  const r = Math.min(sx, sy);

  courtCtx.strokeStyle = 'rgba(255,255,255,0.2)';
  courtCtx.lineWidth = 1.5;

  // Court outline
  courtCtx.strokeRect(2, 2, cw - 4, ch - 4);

  // Half-court line
  const mid = (COURT_FT_W / 2) * sx;
  courtCtx.beginPath();
  courtCtx.moveTo(mid, 2);
  courtCtx.lineTo(mid, ch - 2);
  courtCtx.stroke();

  // Center circle
  courtCtx.beginPath();
  courtCtx.arc(mid, ch / 2, 6 * r, 0, Math.PI * 2);
  courtCtx.stroke();

  // Keys (paint areas): 19ft deep × 16ft wide
  const kd = 19 * sx, kw = 16 * sy, ky = (ch - kw) / 2;
  courtCtx.strokeRect(2, ky, kd, kw);
  courtCtx.strokeRect(cw - 2 - kd, ky, kd, kw);

  // FT circles
  courtCtx.beginPath();
  courtCtx.arc(kd + 2, ch / 2, 6 * r, 0, Math.PI * 2);
  courtCtx.stroke();
  courtCtx.beginPath();
  courtCtx.arc(cw - kd - 2, ch / 2, 6 * r, 0, Math.PI * 2);
  courtCtx.stroke();

  // Baskets (5.25ft from baseline)
  const bx = 5.25 * sx;
  courtCtx.fillStyle = 'rgba(255,100,50,0.4)';
  courtCtx.beginPath();
  courtCtx.arc(bx, ch / 2, 4, 0, Math.PI * 2);
  courtCtx.fill();
  courtCtx.beginPath();
  courtCtx.arc(cw - bx, ch / 2, 4, 0, Math.PI * 2);
  courtCtx.fill();

  // 3-point arcs (simplified)
  const ar = 23.75 * sx;
  courtCtx.strokeStyle = 'rgba(255,255,255,0.12)';
  courtCtx.beginPath();
  courtCtx.arc(bx, ch / 2, ar, -1.2, 1.2);
  courtCtx.stroke();
  courtCtx.beginPath();
  courtCtx.arc(cw - bx, ch / 2, ar, Math.PI - 1.2, Math.PI + 1.2);
  courtCtx.stroke();
}

function drawPlayersOnCourt(dets, trails, frameSize, ball, ballTrail) {
  if (!homography || !dets) return;
  const cw = courtCanvas.width, ch = courtCanvas.height;
  const sx = cw / COURT_FT_W, sy = ch / COURT_FT_H;

  // Draw person trails on court
  if (showTrails && trails) {
    for (const [tidStr, pts] of Object.entries(trails)) {
      if (pts.length < 2) continue;
      const color = trackColor(parseInt(tidStr));
      courtCtx.beginPath();
      courtCtx.strokeStyle = color;
      courtCtx.lineWidth = 2;
      courtCtx.globalAlpha = 0.4;
      let started = false;
      for (const p of pts) {
        const proj = projectPt(homography, p);
        if (!proj || proj[0] < -5 || proj[0] > COURT_FT_W + 5 || proj[1] < -5 || proj[1] > COURT_FT_H + 5) continue;
        if (!started) { courtCtx.moveTo(proj[0] * sx, proj[1] * sy); started = true; }
        else courtCtx.lineTo(proj[0] * sx, proj[1] * sy);
      }
      courtCtx.stroke();
      courtCtx.globalAlpha = 1;
    }
  }

  // Draw ball trail on court
  if (showBall && ballTrail && ballTrail.length > 1) {
    courtCtx.beginPath();
    courtCtx.strokeStyle = '#ff8c00';
    courtCtx.lineWidth = 2;
    courtCtx.globalAlpha = 0.4;
    let started = false;
    for (const p of ballTrail) {
      const proj = projectPt(homography, p);
      if (!proj || proj[0] < -5 || proj[0] > COURT_FT_W + 5 || proj[1] < -5 || proj[1] > COURT_FT_H + 5) continue;
      if (!started) { courtCtx.moveTo(proj[0] * sx, proj[1] * sy); started = true; }
      else courtCtx.lineTo(proj[0] * sx, proj[1] * sy);
    }
    courtCtx.stroke();
    courtCtx.globalAlpha = 1;
  }

  // Draw player dots
  for (const d of dets) {
    const proj = projectPt(homography, d.center);
    if (!proj || proj[0] < -5 || proj[0] > COURT_FT_W + 5 || proj[1] < -5 || proj[1] > COURT_FT_H + 5) continue;
    const px = proj[0] * sx, py = proj[1] * sy;
    const color = d.track_id !== null ? trackColor(d.track_id) : '#fff';

    // Filled dot
    courtCtx.beginPath();
    courtCtx.fillStyle = color;
    courtCtx.arc(px, py, 7, 0, Math.PI * 2);
    courtCtx.fill();

    // Border
    courtCtx.beginPath();
    courtCtx.strokeStyle = '#fff';
    courtCtx.lineWidth = 1.5;
    courtCtx.arc(px, py, 7, 0, Math.PI * 2);
    courtCtx.stroke();

    // ID label
    if (d.track_id !== null) {
      courtCtx.font = 'bold ' + Math.max(10, Math.round(cw / 60)) + 'px monospace';
      courtCtx.fillStyle = '#fff';
      courtCtx.textAlign = 'center';
      courtCtx.fillText('#' + d.track_id, px, py - 11);
      courtCtx.textAlign = 'left';
    }
  }

  // Draw ball on court
  if (showBall && ball) {
    const proj = projectPt(homography, ball.center);
    if (proj && proj[0] > -5 && proj[0] < COURT_FT_W + 5 && proj[1] > -5 && proj[1] < COURT_FT_H + 5) {
      const bx = proj[0] * sx, by = proj[1] * sy;
      // Glow
      courtCtx.beginPath();
      courtCtx.arc(bx, by, 8, 0, Math.PI * 2);
      courtCtx.fillStyle = 'rgba(255,102,0,0.3)';
      courtCtx.fill();
      // Ball dot
      courtCtx.beginPath();
      courtCtx.arc(bx, by, 5, 0, Math.PI * 2);
      courtCtx.fillStyle = '#ff6600';
      courtCtx.fill();
      courtCtx.strokeStyle = '#fff';
      courtCtx.lineWidth = 1.5;
      courtCtx.stroke();
    }
  }
}

function updateMinimap(dets, trails, frameSize, ball, ballTrail) {
  if (!homography) return;
  drawCourtBase();
  drawPlayersOnCourt(dets, trails, frameSize, ball, ballTrail);
}

function autoLoop() {
  if (!autoRunning) return;
  autoRafId = requestAnimationFrame(autoLoop);
  if (video.paused || video.ended) return;
  if (inflight < MAX_CONCURRENT) {
    fireInference(captureFrame());
  }
}

function manualCapture() {
  fireInference(captureFrame());
}

function clearOverlay() {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

/* ── Redraw from latest cached data ── */
function redrawOverlay() {
  clearOverlay();
  if (!latestData) return;
  if (!showBoxes && !showTrails && !showBall && !showMasks) return;
  drawDetections(
    latestData.detections, latestData.trails, latestData.frame_size,
    latestData.ball, latestData.ball_trail, latestData.mask_contours, latestData.movement
  );
}

/* ── Client-side drawing ── */
function drawDetections(dets, trails, frameSize, ball, ballTrail, maskContours, movement) {
  const displayW = overlay.clientWidth;
  const displayH = overlay.clientHeight;
  overlay.width = displayW;
  overlay.height = displayH;
  overlayCtx.clearRect(0, 0, displayW, displayH);

  const [srcW, srcH] = frameSize;
  const scaleX = displayW / srcW;
  const scaleY = displayH / srcH;

  // Draw SAM2 masks (behind everything)
  if (showMasks && maskContours) {
    for (const [tidStr, polys] of Object.entries(maskContours)) {
      const tid = parseInt(tidStr);
      const color = trackColor(tid);
      for (const poly of polys) {
        if (poly.length < 3) continue;
        overlayCtx.beginPath();
        overlayCtx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
        for (let j = 1; j < poly.length; j++) {
          overlayCtx.lineTo(poly[j][0] * scaleX, poly[j][1] * scaleY);
        }
        overlayCtx.closePath();
        overlayCtx.fillStyle = hexToRgba(color, 0.3);
        overlayCtx.fill();
        overlayCtx.save();
        overlayCtx.shadowColor = color;
        overlayCtx.shadowBlur = 4;
        overlayCtx.strokeStyle = hexToRgba(color, 0.8);
        overlayCtx.lineWidth = 1.5;
        overlayCtx.stroke();
        overlayCtx.restore();
      }
    }
  }

  // Draw person trails (behind boxes)
  if (showTrails && trails) {
    for (const [tidStr, points] of Object.entries(trails)) {
      if (points.length < 2) continue;
      const tid = parseInt(tidStr);
      const color = trackColor(tid);

      overlayCtx.beginPath();
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 2;
      overlayCtx.globalAlpha = 0.7;

      for (let j = 0; j < points.length; j++) {
        const px = points[j][0] * scaleX;
        const py = points[j][1] * scaleY;
        if (j === 0) overlayCtx.moveTo(px, py);
        else overlayCtx.lineTo(px, py);
      }
      overlayCtx.stroke();

      // Draw dot at latest position
      const last = points[points.length - 1];
      overlayCtx.beginPath();
      overlayCtx.fillStyle = color;
      overlayCtx.arc(last[0] * scaleX, last[1] * scaleY, 3, 0, Math.PI * 2);
      overlayCtx.fill();
      overlayCtx.globalAlpha = 1.0;
    }
  }

  // Draw bounding boxes + labels
  if (showBoxes && dets) {
    for (const d of dets) {
      const [x1, y1, x2, y2] = d.bbox;
      const sx = x1 * scaleX;
      const sy = y1 * scaleY;
      const sw = (x2 - x1) * scaleX;
      const sh = (y2 - y1) * scaleY;
      const tid = d.track_id;
      const color = tid !== null ? trackColor(tid) : '#fff';

      // Box
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeRect(sx, sy, sw, sh);

      // Label background
      const label = tid !== null ? 'P#' + tid : 'P';
      overlayCtx.font = 'bold 11px monospace';
      const tw = overlayCtx.measureText(label).width + 6;
      overlayCtx.fillStyle = color;
      overlayCtx.fillRect(sx, sy - 16, tw, 16);

      // Label text
      overlayCtx.fillStyle = '#000';
      overlayCtx.fillText(label, sx + 3, sy - 4);
    }
  }

  // Draw movement indicators (speed + direction arrows)
  if (movement && dets) {
    for (const d of dets) {
      const tid = d.track_id;
      if (tid === null || !movement[String(tid)]) continue;
      const m = movement[String(tid)];
      const [bx1, by1, bx2, by2] = d.bbox;
      const labelX = bx2 * scaleX + 4;
      const labelY = (by1 + (by2 - by1) * 0.3) * scaleY;
      overlayCtx.font = 'bold 9px monospace';
      const speedText = m.speed.toFixed(0) + ' px/s';
      const stw = overlayCtx.measureText(speedText).width;
      overlayCtx.fillStyle = 'rgba(0,0,0,0.7)';
      overlayCtx.fillRect(labelX - 2, labelY - 10, stw + 4, 13);
      overlayCtx.fillStyle = '#0f0';
      overlayCtx.fillText(speedText, labelX, labelY);
      if (m.speed > 5) {
        const acx = ((bx1 + bx2) / 2) * scaleX;
        const acy = by2 * scaleY + 8;
        const arrowLen = Math.min(m.speed * 0.3, 20);
        const adx = m.direction[0] * arrowLen;
        const ady = m.direction[1] * arrowLen;
        overlayCtx.beginPath();
        overlayCtx.moveTo(acx, acy);
        overlayCtx.lineTo(acx + adx, acy + ady);
        overlayCtx.strokeStyle = '#0f0';
        overlayCtx.lineWidth = 2;
        overlayCtx.stroke();
        const ang = Math.atan2(ady, adx);
        overlayCtx.beginPath();
        overlayCtx.moveTo(acx + adx, acy + ady);
        overlayCtx.lineTo(acx + adx - 5 * Math.cos(ang - 0.5), acy + ady - 5 * Math.sin(ang - 0.5));
        overlayCtx.lineTo(acx + adx - 5 * Math.cos(ang + 0.5), acy + ady - 5 * Math.sin(ang + 0.5));
        overlayCtx.closePath();
        overlayCtx.fillStyle = '#0f0';
        overlayCtx.fill();
      }
    }
  }

  // Draw ball
  if (showBall && ball) {
    const bx = ball.center[0] * scaleX;
    const by = ball.center[1] * scaleY;
    const br = Math.max(ball.radius * Math.min(scaleX, scaleY), 5);

    // Ball trail
    if (ballTrail && ballTrail.length > 1) {
      overlayCtx.beginPath();
      overlayCtx.strokeStyle = '#ff8c00';
      overlayCtx.lineWidth = 2;
      overlayCtx.globalAlpha = 0.5;
      for (let j = 0; j < ballTrail.length; j++) {
        const px = ballTrail[j][0] * scaleX;
        const py = ballTrail[j][1] * scaleY;
        if (j === 0) overlayCtx.moveTo(px, py);
        else overlayCtx.lineTo(px, py);
      }
      overlayCtx.stroke();
      overlayCtx.globalAlpha = 1;
    }

    // Ball circle with glow
    overlayCtx.beginPath();
    overlayCtx.arc(bx, by, br + 3, 0, Math.PI * 2);
    overlayCtx.strokeStyle = 'rgba(255,140,0,0.5)';
    overlayCtx.lineWidth = 3;
    overlayCtx.stroke();

    overlayCtx.beginPath();
    overlayCtx.arc(bx, by, br, 0, Math.PI * 2);
    overlayCtx.fillStyle = '#ff6600';
    overlayCtx.fill();
    overlayCtx.strokeStyle = '#fff';
    overlayCtx.lineWidth = 1.5;
    overlayCtx.stroke();

    // Label
    overlayCtx.font = 'bold 10px monospace';
    overlayCtx.fillStyle = '#ff6600';
    const lbl = 'BALL';
    const tw = overlayCtx.measureText(lbl).width;
    overlayCtx.fillStyle = 'rgba(0,0,0,0.6)';
    overlayCtx.fillRect(bx - tw / 2 - 3, by - br - 18, tw + 6, 14);
    overlayCtx.fillStyle = '#ff6600';
    overlayCtx.fillText(lbl, bx - tw / 2, by - br - 7);
  }
}

// Keep only the latest N log entries to prevent DOM bloat at high frame rates
const MAX_LOG_ENTRIES = 50;
function trimLog() {
  while (logEntries.children.length > MAX_LOG_ENTRIES) {
    logEntries.removeChild(logEntries.lastChild);
  }
}

function fireInference(frame) {
  const abort = new AbortController();
  const timeoutId = setTimeout(() => abort.abort(), TIMEOUT_MS);

  inflight++;
  updateStats();

  const ts = frame.timestamp.toFixed(1) + 's';
  const entryId = 'e' + Date.now() + Math.random().toString(36).slice(2, 6);

  logEntries.insertAdjacentHTML('afterbegin',
    '<div class="entry pending" id="' + entryId + '">' +
    '<div class="at"><span class="spin"></span>' + ts + '</div>' +
    '<div class="out">...</div></div>'
  );
  trimLog();

  const endpoint = currentModel === 'sam2' ? '/segment' : '/detect';
  fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timestamp: frame.timestamp, base64: frame.base64 }),
    signal: abort.signal,
  })
  .then(r => r.json())
  .then(data => {
    clearTimeout(timeoutId);
    const el = document.getElementById(entryId);
    if (!el) return;
    el.classList.remove('pending');

    if (data.error) {
      el.classList.add('err');
      el.querySelector('.out').textContent = data.skipped ? 'TIMEOUT' : data.error;
      skippedTotal++;
      return;
    }

    const dets = data.detections || [];
    const trails = data.trails || {};
    const ball = data.ball || null;
    const ballTrail = data.ball_trail || [];
    const frameSize = data.frame_size || [480, 270];
    const count = dets.length;
    document.getElementById('peopleCount').textContent = count;

    const maskContours = data.mask_contours || {};
    const movement = data.movement || {};

    // Cache latest data and draw client-side
    latestData = { detections: dets, trails: trails, ball: ball, ball_trail: ballTrail, frame_size: frameSize, mask_contours: maskContours, movement: movement };
    redrawOverlay();
    updateMinimap(dets, trails, frameSize, ball, ballTrail);
    updateMovementPanel(movement);

    // Summary
    let summaryHtml = '';
    if (count > 0) summaryHtml += '<span class="summary-item">People: <span class="summary-count">' + count + '</span></span>';
    if (ball) summaryHtml += '<span class="summary-item ball-tag">Ball: <span class="summary-count">' + ball.source + '</span></span>';
    const maskCount = Object.keys(maskContours).length;
    if (maskCount > 0) summaryHtml += '<span class="summary-item mask-tag">Masks: <span class="summary-count">' + maskCount + '</span></span>';
    summaryPanel.innerHTML = summaryHtml || '<span class="summary-empty">No detections</span>';

    if (count === 0) {
      el.querySelector('.at').textContent = ts;
      el.querySelector('.out').textContent = '(no people)';
      el.style.opacity = '0.4';
    } else {
      el.querySelector('.at').textContent = ts + ' — ' + count + 'P';
      const lines = dets.map(d => {
        const tid = d.track_id !== null ? '#' + d.track_id + ' ' : '';
        const [x1, y1, x2, y2] = d.bbox;
        return tid + 'P [' + x1 + ',' + y1 + ',' + x2 + ',' + y2 + ']';
      });
      el.querySelector('.out').textContent = lines.join('\n');
    }
  })
  .catch(err => {
    clearTimeout(timeoutId);
    const el = document.getElementById(entryId);
    if (!el) return;
    el.classList.remove('pending');
    el.classList.add('err');
    if (err.name === 'AbortError') {
      el.querySelector('.out').textContent = 'TIMEOUT';
      skippedTotal++;
    } else {
      el.querySelector('.out').textContent = err.message;
    }
  })
  .finally(() => {
    inflight--;
    doneTotal++;
    fpsTimestamps.push(performance.now());
    updateStats();
  });
}

function clearLog() {
  logEntries.innerHTML = '';
  summaryPanel.innerHTML = '';
  latestData = null;
  clearOverlay();
  if (homography) drawCourtBase();
  doneTotal = 0;
  skippedTotal = 0;
  document.getElementById('peopleCount').textContent = '0';
  updateStats();
}

function updateMovementPanel(movement) {
  if (!movementPanel || !movement || Object.keys(movement).length === 0) {
    if (movementPanel) movementPanel.innerHTML = '';
    return;
  }
  let html = '<div class="movement-head">Movement</div>';
  const entries = Object.entries(movement).sort((a, b) => b[1].speed - a[1].speed);
  for (const [tidStr, m] of entries) {
    const tid = parseInt(tidStr);
    const color = trackColor(tid);
    html += '<div class="movement-row">'
      + '<span class="movement-id" style="color:' + color + '">#' + tid + '</span>'
      + '<span class="movement-speed">' + m.speed.toFixed(0) + ' px/s</span>'
      + '<span class="movement-dist">' + m.distance.toFixed(0) + 'px total</span>'
      + '</div>';
  }
  movementPanel.innerHTML = html;
}

// Initialize toggle states
document.getElementById('boxLabel').classList.add('active');
document.getElementById('trailLabel').classList.add('active');
document.getElementById('ballLabel').classList.add('active');
document.getElementById('maskLabel').classList.add('active');

// Check capabilities and set initial model
fetch('/capabilities').then(r => r.json()).then(data => {
  if (!data.models.sam2) {
    setModel('yolo');
    const btn = document.getElementById('btn-sam2');
    if (btn) { btn.disabled = true; btn.style.opacity = '0.3'; }
  }
}).catch(() => {});

initCourtCanvas();
window.addEventListener('resize', () => initCourtCanvas());
