// Kalshi price chart — draws Kansas & Arizona lines on a canvas
(function () {
  const KANSAS_COLOR = '#4a9eff';
  const ARIZONA_COLOR = '#ff6b6b';
  const GRID_COLOR = '#1a1a1a';
  const LABEL_COLOR = '#444';
  const PAD = { top: 12, right: 10, bottom: 24, left: 40 };

  let chartData = [];    // [{ts: Date, kansas: number|null, arizona: number|null}]
  let minTs = 0, maxTs = 0;
  let minPrice = 0, maxPrice = 100;

  const chartCanvas = document.getElementById('kalshiChart');
  const chartCtx = chartCanvas.getContext('2d');
  const playhead = document.getElementById('chartPlayhead');

  // Fetch data and render
  fetch('/kalshi-data')
    .then(r => r.json())
    .then(raw => {
      chartData = raw.map(d => ({
        ts: new Date(d.timestamp).getTime(),
        kansas: d.kansas,
        arizona: d.arizona,
      }));
      if (chartData.length === 0) return;
      minTs = chartData[0].ts;
      maxTs = chartData[chartData.length - 1].ts;

      // Compute price range from actual data
      let lo = Infinity, hi = -Infinity;
      for (const d of chartData) {
        if (d.kansas != null) { lo = Math.min(lo, d.kansas); hi = Math.max(hi, d.kansas); }
        if (d.arizona != null) { lo = Math.min(lo, d.arizona); hi = Math.max(hi, d.arizona); }
      }
      minPrice = Math.max(0, Math.floor(lo - 2));
      maxPrice = Math.min(100, Math.ceil(hi + 2));

      drawChart();
      window.addEventListener('resize', drawChart);
    });

  function drawChart() {
    const dpr = window.devicePixelRatio || 1;
    const rect = chartCanvas.getBoundingClientRect();
    chartCanvas.width = rect.width * dpr;
    chartCanvas.height = rect.height * dpr;
    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;

    // Clear
    chartCtx.clearRect(0, 0, w, h);

    if (chartData.length === 0) return;

    // Helpers
    const xOf = ts => PAD.left + ((ts - minTs) / (maxTs - minTs)) * plotW;
    const yOf = price => PAD.top + (1 - (price - minPrice) / (maxPrice - minPrice)) * plotH;

    // Grid lines & labels
    chartCtx.strokeStyle = GRID_COLOR;
    chartCtx.lineWidth = 1;
    chartCtx.font = '9px monospace';
    chartCtx.fillStyle = LABEL_COLOR;
    chartCtx.textAlign = 'right';
    chartCtx.textBaseline = 'middle';

    const priceStep = maxPrice - minPrice > 40 ? 20 : 10;
    for (let p = Math.ceil(minPrice / priceStep) * priceStep; p <= maxPrice; p += priceStep) {
      const y = yOf(p);
      chartCtx.beginPath();
      chartCtx.moveTo(PAD.left, y);
      chartCtx.lineTo(w - PAD.right, y);
      chartCtx.stroke();
      chartCtx.fillText(p + '\u00a2', PAD.left - 4, y);
    }

    // Time labels along bottom
    chartCtx.textAlign = 'center';
    chartCtx.textBaseline = 'top';
    const totalMs = maxTs - minTs;
    const labelCount = Math.min(6, Math.floor(plotW / 70));
    for (let i = 0; i <= labelCount; i++) {
      const ts = minTs + (totalMs * i) / labelCount;
      const d = new Date(ts);
      const label = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      chartCtx.fillText(label, xOf(ts), h - PAD.bottom + 6);
    }

    // Draw line helper
    function drawLine(key, color) {
      chartCtx.strokeStyle = color;
      chartCtx.lineWidth = 1.5;
      chartCtx.lineJoin = 'round';
      chartCtx.beginPath();
      let started = false;
      for (const d of chartData) {
        const val = d[key];
        if (val == null) continue;
        const x = xOf(d.ts);
        const y = yOf(val);
        if (!started) { chartCtx.moveTo(x, y); started = true; }
        else chartCtx.lineTo(x, y);
      }
      chartCtx.stroke();
    }

    drawLine('kansas', KANSAS_COLOR);
    drawLine('arizona', ARIZONA_COLOR);
  }

  // Playhead: position by real wallclock timestamp from score matching
  // Exposed globally so app.js can call it when VLM returns scores
  window.setPlayheadByWallclock = function (isoTimestamp) {
    if (chartData.length === 0 || !isoTimestamp) {
      playhead.style.display = 'none';
      return;
    }
    const ts = new Date(isoTimestamp).getTime();
    if (ts < minTs || ts > maxTs) { playhead.style.display = 'none'; return; }

    const frac = (ts - minTs) / (maxTs - minTs);
    const rect = chartCanvas.getBoundingClientRect();
    const plotW = rect.width - PAD.left - PAD.right;
    const px = PAD.left + frac * plotW;

    playhead.style.display = 'block';
    playhead.style.left = px + 'px';
  };

  // Fallback: linear mapping when no score match is available yet
  video.addEventListener('timeupdate', fallbackPlayhead);
  video.addEventListener('loadedmetadata', fallbackPlayhead);

  function fallbackPlayhead() {
    // Only use fallback if no score-based wallclock has been set recently
    if (window._lastScorePlayheadUpdate && (Date.now() - window._lastScorePlayheadUpdate < 2000)) return;
    if (chartData.length === 0 || !video.duration) {
      playhead.style.display = 'none';
      return;
    }
    const frac = video.currentTime / video.duration;
    if (frac < 0 || frac > 1) { playhead.style.display = 'none'; return; }

    const rect = chartCanvas.getBoundingClientRect();
    const plotW = rect.width - PAD.left - PAD.right;
    const px = PAD.left + frac * plotW;

    playhead.style.display = 'block';
    playhead.style.left = px + 'px';
  }
})();
