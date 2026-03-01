// Kalshi price chart — draws lines on a canvas for each series in the data
(function () {
  const SERIES_COLORS = ['#4a9eff', '#ff6b6b', '#4aff9e', '#ffdb4a', '#c74aff'];
  const GRID_COLOR = '#1a1a1a';
  const LABEL_COLOR = '#444';
  const PAD = { top: 12, right: 10, bottom: 24, left: 40 };

  let chartData = [];    // [{ts: Date, ...seriesValues}]
  let seriesKeys = [];   // dynamic keys from the data (everything except timestamp/ts)
  let minTs = 0, maxTs = 0;
  let minPrice = 0, maxPrice = 100;

  const chartCanvas = document.getElementById('kalshiChart');
  const chartCtx = chartCanvas.getContext('2d');
  const playhead = document.getElementById('chartPlayhead');

  // Fetch data and render
  fetch('/price-history')
    .then(r => r.json())
    .then(raw => {
      if (raw.length === 0) return;

      // Discover series keys dynamically (all keys except timestamp)
      seriesKeys = Object.keys(raw[0]).filter(k => k !== 'timestamp');

      chartData = raw.map(d => {
        const entry = { ts: new Date(d.timestamp).getTime() };
        for (const key of seriesKeys) {
          entry[key] = d[key];
        }
        return entry;
      });

      // Update Kalshi chart legend labels from data columns
      const legA = document.getElementById('kalshiLegA');
      const legB = document.getElementById('kalshiLegB');
      if (legA && seriesKeys[0]) legA.textContent = seriesKeys[0];
      if (legB && seriesKeys[1]) legB.textContent = seriesKeys[1];

      minTs = chartData[0].ts;
      maxTs = chartData[chartData.length - 1].ts;

      // Compute price range from actual data
      let lo = Infinity, hi = -Infinity;
      for (const d of chartData) {
        for (const key of seriesKeys) {
          if (d[key] != null) {
            lo = Math.min(lo, d[key]);
            hi = Math.max(hi, d[key]);
          }
        }
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

    for (let i = 0; i < seriesKeys.length; i++) {
      drawLine(seriesKeys[i], SERIES_COLORS[i % SERIES_COLORS.length]);
    }
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
