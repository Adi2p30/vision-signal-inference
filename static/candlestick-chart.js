// Candlestick chart for Kalshi yes_bid / yes_ask OHLC data
(function () {
  const BID_UP = '#4a9eff';
  const BID_DOWN = '#2a6ecc';
  const ASK_UP = '#ff6b6b';
  const ASK_DOWN = '#cc3a3a';
  const GRID_COLOR = '#1a1a1a';
  const LABEL_COLOR = '#444';
  const PAD = { top: 12, right: 10, bottom: 24, left: 40 };

  const canvas = document.getElementById('candlestickChart');
  const ctx = canvas.getContext('2d');

  let candles = [];
  let minTs = 0, maxTs = 0;
  let minPrice = 0, maxPrice = 100;

  // Toggle state: which series are visible
  const visible = { bid: false, ask: true };

  // Wire up legend clicks
  const legendEls = canvas.parentElement.querySelectorAll('.leg');
  function updateLegendStyle() {
    legendEls[0].style.opacity = visible.bid ? 1 : 0.35;
    legendEls[1].style.opacity = visible.ask ? 1 : 0.35;
  }
  legendEls[0].style.cursor = 'pointer';
  legendEls[1].style.cursor = 'pointer';
  legendEls[0].addEventListener('click', () => { visible.bid = !visible.bid; updateLegendStyle(); recomputeRange(); draw(); });
  legendEls[1].addEventListener('click', () => { visible.ask = !visible.ask; updateLegendStyle(); recomputeRange(); draw(); });
  updateLegendStyle();

  function recomputeRange() {
    if (candles.length === 0) return;
    let lo = Infinity, hi = -Infinity;
    for (const c of candles) {
      if (visible.bid) { lo = Math.min(lo, c.bid.low); hi = Math.max(hi, c.bid.high); }
      if (visible.ask) { lo = Math.min(lo, c.ask.low); hi = Math.max(hi, c.ask.high); }
    }
    if (lo === Infinity) { lo = 0; hi = 100; }
    minPrice = Math.max(0, Math.floor(lo - 2));
    maxPrice = Math.min(100, Math.ceil(hi + 2));
  }

  fetch('/candlesticks')
    .then(r => r.json())
    .then(raw => {
      if (!raw || raw.length === 0) return;

      candles = raw.map(d => ({
        ts: d.end_period_ts * 1000,
        bid: d.yes_bid,
        ask: d.yes_ask,
        volume: d.volume,
      }));

      minTs = candles[0].ts;
      maxTs = candles[candles.length - 1].ts;

      recomputeRange();
      draw();
      window.addEventListener('resize', draw);
    });

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;

    ctx.clearRect(0, 0, w, h);
    if (candles.length === 0) return;

    const xOf = ts => PAD.left + ((ts - minTs) / (maxTs - minTs)) * plotW;
    const yOf = price => PAD.top + (1 - (price - minPrice) / (maxPrice - minPrice)) * plotH;

    // Grid lines & price labels
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;
    ctx.font = '9px monospace';
    ctx.fillStyle = LABEL_COLOR;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const priceStep = maxPrice - minPrice > 40 ? 20 : 10;
    for (let p = Math.ceil(minPrice / priceStep) * priceStep; p <= maxPrice; p += priceStep) {
      const y = yOf(p);
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(w - PAD.right, y);
      ctx.stroke();
      ctx.fillText(p + '\u00a2', PAD.left - 4, y);
    }

    // Time labels along bottom
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const totalMs = maxTs - minTs;
    const labelCount = Math.min(6, Math.floor(plotW / 70));
    for (let i = 0; i <= labelCount; i++) {
      const ts = minTs + (totalMs * i) / labelCount;
      const d = new Date(ts);
      const label = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      ctx.fillText(label, xOf(ts), h - PAD.bottom + 6);
    }

    // Compute candle width
    const avgGap = plotW / candles.length;
    const both = visible.bid && visible.ask;
    const candleW = Math.max(1, Math.min(avgGap * (both ? 0.35 : 0.6), 6));
    const offset = both ? candleW + 1 : 0;

    // Draw candlesticks
    for (const c of candles) {
      const cx = xOf(c.ts);
      if (both) {
        drawCandle(cx - offset / 2, c.bid, BID_UP, BID_DOWN, candleW);
        drawCandle(cx + offset / 2, c.ask, ASK_UP, ASK_DOWN, candleW);
      } else if (visible.bid) {
        drawCandle(cx, c.bid, BID_UP, BID_DOWN, candleW);
      } else if (visible.ask) {
        drawCandle(cx, c.ask, ASK_UP, ASK_DOWN, candleW);
      }
    }

    function drawCandle(x, ohlc, upColor, downColor, w) {
      const isUp = ohlc.close >= ohlc.open;
      const color = isUp ? upColor : downColor;

      // Wick (high-low line)
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, yOf(ohlc.high));
      ctx.lineTo(x, yOf(ohlc.low));
      ctx.stroke();

      // Body (open-close rect)
      const yTop = yOf(Math.max(ohlc.open, ohlc.close));
      const yBot = yOf(Math.min(ohlc.open, ohlc.close));
      const bodyH = Math.max(1, yBot - yTop);

      ctx.fillStyle = color;
      ctx.fillRect(x - w / 2, yTop, w, bodyH);
    }
  }
})();
