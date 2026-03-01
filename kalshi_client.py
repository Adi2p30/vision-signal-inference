"""
Kalshi market data client.

Auth (production): RSA-PSS signed request headers
  kalshi_apiKey       — API key
  kalshi_private_key  — PEM-encoded RSA private key for signing

Auth (demo): Bearer token
  kalshi_apiKey       — API key (Bearer)
  KALSHI_USE_DEMO=1   — set to use demo environment

All functions return empty / None gracefully when credentials are absent.
"""

import base64
import os
import re
import time
from typing import Optional
from urllib.parse import urlparse

import requests

_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"

_session: Optional[requests.Session] = None

# ── TTL caches for market data ───────────────────────────────────────────────
_MARKET_CACHE: dict[str, tuple[float, dict]] = {}
_MARKET_TTL = 30  # seconds — market info (title, status) changes slowly

_TREND_CACHE: dict[str, tuple[float, dict]] = {}
_TREND_TTL = 10   # seconds — price trends tolerate a short stale window


def _base_url() -> str:
    return _DEMO_BASE if os.environ.get("KALSHI_USE_DEMO") else _PROD_BASE


def _build_session() -> Optional[requests.Session]:
    api_key = os.environ.get("kalshi_apiKey", "")
    if not api_key:
        return None

    session = requests.Session()

    if os.environ.get("KALSHI_USE_DEMO"):
        session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
    else:
        private_key_pem = os.environ.get("kalshi_private_key", "")
        session.auth = _RSAPSSAuth(api_key, private_key_pem)

    return session


class _RSAPSSAuth(requests.auth.AuthBase):
    """Attach RSA-PSS signed headers to every request for Kalshi production API."""

    def __init__(self, api_key: str, private_key_pem: str) -> None:
        self.api_key = api_key
        self._private_key = None
        if private_key_pem:
            try:
                from cryptography.hazmat.primitives.serialization import load_pem_private_key
                pem = private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem
                self._private_key = load_pem_private_key(pem, password=None)
            except Exception as e:
                print(f"[kalshi] Could not load RSA private key: {e}")

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        ts_ms = str(int(time.time() * 1000))
        method = (r.method or "GET").upper()

        parsed = urlparse(r.url or "")
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"

        r.headers["KALSHI-ACCESS-KEY"] = self.api_key
        r.headers["KALSHI-ACCESS-TIMESTAMP"] = ts_ms
        r.headers["Content-Type"] = "application/json"

        if self._private_key is not None:
            try:
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                msg = (ts_ms + method + path).encode("utf-8")
                sig = self._private_key.sign(
                    msg,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH,
                    ),
                    hashes.SHA256(),
                )
                r.headers["KALSHI-ACCESS-SIGNATURE"] = base64.b64encode(sig).decode()
            except Exception as e:
                print(f"[kalshi] RSA signing failed: {e}")
        return r


def _get_session() -> Optional[requests.Session]:
    global _session
    if _session is None:
        _session = _build_session()
    return _session


def _get(path: str, params: Optional[dict] = None, silent: bool = False) -> Optional[dict]:
    """Authenticated GET. Returns parsed JSON or None on any failure."""
    session = _get_session()
    if session is None:
        print("[kalshi] No credentials configured (set kalshi_apiKey)")
        return None

    url = f"{_base_url()}{path}"
    try:
        resp = session.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        if not silent or "/trades" in path:
            print(f"[kalshi] {path} params={params} → HTTP {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[kalshi] {path} → exception: {e}")
    return None


def _normalize_market(raw: dict) -> dict:
    return {
        "ticker":     raw.get("ticker", ""),
        "title":      raw.get("title", ""),
        "subtitle":   raw.get("subtitle", ""),
        "yes_bid":    raw.get("yes_bid"),
        "yes_ask":    raw.get("yes_ask"),
        "last_price": raw.get("last_price"),
        "status":     raw.get("status", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_markets(query: str, series_ticker: str = "KXNCAAMBGAME") -> list[dict]:
    """
    Search Kalshi markets by title/ticker substring within a series.

    Args:
        query:         Substring to match (case-insensitive).
        series_ticker: Series to search (default: KXNCAAMBGAME for NCAA basketball games).

    Returns:
        List of normalized market dicts. Empty list if unavailable.
    """
    data = _get("/markets", params={"series_ticker": series_ticker, "limit": 200})
    if not data:
        return []

    q = query.lower()
    results = []
    for m in data.get("markets", []):
        if q in (m.get("title") or "").lower() or q in (m.get("ticker") or "").lower():
            results.append(_normalize_market(m))
    return results


def get_market(ticker: str) -> dict:
    """
    Fetch a single market by ticker, enriched with orderbook midpoint price.
    Results are cached for 30 s to avoid redundant API calls.

    Returns normalized market dict or {"error": "..."}.
    """
    if ticker in _MARKET_CACHE:
        ts, cached = _MARKET_CACHE[ticker]
        if time.time() - ts < _MARKET_TTL:
            return cached

    data = _get(f"/markets/{ticker}")
    if data is None:
        return {"error": "Kalshi unavailable or no credentials"}

    raw = data.get("market", data)
    if not raw or not raw.get("ticker"):
        return {"error": f"Market {ticker!r} not found"}

    market = _normalize_market(raw)

    ob = _get(f"/markets/{ticker}/orderbook")
    if ob:
        orderbook = ob.get("orderbook", {})
        yes_side = orderbook.get("yes", [])
        bids = [o["price"] for o in yes_side if o.get("side") == "bid"]
        asks = [o["price"] for o in yes_side if o.get("side") == "ask"]
        if not bids and not asks:
            bids = [o["price"] for o in yes_side if o.get("is_buy")]
            asks = [o["price"] for o in yes_side if not o.get("is_buy")]
        if bids:
            market["yes_bid"] = max(bids)
        if asks:
            market["yes_ask"] = min(asks)
        if bids and asks:
            market["last_price"] = (max(bids) + min(asks)) / 2

    _MARKET_CACHE[ticker] = (time.time(), market)
    return market


def _series_from_ticker(ticker: str) -> str:
    """Extract series ticker from a market ticker, e.g. KXNCAAMBGAME-26FEB09ARIZKU-ARIZ → KXNCAAMBGAME."""
    m = re.match(r"^([A-Z]+)-\d{2}[A-Z]{3}\d{2}", ticker)
    return m.group(1) if m else ticker.split("-")[0]


def _candlestick_price(ticker: str, unix_ts: int, lookback_mins: int) -> Optional[tuple[float, float]]:
    """
    Fetch 1-minute candlesticks for the window [unix_ts - lookback_mins*60, unix_ts].

    Returns (open_price, close_price) as 0.0–1.0 floats, or None if unavailable.
    """
    series = _series_from_ticker(ticker)
    start_ts = unix_ts - lookback_mins * 60
    print(f"[kalshi] candlestick fallback: series={series} ticker={ticker} window=[{start_ts}, {unix_ts}]")
    data = _get(
        f"/series/{series}/markets/{ticker}/candlesticks",
        params={"start_ts": start_ts, "end_ts": unix_ts, "period_interval": 1},
    )
    if not data:
        return None
    candles = data.get("candlesticks", [])
    print(f"[kalshi] candlestick fallback → {len(candles)} candles")
    if not candles:
        return None

    def _cents_to_float(v) -> Optional[float]:
        try:
            p = float(v)
            return p / 100.0 if p > 1.0 else p
        except (TypeError, ValueError):
            return None

    first, last = candles[0], candles[-1]
    open_price  = _cents_to_float(first.get("price", {}).get("open"))
    close_price = _cents_to_float(last.get("price", {}).get("close"))
    print(f"[kalshi] candlestick prices: open={open_price} close={close_price}")
    return (open_price, close_price) if close_price is not None else None


def get_market_trend(ticker: str, unix_ts: int, lookback_mins: int = 2) -> dict:
    """
    Return YES price trend in the window ending at unix_ts.
    Results are cached for 10 s to avoid redundant API calls.

    Returns:
        {current_price, open_price, price_change, trend, trades_in_window}
        or {} if no data.
    """
    if not unix_ts:
        return {}

    cache_key = f"{ticker}:{unix_ts}:{lookback_mins}"
    if cache_key in _TREND_CACHE:
        ts, cached = _TREND_CACHE[cache_key]
        if time.time() - ts < _TREND_TTL:
            return cached

    min_ts = unix_ts - lookback_mins * 60
    print(f"[kalshi] get_market_trend ticker={ticker} window=[{min_ts}, {unix_ts}] ({lookback_mins}min lookback)")
    data = _get(
        f"/markets/{ticker}/trades",
        params={"min_ts": min_ts, "max_ts": unix_ts, "limit": 100},
        silent=True,
    )
    if not data:
        print(f"[kalshi] get_market_trend → trades 404, trying candlestick fallback for {ticker}")
        prices = _candlestick_price(ticker, unix_ts, lookback_mins)
        if prices:
            open_price, close_price = prices
            change = round(close_price - open_price, 3)
            result = {
                "current_price":    close_price,
                "open_price":       open_price,
                "price_change":     change,
                "trend":            "rising" if change > 0.01 else "falling" if change < -0.01 else "flat",
                "trades_in_window": 0,
            }
            _TREND_CACHE[cache_key] = (time.time(), result)
            return result
        return {}

    trades = data.get("trades", [])
    print(f"[kalshi] get_market_trend → {len(trades)} trades in window")
    if not trades:
        return {}

    def _norm(p) -> Optional[float]:
        try:
            v = float(p)
            return v / 100.0 if v > 1.0 else v
        except (TypeError, ValueError):
            return None

    current_price = _norm(trades[0].get("yes_price"))
    open_price    = _norm(trades[-1].get("yes_price"))

    if current_price is None:
        return {}

    result: dict = {"current_price": current_price, "trades_in_window": len(trades)}

    if open_price is not None:
        change = round(current_price - open_price, 3)
        result["open_price"]   = open_price
        result["price_change"] = change
        result["trend"] = "rising" if change > 0.01 else "falling" if change < -0.01 else "flat"
    else:
        result["trend"] = "flat"

    _TREND_CACHE[cache_key] = (time.time(), result)
    return result
