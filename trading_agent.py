"""
Trading signal agent.

Inputs:
  - trend    : last 2 min of Kalshi YES prices from get_market_trend()
  - analysis : game momentum output from synthesize_momentum() (match-analysis)

Output:
  BUY YES / BUY NO / HOLD with confidence and reasoning.
"""

import os
import re
import time as _time

_PROMPT = """\
You are a profit-maximizing quantitative sports prediction market trader.
Your goal is to maximize total P&L — let winning positions run, cut only genuine losers.

--- KALSHI MARKET (last 2 minutes of price data) ---
Ticker        : {ticker}
Market        : {title}
Current price : {current_price}
2 min ago     : {open_price}
Change        : {price_change}
Direction     : {trend}
Trades seen   : {trades_in_window}

--- GAME MOMENTUM (match analysis) ---
{analysis}

--- YOUR CURRENT POSITIONS & TRADE HISTORY ---
{positions_context}
Unrealized P&L on this ticker: {unrealized_pnl}

--- DECISION RULES ---

BUY YES  — only when:
  • Momentum strongly favors YES team (+) AND price is lagging (flat or falling)
  • Price edge ≥ 0.06 (current price at least 6¢ below fair value)
  • You have buying power remaining

BUY NO   — only when:
  • Momentum strongly favors NO team (+) AND price is inflated (flat or rising)
  • Price edge ≥ 0.06
  • You have buying power remaining

SELL YES — only when ALL THREE are true:
  1. You actually hold YES contracts (check positions above)
  2. Momentum has CLEARLY and STRONGLY reversed against YES (not just a pause)
  3. Either: unrealized P&L is positive (lock profit) OR momentum reversal is HIGH confidence (cut loss)
  — Do NOT sell on weak or uncertain momentum shifts
  — Do NOT sell just because the price dipped slightly

SELL NO  — same rules as SELL YES but for NO contracts

HOLD     — default when:
  • Momentum is neutral or mixed
  • Market price already reflects momentum
  • You hold a position and momentum has NOT clearly reversed (let it run)
  • Price edge is too small (< 0.06) to justify a new entry
  • Buying power is $0.00 and you have no position to sell

HARD RULES:
- NEVER sell a side you have 0 contracts in
- NEVER buy if buying power is $0.00
- PREFER HOLD over a premature sell — a good position held is better than an early exit
- Only sell when the game situation has materially changed against your position

Confidence:
  HIGH   — momentum strongly favors one team (+) AND price edge > 0.08
  MEDIUM — moderate momentum (~) OR edge 0.04–0.08
  LOW    — neutral momentum, conflicting signals, or edge < 0.04

Respond in EXACTLY this format, no other text:

ACTION: BUY YES | BUY NO | SELL YES | SELL NO | HOLD
CONFIDENCE: HIGH | MEDIUM | LOW
YES_PRICE: {current_price}
TREND: {trend} ({price_change} over 2 min)
REASONING: [1-2 sentences: what the market is doing vs what the game is doing, and why this action maximizes profit — cite the player's stats if a key player drove this play]
EDGE: [specific price discrepancy, or "No edge identified"]\
"""


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if v > 1.0:
            v /= 100.0
        return f"{v:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fallback(analysis: str) -> str:
    action = "HOLD"
    m = re.search(r"SIGNAL:\s*(BUY\s+YES|BUY\s+NO|HOLD)", analysis, re.IGNORECASE)
    if m:
        action = m.group(1).upper()
    return (
        f"ACTION: {action}\n"
        f"CONFIDENCE: LOW\n"
        f"YES_PRICE: N/A\n"
        f"TREND: unknown\n"
        f"REASONING: Gemini unavailable — derived from match-analysis SIGNAL line.\n"
        f"EDGE: No edge identified"
    )


def generate_trading_signal(
    ticker: str,
    title: str,
    trend: dict,
    analysis: str,
    positions_context: str = "",
    positions_state: dict | None = None,
) -> str:
    """
    Generate a BUY YES / BUY NO / SELL YES / SELL NO / HOLD signal.

    Args:
        ticker:           Kalshi market ticker.
        title:            Market display title.
        trend:            Output of get_market_trend() — price trend over last 2 min.
        analysis:         Output of synthesize_momentum() — game momentum analysis.
        positions_context: Formatted positions summary string.
        positions_state:  Raw positions dict for computing unrealized P&L.

    Returns:
        Structured signal string.
    """
    try:
        from google import genai
    except ImportError:
        return _fallback(analysis)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return _fallback(analysis)

    change = trend.get("price_change")
    change_str = f"{change:+.3f}" if change is not None else "N/A"
    current_price = trend.get("current_price")

    # Compute unrealized P&L for this ticker if we have a position
    unrealized_pnl = "No position"
    if positions_state and current_price is not None:
        held = positions_state.get("positions", {}).get(ticker)
        if held:
            avg  = held.get("avg_price", 0.0)
            qty  = held.get("contracts", 0)
            side = held.get("side", "yes")
            # For YES contracts: profit when price rises. For NO contracts: profit when price falls.
            price_move = (current_price - avg) if side == "yes" else (avg - current_price)
            pnl = round(price_move * qty, 4)
            sign = "+" if pnl >= 0 else ""
            unrealized_pnl = f"{sign}${pnl:.2f} ({sign}{price_move*100:.1f}¢/contract on {qty} {side.upper()} contracts)"

    prompt = _PROMPT.format(
        ticker            = ticker,
        title             = title,
        current_price     = _fmt(current_price),
        open_price        = _fmt(trend.get("open_price")),
        price_change      = change_str,
        trend             = trend.get("trend", "unknown"),
        trades_in_window  = trend.get("trades_in_window", 0),
        analysis          = analysis,
        positions_context = positions_context or "No open positions. Buying power unknown.",
        unrealized_pnl    = unrealized_pnl,
    )

    try:
        t0 = _time.time()
        print(f"[trading_agent] calling Gemini for signal — ticker={ticker}", flush=True)
        client = genai.Client(api_key=api_key)
        resp   = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        print(f"[trading_agent] Gemini signal done — {_time.time()-t0:.1f}s", flush=True)
        return resp.text.strip()
    except Exception as e:
        print(f"[trading_agent] Gemini signal failed: {e}", flush=True)
        return _fallback(analysis)
