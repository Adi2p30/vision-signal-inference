"""
POST /live — process one play and return trading signals.

Stores the event in memory, synthesizes momentum, fetches Kalshi price trends,
generates trading signals, and executes simulated trades — all in one call.
"""

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, HTTPException

from supermemory_client import add_game_event, retrieve_context, synthesize_momentum
from kalshi_client import get_market, get_market_trend
from trading_agent import generate_trading_signal
import positions_manager
import events_store
from models import LiveRequest, LiveResponse, MarketSignal

router = APIRouter()

_CONTRACTS_PER_TRADE = 10


def _parse_action(signal: str) -> str:
    m = re.search(r"ACTION:\s*(BUY YES|BUY NO|SELL YES|SELL NO|HOLD)", signal, re.IGNORECASE)
    return m.group(1).upper() if m else "HOLD"


@router.post("/live", response_model=LiveResponse)
async def live(request: LiveRequest):
    """
    Process a live game event and return trading signals.

    1. Stores the event in Supermemory and the local events buffer
    2. Retrieves context from all four memory containers in parallel
    3. Synthesizes a momentum assessment with Gemini
    4. Fetches the last 2 minutes of Kalshi price data for each ticker
    5. Generates a BUY/SELL/HOLD signal per ticker
    6. Executes simulated trades and updates positions.json

    - **game_tag / h_tag / a_tag / e_tag**: Tags returned by /pregame
    - **event**: What just happened (e.g. "Kansas made a 3-pointer, leading 21-14")
    - **game_time**: Game clock string (e.g. "Q2 14:32")
    - **sequence**: Monotonically increasing event counter
    - **market_tickers**: One or more Kalshi market tickers
    - **current_time**: Unix timestamp of this moment
    """
    if not request.event.strip():
        raise HTTPException(status_code=400, detail="'event' must not be empty.")
    if not request.market_tickers:
        raise HTTPException(status_code=400, detail="'market_tickers' must not be empty.")

    event   = request.event.strip()
    tickers = request.market_tickers
    t_start = time.time()
    print(f"[live] START event={event!r} tickers={tickers}", flush=True)

    # Step 1: store the event (Supermemory + local buffer)
    time_str = request.game_time or "?"
    entry    = f"[{time_str}] #{request.sequence:04d} {event}"
    try:
        await asyncio.to_thread(
            add_game_event, request.e_tag, event, request.game_time, request.sequence
        )
    except Exception as e:
        print(f"[live] add_game_event failed: {e}", flush=True)
    events_store.push_event(request.e_tag, entry)

    recent = events_store.get_recent(request.e_tag)
    print(f"[live] event stored, {len(recent)} in buffer — {time.time()-t_start:.1f}s", flush=True)

    # Step 2 (parallel): context + market trends + market titles + positions
    def _fetch_all():
        with ThreadPoolExecutor(max_workers=3 + len(tickers)) as pool:
            analysis_f  = pool.submit(
                lambda: synthesize_momentum(
                    event,
                    retrieve_context(
                        request.game_tag, event,
                        request.h_tag, request.a_tag, request.e_tag, recent,
                    ),
                )
            )
            trend_fs    = [pool.submit(get_market_trend, t, request.current_time) for t in tickers]
            market_fs   = [pool.submit(get_market, t) for t in tickers]
            positions_f = pool.submit(
                positions_manager.format_positions_context,
                tickers[0] if len(tickers) == 1 else "",
            )

            trends        = [f.result() for f in trend_fs]
            markets       = [f.result() for f in market_fs]
            positions_ctx = positions_f.result()
            analysis      = analysis_f.result()
            return analysis, trends, markets, positions_ctx

    try:
        analysis, trends, markets, positions_ctx = await asyncio.to_thread(_fetch_all)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context fetch failed: {e}")

    print(f"[live] context + trends done — {time.time()-t_start:.1f}s", flush=True)

    titles = [
        m.get("title", t) if isinstance(m, dict) and "error" not in m else t
        for t, m in zip(tickers, markets)
    ]

    # Step 3 (parallel): generate signals with Gemini
    positions_state = positions_manager.load()

    def _gen_signals():
        with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
            futs = [
                pool.submit(generate_trading_signal, t, title, trend, analysis, positions_ctx, positions_state)
                for t, title, trend in zip(tickers, titles, trends)
            ]
            return [f.result() for f in futs]

    signals = await asyncio.to_thread(_gen_signals)
    print(f"[live] signals done — {time.time()-t_start:.1f}s", flush=True)

    # Step 4: validate actions and execute trades
    market_signals: list[MarketSignal] = []
    current_positions = positions_manager.load().get("positions", {})

    for ticker, trend, signal in zip(tickers, trends, signals):
        price  = trend.get("current_price")
        action = _parse_action(signal)

        if action.startswith("SELL"):
            held      = current_positions.get(ticker)
            sell_side = "yes" if "YES" in action else "no"
            if not held or held.get("side") != sell_side or held.get("contracts", 0) <= 0:
                print(f"[live] {action} blocked — no {sell_side.upper()} held on {ticker}, overriding to HOLD", flush=True)
                action = "HOLD"

        order_id: Optional[str] = None
        if action != "HOLD" and price is not None:
            order_id = positions_manager.record_trade(ticker, action, _CONTRACTS_PER_TRADE, price)
            current_positions = positions_manager.load().get("positions", {})
            print(f"[live] {action} {_CONTRACTS_PER_TRADE}x {ticker} @ {price:.2f} → {order_id}", flush=True)

        market_signals.append(MarketSignal(
            market_ticker = ticker,
            yes_price     = price,
            trend         = trend.get("trend", "unknown"),
            signal        = signal,
            action_taken  = action,
            order_id      = order_id,
        ))

    print(f"[live] done — {time.time()-t_start:.1f}s total", flush=True)
    return LiveResponse(event=event, analysis=analysis, market_signals=market_signals)
