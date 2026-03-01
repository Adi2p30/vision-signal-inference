"""
POST /backtest — replay a historical game and generate signals for each play.

Fetches ESPN play-by-play, loads plays into Supermemory, then replays them
sequentially: momentum analysis + Kalshi price lookup + trading signal per play.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, HTTPException

from historical_context_agent.tools.espn_tools import fetch_play_by_play
from supermemory_client import add_game_event, retrieve_context, synthesize_momentum
from kalshi_client import get_market_trend
from trading_agent import generate_trading_signal
import events_store
from models import BacktestRequest, BacktestResponse, BacktestPlay, MarketSignal

router = APIRouter()


# ---------------------------------------------------------------------------
# ESPN play helpers
# ---------------------------------------------------------------------------

def _clock_to_seconds(clock: str) -> int:
    try:
        m, s = clock.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def _parse_cutoff(cutoff: str) -> tuple[int, int]:
    c = cutoff.lower().strip()
    if "full" in c or "entire" in c or "whole" in c:
        return (99, 0)
    if "halftime" in c or "half time" in c:
        return (1, 0)
    period = 2 if ("2nd" in c or "second" in c) else 3 if ("ot" in c or "overtime" in c) else 1
    m = re.search(r"(\d+):(\d+)", c)
    secs = int(m.group(1)) * 60 + int(m.group(2)) if m else 0
    return (period, secs)


def _play_before_cutoff(play: dict, cutoff_period: int, cutoff_secs: int) -> bool:
    if cutoff_period == 99:
        return True
    p = play["period"]
    if p < cutoff_period:
        return True
    if p == cutoff_period:
        return _clock_to_seconds(play["clock"]) >= cutoff_secs
    return False


def _play_to_elapsed(play: dict) -> int:
    period    = play.get("period", 1)
    clock_sec = _clock_to_seconds(play.get("clock", "20:00"))
    if period == 1:
        return max(0, 1200 - clock_sec)
    if period == 2:
        return max(0, 1200 + (1200 - clock_sec))
    return max(0, 2400 + (period - 3) * 300 + (300 - clock_sec))


_PLAYER_RE    = re.compile(r"\b[A-Z][A-Za-z'-]*\.?\s+[A-Z][A-Za-z'-]+")
_SECONDARY_RE = re.compile(
    r"(Assisted by|Blocked by|Stolen by|Foul on)\s+[A-Z][A-Za-z'-]*\.?\s+[A-Z][A-Za-z'-]+",
    re.IGNORECASE,
)


def _format_play(play: dict, home_name: str, away_name: str, team_map: dict) -> str:
    label = team_map.get(play.get("team_id", ""), "")
    text  = play["text"]
    if label:
        text = _SECONDARY_RE.sub(lambda m: f"{m.group(1)} {label}", text)
        text = _PLAYER_RE.sub(label, text, count=1)
    score = f"{home_name} {play['home_score']} - {away_name} {play['away_score']}"
    return f"[{play['period_display']} {play['clock']}] {text} | {score}"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest):
    """
    Replay a historical game and generate a trading signal for each play.

    1. Fetches ESPN play-by-play up to the cutoff
    2. Loads every play into Supermemory (parallel)
    3. For each play: retrieves context, synthesizes momentum, fetches historical
       Kalshi price, and generates a trading signal

    - **game_tag / h_tag / a_tag / e_tag**: Tags from /pregame
    - **event_id**: ESPN game ID (found in ESPN game URLs)
    - **cutoff**: e.g. "halftime", "2nd half 5:30", "full game"
    - **market_tickers**: Kalshi tickers to generate signals for
    - **game_start_ts**: Optional unix timestamp of tip-off for historical price lookup
    """
    if not request.event_id.strip() or not request.market_tickers:
        raise HTTPException(status_code=400, detail="'event_id' and 'market_tickers' must not be empty.")

    # Fetch play-by-play from ESPN
    pbp = await asyncio.to_thread(fetch_play_by_play, request.event_id.strip())
    if "error" in pbp:
        raise HTTPException(status_code=404, detail=f"ESPN error: {pbp['error']}")

    home_team = pbp.get("home_team") or {}
    away_team = pbp.get("away_team") or {}
    home_name = home_team.get("name", "Home")
    away_name = away_team.get("name", "Away")
    team_map  = {
        str(home_team.get("id", "")): home_name,
        str(away_team.get("id", "")): away_name,
    }

    cutoff_period, cutoff_secs = _parse_cutoff(request.cutoff)

    # Filter plays to cutoff and format them
    filtered = [
        (play, _format_play(play, home_name, away_name, team_map))
        for play in pbp["plays"]
        if _play_before_cutoff(play, cutoff_period, cutoff_secs)
    ]

    # Load all plays into Supermemory in parallel
    def _store_all():
        args = [
            (request.e_tag, text, f"{p['period_display']} {p['clock']}", seq)
            for seq, (p, text) in enumerate(filtered)
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(add_game_event, *a) for a in args]
            for f in futs:
                f.result()

    if filtered:
        await asyncio.to_thread(_store_all)
        events_store.push_batch(request.e_tag, [text for _, text in filtered])

    # Replay each play sequentially (Gemini rate limits)
    tickers      = request.market_tickers
    result_plays: list[BacktestPlay] = []

    for seq, (play, play_text) in enumerate(filtered):
        game_time    = f"{play['period_display']} {play['clock']}"
        elapsed      = _play_to_elapsed(play)
        play_unix_ts: Optional[int] = (
            request.game_start_ts + elapsed if request.game_start_ts else None
        )

        recent = events_store.get_recent(request.e_tag)
        try:
            ctx      = await asyncio.to_thread(
                retrieve_context,
                request.game_tag, play_text,
                request.h_tag, request.a_tag, request.e_tag, recent,
            )
            analysis = await asyncio.to_thread(synthesize_momentum, play_text, ctx)
        except Exception as e:
            analysis = f"Analysis error: {e}"

        play_signals: list[MarketSignal] = []
        for t in tickers:
            trend: dict = {}
            if play_unix_ts:
                trend = await asyncio.to_thread(get_market_trend, t, play_unix_ts)
            signal = await asyncio.to_thread(
                generate_trading_signal, t, f"Market: {t}", trend, analysis
            )
            play_signals.append(MarketSignal(
                market_ticker = t,
                yes_price     = trend.get("current_price"),
                trend         = trend.get("trend", "unknown"),
                signal        = signal,
            ))

        result_plays.append(BacktestPlay(
            play_text      = play_text,
            game_time      = game_time,
            elapsed_secs   = elapsed,
            play_unix_ts   = play_unix_ts,
            analysis       = analysis,
            market_signals = play_signals,
        ))

    return BacktestResponse(
        event_id    = request.event_id,
        cutoff      = request.cutoff,
        total_plays = len(result_plays),
        plays       = result_plays,
    )
