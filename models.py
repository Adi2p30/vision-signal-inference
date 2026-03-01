"""Pydantic request/response models."""

from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# /pregame
# ---------------------------------------------------------------------------

class PregameRequest(BaseModel):
    home_team: str
    away_team: str
    game_date: str | None = None     # YYYYMMDD, defaults to today


class PregameResponse(BaseModel):
    game_tag: str
    h_tag: str
    a_tag: str
    e_tag: str
    home_team: str
    away_team: str


# ---------------------------------------------------------------------------
# /live
# ---------------------------------------------------------------------------

class LiveRequest(BaseModel):
    game_tag: str
    h_tag: str
    a_tag: str
    e_tag: str
    event: str                       # what just happened (free text)
    game_time: str | None = None     # e.g. "Q2 14:32"
    sequence: int = 0
    market_tickers: list[str]        # one or more Kalshi tickers
    current_time: int                # unix timestamp of this moment


class MarketSignal(BaseModel):
    market_ticker: str
    yes_price: Optional[float] = None
    trend: str
    signal: str
    action_taken: Optional[str] = None
    order_id: Optional[str] = None


class LiveResponse(BaseModel):
    event: str
    analysis: str
    market_signals: list[MarketSignal]


# ---------------------------------------------------------------------------
# /backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    game_tag: str
    h_tag: str
    a_tag: str
    e_tag: str
    event_id: str                         # ESPN game ID
    cutoff: str                           # e.g. "1st half 12:00", "halftime", "full game"
    market_tickers: list[str]
    game_start_ts: Optional[int] = None   # unix timestamp of tipoff for price lookup


class BacktestPlay(BaseModel):
    play_text: str
    game_time: str
    elapsed_secs: int
    play_unix_ts: Optional[int] = None
    analysis: str
    market_signals: list[MarketSignal] = []


class BacktestResponse(BaseModel):
    event_id: str
    cutoff: str
    total_plays: int
    plays: list[BacktestPlay]
