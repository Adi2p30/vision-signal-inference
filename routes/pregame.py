"""
POST /pregame — load a game into memory before tip-off.

Fetches team and game context, stores everything in Supermemory,
and returns the four container tags needed for /live calls.
"""

import asyncio

from fastapi import APIRouter, HTTPException

from supermemory_client import load_game
from models import PregameRequest, PregameResponse

router = APIRouter()


@router.post("/pregame", response_model=PregameResponse)
async def pregame(request: PregameRequest):
    """
    Pre-game setup. Call once before tip-off.

    Fetches ESPN stats, Barttorvik efficiency, and game context for both teams,
    then loads everything into Supermemory. Returns the tags you pass to /live.

    - **home_team**: Home team name (e.g. "Kansas")
    - **away_team**: Away team name (e.g. "Arizona")
    - **game_date**: Optional YYYYMMDD — defaults to today
    """
    if not request.home_team.strip() or not request.away_team.strip():
        raise HTTPException(status_code=400, detail="'home_team' and 'away_team' must not be empty.")

    try:
        tags = await asyncio.to_thread(
            load_game,
            request.home_team.strip(),
            request.away_team.strip(),
            request.game_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pregame setup failed: {e}")

    return PregameResponse(
        game_tag   = tags["game_tag"],
        h_tag      = tags["h_tag"],
        a_tag      = tags["a_tag"],
        e_tag      = tags["e_tag"],
        home_team  = request.home_team.strip(),
        away_team  = request.away_team.strip(),
    )
