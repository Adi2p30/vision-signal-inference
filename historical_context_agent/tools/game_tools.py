"""
Game-level context tools: venue info, head-to-head history, rivalry detection,
conference membership. Used by compile_game_context.
"""

import requests
from difflib import SequenceMatcher

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HistoricalContextAgent/1.0)",
    "Accept": "application/json",
}

# Known rivalry pairs (order-independent keyword matching → rivalry name)
_RIVALRY_MAP = [
    ({"duke", "north carolina"}, "Tobacco Road Rivalry"),
    ({"duke", "unc"}, "Tobacco Road Rivalry"),
    ({"kansas", "missouri"}, "Border War"),
    ({"kentucky", "louisville"}, "Battle for the Bluegrass"),
    ({"illinois", "indiana"}, "Illibuck Trophy"),
    ({"illinois", "iowa"}, "Heartland Trophy"),
    ({"michigan", "michigan state"}, "Battle for Michigan"),
    ({"indiana", "purdue"}, "Bucket Game"),
    ({"ucla", "usc"}, "Crosstown Classic"),
    ({"kansas", "kansas state"}, "Sunflower Showdown"),
    ({"north carolina", "nc state"}, "Tobacco Road Rivalry"),
    ({"arizona", "arizona state"}, "Duel in the Desert"),
    ({"ohio state", "michigan"}, "Big Ten Rivalry"),
    ({"florida", "florida state"}, "Sunshine State Rivalry"),
    ({"villanova", "georgetown"}, "Big East Rivalry"),
    ({"connecticut", "syracuse"}, "Big East Classic"),
    ({"gonzaga", "saint mary's"}, "WCC Rivalry"),
    ({"memphis", "tennessee"}, "Tennessee Rivalry"),
    ({"arkansas", "missouri"}, "Border War"),
    ({"texas", "oklahoma"}, "Red River Rivalry"),
    ({"byu", "utah"}, "Holy War"),
]


def _get(url: str, params: dict = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "url": url}


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fetch_venue_info(team_id: str) -> dict:
    """
    Fetch the home arena name and location for a team.

    Venue data lives in the schedule endpoint (not team detail).
    Scans the first home game to extract venue info.

    Args:
        team_id: ESPN team ID.

    Returns:
        dict with venue_name, city, state. 'error' key on failure.
    """
    data = _get(f"{ESPN_BASE}/teams/{team_id}/schedule")
    if "error" in data:
        return data

    for event in data.get("events", []):
        competitions = event.get("competitions", [{}])
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        subject = next((c for c in competitors if c.get("id") == team_id), None)
        if not subject or subject.get("homeAway") != "home":
            continue
        venue = comp.get("venue", {})
        address = venue.get("address", {})
        if venue.get("fullName"):
            return {
                "venue_name": venue["fullName"],
                "city": address.get("city", ""),
                "state": address.get("state", ""),
            }

    return {"venue_name": "Unknown Arena", "city": "", "state": ""}


def fetch_head_to_head(home_team_id: str, away_team_id: str, away_display: str) -> dict:
    """
    Scan the home team's current-season schedule for completed games against
    the away team and return the head-to-head record and results.

    Args:
        home_team_id: ESPN ID of the home team.
        away_team_id: ESPN ID of the away team.
        away_display: Display name of the away team (used as fallback for name matching).

    Returns:
        dict with h2h_games list, h2h_wins, h2h_losses, h2h_record string.
    """
    data = _get(f"{ESPN_BASE}/teams/{home_team_id}/schedule")
    if "error" in data:
        return {"h2h_games": [], "h2h_record": "No data", "h2h_wins": 0, "h2h_losses": 0}

    h2h_games = []

    for event in data.get("events", []):
        competitions = event.get("competitions", [{}])
        if not competitions:
            continue
        comp = competitions[0]

        if comp.get("status", {}).get("type", {}).get("name", "") != "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        subject  = next((c for c in competitors if c.get("id") == home_team_id), {})
        opponent = next((c for c in competitors if c.get("id") != home_team_id), {})

        if not subject or not opponent:
            continue

        opp_id   = opponent.get("id", "")
        opp_name = opponent.get("team", {}).get("displayName", "")

        if opp_id != away_team_id and _similarity(opp_name, away_display) < 0.7:
            continue

        won = subject.get("winner", False)
        s   = subject.get("score")
        o   = opponent.get("score")
        sv  = (s or {}).get("value") if isinstance(s, dict) else s
        ov  = (o or {}).get("value") if isinstance(o, dict) else o
        score_str = f"{int(sv)}–{int(ov)}" if sv is not None and ov is not None else "N/A"
        is_home = subject.get("homeAway") == "home"

        h2h_games.append({
            "date": event.get("date", "")[:10],
            "result": "W" if won else "L",
            "score": score_str,
            "location": "Home" if is_home else "Away",
        })

    wins   = sum(1 for g in h2h_games if g["result"] == "W")
    losses = len(h2h_games) - wins

    return {
        "h2h_games": h2h_games,
        "h2h_wins": wins,
        "h2h_losses": losses,
        "h2h_record": f"{wins}-{losses}" if h2h_games else "No H2H this season",
    }


def detect_rivalry(home_name: str, away_name: str, same_conference: bool, h2h_count: int) -> str | None:
    """
    Return a rivalry name if the two teams are known rivals, or None.

    Checks hardcoded rivalry pairs first, then infers from same-conference
    teams that have met 2+ times in the current season (conf tournament rematch).
    """
    home_lower = home_name.lower()
    away_lower = away_name.lower()

    for keywords, name in _RIVALRY_MAP:
        keys = list(keywords)
        a, b = keys[0], keys[1]
        if (a in home_lower or a in away_lower) and (b in home_lower or b in away_lower):
            return name

    if same_conference and h2h_count >= 2:
        return "Conference Rivalry"

    return None


def compute_matchup_type(home_net_eff: float, away_net_eff: float) -> str:
    """
    Classify the matchup based on net efficiency gap (adjOE - adjDE).
    Gap is from the perspective of the favored team.
    """
    gap = abs(home_net_eff - away_net_eff)
    if gap >= 12:
        return "heavy_favorite"
    elif gap >= 6:
        return "moderate_favorite"
    elif gap >= 3:
        return "slight_favorite"
    else:
        return "even"
