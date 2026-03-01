"""
ESPN unofficial API tools for fetching college basketball data.
No API key required — these are the same endpoints ESPN's own website uses.
"""

import requests
from difflib import SequenceMatcher

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HistoricalContextAgent/1.0)",
    "Accept": "application/json",
}


def _get(url: str, params: dict = None) -> dict:
    """Safe GET with timeout and error handling."""
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "url": url}


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_team_by_name(team_name: str) -> dict:
    """
    Search ESPN for a college basketball team and return their ESPN team ID,
    full name, abbreviation, and official current-season record (overall, home, road).

    Use this as the first step for every team — the returned team_id is needed
    by all other ESPN tools.

    Args:
        team_name: The college basketball team name (e.g., "Illinois", "Duke Blue Devils",
                   "Kentucky Wildcats"). Partial names and nicknames are supported.

    Returns:
        dict with keys: team_id, display_name, abbreviation, overall_record, home_record,
                        road_record, or 'error' if not found.
    """
    data = _get(f"{ESPN_BASE}/teams", params={"limit": 900})
    if "error" in data:
        return data

    teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

    best_match = None
    best_score = 0.0

    for entry in teams:
        team = entry.get("team", {})
        candidates = [
            team.get("displayName", ""),
            team.get("shortDisplayName", ""),
            team.get("name", ""),
            team.get("location", ""),
            team.get("abbreviation", ""),
        ]
        score = max(_similarity(team_name, c) for c in candidates if c)
        if any(team_name.lower() in c.lower() for c in candidates if c):
            score = max(score, 0.9)
        if score > best_score:
            best_score = score
            best_match = team

    if not best_match or best_score < 0.4:
        return {"error": f"Team '{team_name}' not found on ESPN (best score: {best_score:.2f})"}

    team_id = best_match.get("id")

    # Fetch official record from team detail endpoint (more accurate than schedule math)
    detail = _get(f"{ESPN_BASE}/teams/{team_id}")
    overall_record = home_record = road_record = "N/A"

    record_items = detail.get("team", {}).get("record", {}).get("items", [])
    for item in record_items:
        rtype = item.get("type", "")
        summary = item.get("summary", "")
        if rtype == "total":
            overall_record = summary
        elif rtype == "home":
            home_record = summary
        elif rtype == "road":
            road_record = summary

    return {
        "team_id": team_id,
        "display_name": best_match.get("displayName"),
        "short_name": best_match.get("shortDisplayName"),
        "abbreviation": best_match.get("abbreviation"),
        "location": best_match.get("location"),
        "nickname": best_match.get("name"),
        "overall_record": overall_record,
        "home_record": home_record,
        "road_record": road_record,
        "match_confidence": round(best_score, 3),
    }


def fetch_team_statistics(team_id: str) -> dict:
    """
    Fetch current-season team statistics from ESPN and return possession-level
    efficiency and volatility features. Focuses on rates and derived metrics
    rather than raw per-game totals.

    Args:
        team_id: ESPN team ID (obtained from find_team_by_name).

    Returns:
        dict with possession-level features:
          - three_point_attempt_rate_pct: % of FGA that are 3s (volatility signal)
          - three_point_pct: 3P shooting efficiency
          - to_rate_per_100: turnovers per 100 possessions (possession waste rate)
          - ft_rate: FTA / FGA (how often team gets to the line / draws fouls)
          - ft_pct: free throw shooting %
          - offensive_rebound_rate_pct: OR / (OR + DR) — second-chance frequency
          - defensive_rebound_rate_pct: DR / (OR + DR)
          - fouls_per_game: team foul rate
          - blocks_per_game: rim protection
          - steals_per_game: live-ball turnover generation
          - effective_fg_pct, true_shooting_pct: shooting quality
        Returns 'error' key on failure.
    """
    data = _get(f"{ESPN_BASE}/teams/{team_id}/statistics")
    if "error" in data:
        return data

    results_block = data.get("results", {})
    categories = (
        results_block.get("stats", {}).get("categories", [])
        or results_block.get("splits", {}).get("categories", [])
    )

    # Flatten all stats into a single dict by name
    raw = {}
    for cat in categories:
        for stat in cat.get("stats", []):
            name = stat.get("name")
            val = stat.get("value")
            if name and val is not None:
                raw[name] = val

    # Extract key values
    pts   = raw.get("avgPoints", 0)
    fgm   = raw.get("avgFieldGoalsMade", 0)
    fga   = raw.get("avgFieldGoalsAttempted", 0)
    fg3m  = raw.get("avgThreePointFieldGoalsMade", 0)
    fg3a  = raw.get("avgThreePointFieldGoalsAttempted", 0)
    ftm   = raw.get("avgFreeThrowsMade", 0)
    fta   = raw.get("avgFreeThrowsAttempted", 0)
    reb   = raw.get("avgRebounds", 0)
    oreb  = raw.get("avgOffensiveRebounds", 0)
    dreb  = raw.get("avgDefensiveRebounds", 0)
    ast   = raw.get("avgAssists", 0)
    tov   = raw.get("avgTurnovers", 0)
    stl   = raw.get("avgSteals", 0)
    blk   = raw.get("avgBlocks", 0)
    fouls = raw.get("avgFouls", 0)
    gp    = int(raw.get("gamesPlayed", 0))

    # Shooting quality
    fg3_pct   = round(fg3m / fg3a * 100, 1) if fg3a else None
    fg3a_rate = round(fg3a / fga * 100, 1)  if fga else None
    ft_pct    = round(ftm / fta * 100, 1)   if fta else None
    ft_rate   = round(fta / fga, 3)          if fga else None   # FTA per FGA
    efg_pct   = round((fgm + 0.5 * fg3m) / fga * 100, 1) if fga else None
    ts_denom  = 2 * (fga + 0.44 * fta)
    ts_pct    = round(pts / ts_denom * 100, 1) if ts_denom else None

    # Turnover rate per 100 possessions  (standard formula: TOV / (FGA + 0.44*FTA + TOV))
    poss_denom = fga + 0.44 * fta + tov
    to_rate    = round(tov / poss_denom * 100, 1) if poss_denom else None

    # Rebounding rates (within-team split; OR% = OR share of team's total boards)
    total_reb  = oreb + dreb
    or_rate    = round(oreb / total_reb * 100, 1) if total_reb else None
    dr_rate    = round(dreb / total_reb * 100, 1) if total_reb else None

    return {
        "team_id": team_id,
        "games_played": gp,
        # Possession profile
        "three_point_attempt_rate_pct": fg3a_rate,
        "three_point_pct": fg3_pct,
        "to_rate_per_100_poss": to_rate,
        "ft_rate_fta_per_fga": ft_rate,
        "ft_pct": ft_pct,
        # Rebounding rates
        "offensive_rebound_rate_pct": or_rate,
        "defensive_rebound_rate_pct": dr_rate,
        "offensive_rebounds_per_game": round(oreb, 1),
        "defensive_rebounds_per_game": round(dreb, 1),
        # Foul / disruption
        "fouls_per_game": round(fouls, 1),
        "blocks_per_game": round(blk, 1),
        "steals_per_game": round(stl, 1),
        # Shooting quality (context for adjOE)
        "effective_fg_pct": efg_pct,
        "true_shooting_pct": ts_pct,
    }


def fetch_team_schedule(team_id: str, limit: int = 10) -> dict:
    """
    Fetch a team's completed game results to analyze recent form and home/away splits.

    Only STATUS_FINAL games are included — scheduled future games are excluded so
    W/L tallies and form strings are always accurate.

    Args:
        team_id: ESPN team ID.
        limit: Number of most recent completed games to return (default 10).

    Returns:
        dict with last_N_games list, last_5_form string (e.g. "W L W W L"),
        and home/away split from completed games only.
    """
    data = _get(f"{ESPN_BASE}/teams/{team_id}/schedule")
    if "error" in data:
        return data

    completed_games = []

    for event in data.get("events", []):
        competitions = event.get("competitions", [{}])
        if not competitions:
            continue
        comp = competitions[0]

        # Skip any game that is not finalized
        status_name = comp.get("status", {}).get("type", {}).get("name", "")
        if status_name != "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        subject  = next((c for c in competitors if c.get("id") == team_id), {})
        opponent = next((c for c in competitors if c.get("id") != team_id), {})

        if not subject:
            continue

        is_home = subject.get("homeAway") == "home"
        won     = subject.get("winner", False)

        score_val     = (subject.get("score") or {}).get("value") if isinstance(subject.get("score"), dict) else subject.get("score")
        opp_score_val = (opponent.get("score") or {}).get("value") if isinstance(opponent.get("score"), dict) else opponent.get("score")

        score_str = (
            f"{int(score_val)}–{int(opp_score_val)}"
            if score_val is not None and opp_score_val is not None
            else "N/A"
        )

        completed_games.append({
            "date": event.get("date", "")[:10],
            "opponent": opponent.get("team", {}).get("displayName", "Unknown"),
            "home_or_away": "Home" if is_home else "Away",
            "result": "W" if won else "L",
            "score": score_str,
        })

    recent = completed_games[-limit:]
    last_5 = completed_games[-5:]

    last_5_wins  = sum(1 for g in last_5 if g["result"] == "W")
    last_5_form  = " ".join(g["result"] for g in last_5)

    return {
        "team_id": team_id,
        "last_5_form": last_5_form,
        "last_5_record": f"{last_5_wins}-{5 - last_5_wins}",
        "last_5_games": [
            f"{g['result']} {g['score']} {'vs' if g['home_or_away'] == 'Home' else '@'} {g['opponent']} ({g['date']})"
            for g in last_5
        ],
        "recent_games": recent,
        "note": "Only STATUS_FINAL games included. Use find_team_by_name for official overall/home/road record.",
    }


def fetch_team_roster_stats(team_id: str) -> dict:
    """
    Fetch individual player statistics for a team's current roster.

    Returns per-game averages for the top players sorted by minutes played.
    Each player entry includes a pre-formatted summary line for easy citation.

    Args:
        team_id: ESPN team ID.

    Returns:
        dict with 'players' list (top 10 by minutes) and 'team_id'.
        Each player has: name, position, year, status, and a stats dict with
        clean per-game values (points, rebounds, assists, steals, blocks,
        minutes, FG%, 3P%, FT%, turnovers) plus a 'summary' string.
    """
    data = _get(f"{ESPN_BASE}/teams/{team_id}/roster")
    if "error" in data:
        return data

    athletes = data.get("athletes", [])
    players = []

    for group in athletes:
        items = group.get("items", [group]) if "items" in group else [group]
        for athlete in items:
            athlete_id = athlete.get("id")
            stats_url = (
                f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
                f"mens-college-basketball/seasons/2026/types/2/athletes/{athlete_id}/statistics"
            )
            stat_data = _get(stats_url)

            # Flatten all stat categories
            raw = {}
            for cat in stat_data.get("splits", {}).get("categories", []):
                for stat in cat.get("stats", []):
                    name = stat.get("name")
                    val  = stat.get("value")
                    if name and val is not None:
                        raw[name] = round(float(val), 2)

            # Derive clean per-game stats
            pts  = raw.get("avgPoints", 0)
            reb  = raw.get("avgRebounds", 0)
            ast  = raw.get("avgAssists", 0)
            stl  = raw.get("avgSteals", 0)
            blk  = raw.get("avgBlocks", 0)
            mins = raw.get("avgMinutes", 0)
            tov  = raw.get("avgTurnovers", 0)
            fga  = raw.get("avgFieldGoalsAttempted", 0)
            fgm  = raw.get("avgFieldGoalsMade", 0)
            fg3a = raw.get("avgThreePointFieldGoalsAttempted", 0)
            fg3m = raw.get("avgThreePointFieldGoalsMade", 0)
            fta  = raw.get("avgFreeThrowsAttempted", 0)
            fouls = raw.get("avgFouls", 0)

            fg_pct  = round(fgm  / fga  * 100, 1) if fga  else None
            fg3_pct = round(fg3m / fg3a * 100, 1) if fg3a else None
            ft_pct  = round(raw.get("avgFreeThrowsMade", 0) / fta * 100, 1) if fta else None
            ts_pct  = round(pts / (2 * (fga + 0.44 * fta)) * 100, 1) if (fga + 0.44 * fta) else None

            status = (athlete.get("status") or {}).get("name", "Active")

            # Pre-formatted summary line for easy agent citation
            parts = [f"{pts} PPG", f"{reb} RPG", f"{ast} APG"]
            if mins:
                parts.append(f"{mins} MPG")
            if fg3_pct is not None and fg3a >= 1.0:
                parts.append(f"{fg3_pct}% 3P on {fg3a:.1f} att")
            elif fg_pct is not None:
                parts.append(f"{fg_pct}% FG")
            if fouls >= 2.5:
                parts.append(f"⚠ {fouls} fouls/game")
            if status != "Active":
                parts.append(f"STATUS: {status}")

            players.append({
                "name": athlete.get("displayName") or athlete.get("fullName", "Unknown"),
                "position": (athlete.get("position") or {}).get("abbreviation"),
                "year": (athlete.get("experience") or {}).get("displayValue"),
                "status": status,
                "stats": {
                    "points_per_game": pts,
                    "rebounds_per_game": reb,
                    "assists_per_game": ast,
                    "steals_per_game": stl,
                    "blocks_per_game": blk,
                    "minutes_per_game": mins,
                    "turnovers_per_game": tov,
                    "fouls_per_game": fouls,
                    "fg_pct": fg_pct,
                    "three_point_pct": fg3_pct,
                    "three_point_attempts_per_game": fg3a,
                    "ft_pct": ft_pct,
                    "true_shooting_pct": ts_pct,
                },
                "summary": ", ".join(parts),
            })

    players.sort(key=lambda p: p["stats"].get("minutes_per_game", 0), reverse=True)

    return {
        "team_id": team_id,
        "players": players[:10],
    }


def fetch_play_by_play(event_id: str) -> dict:
    """
    Fetch play-by-play data for a completed college basketball game.

    Args:
        event_id: ESPN game ID (visible in ESPN game URLs, e.g. 401704064).

    Returns:
        dict with:
          - event_id, home_team, away_team dicts
          - plays: list of normalized play objects with period, clock, text,
            home_score, away_score, team_id, scoring_play, score_value, play_type
        or 'error' key on failure.
    """
    data = _get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary",
        params={"event": event_id},
    )
    if "error" in data:
        return data

    # Extract team identities from header
    home_team = away_team = None
    competitors = (
        data.get("header", {})
            .get("competitions", [{}])[0]
            .get("competitors", [])
    )
    for comp in competitors:
        info = {
            "id":   comp.get("id"),
            "name": comp.get("team", {}).get("displayName", ""),
        }
        if comp.get("homeAway") == "home":
            home_team = info
        else:
            away_team = info

    # Normalize plays — skip system/empty entries
    raw_plays = data.get("plays", [])
    plays = []
    for p in raw_plays:
        text = p.get("text", "").strip()
        if not text:
            continue
        plays.append({
            "period":         p.get("period", {}).get("number", 1),
            "period_display": p.get("period", {}).get("displayValue", "1st Half"),
            "clock":          p.get("clock", {}).get("displayValue", "0:00"),
            "text":           text,
            "home_score":     p.get("homeScore", 0),
            "away_score":     p.get("awayScore", 0),
            "team_id":        p.get("team", {}).get("id"),
            "scoring_play":   p.get("scoringPlay", False),
            "score_value":    p.get("scoreValue", 0),
            "play_type":      p.get("type", {}).get("text", ""),
        })

    return {
        "event_id":    event_id,
        "home_team":   home_team,
        "away_team":   away_team,
        "plays":       plays,
        "total_plays": len(plays),
    }
