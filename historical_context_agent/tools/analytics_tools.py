"""
Analytics tools: AP Poll rankings, Barttorvik (KenPom-style) efficiency metrics,
and ESPN injury/news data. All free, no API key required.
"""

import requests
from difflib import SequenceMatcher

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HistoricalContextAgent/1.0)",
    "Accept": "application/json",
}

# Cached Barttorvik data so we only fetch the full list once per run
_barttorvik_cache: list | None = None


def _get(url: str, params: dict = None) -> dict | list:
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fetch_barttorvik_stats(team_name: str) -> dict:
    """
    Fetch KenPom-style advanced efficiency metrics from Barttorvik (T-Rank).

    Returns adjusted offensive and defensive efficiency ratings, T-Rank (overall
    efficiency rank), adjusted tempo (possessions per 40 min), and Barthag
    (probability of beating an average D1 team). These are the industry-standard
    metrics for evaluating team quality.

    Args:
        team_name: College basketball team name. Fuzzy matched against Barttorvik's
                   team list (e.g., "Illinois", "Duke", "Kentucky").

    Returns:
        dict with: t_rank (overall rank), adj_offensive_efficiency,
        adj_off_rank, adj_defensive_efficiency, adj_def_rank,
        adj_tempo, barthag, conference_record, or 'error' on failure.
    """
    global _barttorvik_cache

    if _barttorvik_cache is None:
        raw = _get("https://barttorvik.com/2026_team_results.json")
        if isinstance(raw, dict) and "error" in raw:
            return raw
        _barttorvik_cache = raw

    # Column layout (confirmed from Barttorvik source):
    # [0]=rank, [1]=team, [2]=conf, [3]=record, [4]=adjOE, [5]=adjOE_rank,
    # [6]=adjDE, [7]=adjDE_rank, [8]=barthag, [9]=barthag_rank,
    # [14]=conf_record, [44]=adj_tempo
    best_team = None
    best_score = 0.0

    for entry in _barttorvik_cache:
        if not isinstance(entry, list) or len(entry) < 10:
            continue
        name = str(entry[1])
        score = _similarity(team_name, name)
        if team_name.lower() in name.lower():
            score = max(score, 0.9)
        if score > best_score:
            best_score = score
            best_team = entry

    if not best_team or best_score < 0.4:
        return {"error": f"Team '{team_name}' not found in Barttorvik data"}

    adj_oe  = best_team[4]
    adj_oe_rank = int(best_team[5])
    adj_de  = best_team[6]
    adj_de_rank = int(best_team[7])
    barthag = best_team[8]
    t_rank  = int(best_team[9])
    conf_abbr   = best_team[2] if len(best_team) > 2 else None
    conf_record = best_team[14] if len(best_team) > 14 else "N/A"
    adj_tempo = round(best_team[44], 1) if len(best_team) > 44 else None

    return {
        "team_name": best_team[1],
        "conference": conf_abbr,
        "t_rank": t_rank,
        "adj_offensive_efficiency": round(adj_oe, 1),
        "adj_off_rank": adj_oe_rank,
        "adj_defensive_efficiency": round(adj_de, 1),
        "adj_def_rank": adj_de_rank,
        "barthag": round(barthag, 4),
        "adj_tempo_possessions_per_40min": adj_tempo,
        "conference_record": conf_record,
        "match_confidence": round(best_score, 3),
        "note": (
            "adjOE = points scored per 100 possessions (adjusted). "
            "adjDE = points allowed per 100 possessions (adjusted, lower is better). "
            "Avg D1 team is ~100 for both. T-Rank = overall efficiency rank (lower is better)."
        ),
    }


def fetch_injury_report(team_id: str, team_name: str) -> dict:
    """
    Fetch current injury and roster status information for a team.

    Checks ESPN team news for injury-related headlines and returns a summary
    of any known player injuries, absences, or questionable designations.

    Args:
        team_id: ESPN team ID.
        team_name: Team name used for context in the returned summary.

    Returns:
        dict with injury_headlines (list of recent injury news), roster_concerns
        (any player whose status ESPN flags as non-Active), or 'no_known_injuries'
        flag if everything is clean.
    """
    # Check news for injury keywords
    news_data = _get(
        f"{ESPN_BASE}/news",
        params={"team": team_id, "limit": 10},
    )

    # Keywords that strongly signal a player-specific injury article when in the headline
    injury_headline_keywords = {
        "injur", "out for", "questionable", "doubtful", "suspended",
        "day-to-day", "won't play", "will miss", "ruled out", "sidelined",
    }
    injury_headlines = []

    for article in news_data.get("articles", []) if isinstance(news_data, dict) else []:
        headline = article.get("headline", "").lower()
        description = article.get("description", "").lower()
        # Only flag if the headline itself contains an injury-specific term
        if any(kw in headline for kw in injury_headline_keywords):
            injury_headlines.append({
                "headline": article.get("headline", ""),
                "description": article.get("description", "")[:200],
                "published": article.get("published", "")[:10],
            })

    # Check roster for non-Active player statuses
    roster_data = _get(f"{ESPN_BASE}/teams/{team_id}/roster")
    roster_concerns = []

    for group in roster_data.get("athletes", []) if isinstance(roster_data, dict) else []:
        items = group.get("items", [group]) if "items" in group else [group]
        for athlete in items:
            status = (athlete.get("status") or {}).get("name", "Active")
            if status != "Active":
                roster_concerns.append({
                    "name": athlete.get("displayName", "Unknown"),
                    "status": status,
                    "position": (athlete.get("position") or {}).get("abbreviation"),
                })

    return {
        "team_id": team_id,
        "team_name": team_name,
        "injury_news": injury_headlines[:5],
        "non_active_players": roster_concerns,
        "no_known_injuries": len(injury_headlines) == 0 and len(roster_concerns) == 0,
    }
