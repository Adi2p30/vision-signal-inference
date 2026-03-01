"""
Two-stage team context pipeline:

1. compile_team_context()  — deterministic Python, fetches all raw stats
2. interpret_team_context() — single Gemini call, translates stats into
                               trading-focused implications (no hallucination
                               risk because stats are injected into the prompt)
"""

import os

from .espn_tools import (
    find_team_by_name,
    fetch_team_statistics,
    fetch_team_schedule,
    fetch_team_roster_stats,
)
from .analytics_tools import (
    fetch_barttorvik_stats,
    fetch_injury_report,
)
from .game_tools import (
    fetch_venue_info,
    fetch_head_to_head,
    detect_rivalry,
    compute_matchup_type,
)

_INTERPRETATION_PROMPT = """\
You are calibrating a live VLM that watches basketball games and tracks score-state for {team}.

The VLM sees live plays. It needs to know: for THIS team, which observable events are \
high-leverage (directly cause point swings or possession losses) vs. noise.

Using ONLY the stats below, output:

1. BASELINE (2 lines): describe the team's normal scoring engine and pace so the VLM \
   knows what "expected play" looks like. No numbers — just the pattern.

2. CRITICAL EVENTS (4 events): observable plays that cause a direct, \
   stat-justified score or possession shift for this team. Format each as:
   [OBSERVABLE EVENT] → [POINT/POSSESSION OUTCOME, quantified where possible]

Rules:
- Derive each event from a statistical extreme in this team's profile
- Quantify outcomes using the stats (e.g. "costs ~1.0 pts per occurrence", \
  "extends possession by 1 extra shot")
- No speculation, no psychology, no generic basketball logic
- Each event must only matter because of THIS team's specific numbers

STATS:
{stats_block}

Output:"""


def compile_team_context(team_name: str) -> dict:
    """
    Fetch all possession-level stats for a college basketball team.

    Returns a raw data dict (not formatted text) suitable for further processing
    or direct injection into an LLM interpretation prompt.

    Args:
        team_name: College basketball team name (e.g. "Purdue", "Illinois", "Duke").

    Returns:
        dict with keys: team, team_id, record, home_record, road_record, stats,
        barttorvik, foul_risks, injuries, schedule. Or 'error' key on failure.
    """
    # ── 1. Team identity ─────────────────────────────────────────────────────
    team = find_team_by_name(team_name)
    if "error" in team:
        return {"error": team["error"]}

    team_id  = team["team_id"]
    display  = team["display_name"]

    # ── 2. Possession stats ──────────────────────────────────────────────────
    stats = fetch_team_statistics(team_id)
    if "error" in stats:
        return {"error": f"Stats: {stats['error']}"}

    # ── 3. Schedule / recent form ────────────────────────────────────────────
    sched = fetch_team_schedule(team_id)
    if "error" in sched:
        return {"error": f"Schedule: {sched['error']}"}

    # ── 4. Foul risks from roster ────────────────────────────────────────────
    roster = fetch_team_roster_stats(team_id)
    foul_risks = []
    if "error" not in roster:
        foul_risks = [
            {"name": p["name"], "position": p["position"] or "?",
             "fouls_per_game": p["stats"].get("fouls_per_game", 0)}
            for p in roster.get("players", [])
            if p["stats"].get("fouls_per_game", 0) >= 2.5
        ]

    # ── 5. Barttorvik ────────────────────────────────────────────────────────
    bart = fetch_barttorvik_stats(team_name)
    if "error" in bart:
        bart = fetch_barttorvik_stats(team_name.split()[0])

    # ── 6. Injury report ─────────────────────────────────────────────────────
    inj = fetch_injury_report(team_id, display)

    return {
        "team": display,
        "team_id": team_id,
        "record": team["overall_record"],
        "home_record": team["home_record"],
        "road_record": team["road_record"],
        "stats": stats,
        "barttorvik": bart if "error" not in bart else {},
        "foul_risks": foul_risks,
        "players": roster.get("players", []) if "error" not in roster else [],
        "injuries": inj,
        "schedule": sched,
    }


def _build_stats_block(data: dict) -> str:
    """Format the raw data dict into a compact stats block for the LLM prompt."""
    s    = data["stats"]
    bart = data["barttorvik"]
    sched = data["schedule"]
    foul_risks = data["foul_risks"]
    inj  = data["injuries"]

    lines = [
        f"Team: {data['team']} | {data['record']} ({data['home_record']} H / {data['road_record']} R)",
    ]

    if bart:
        net = round(bart["adj_offensive_efficiency"] - bart["adj_defensive_efficiency"], 1)
        lines += [
            f"adjOE: {bart['adj_offensive_efficiency']} | adjDE: {bart['adj_defensive_efficiency']} | Net: {'+' if net >= 0 else ''}{net}",
            f"Tempo: {bart.get('adj_tempo_possessions_per_40min', 'N/A')} poss/40 min",
        ]

    lines += [
        f"3PA Rate: {s.get('three_point_attempt_rate_pct')}% | 3P%: {s.get('three_point_pct')}%",
        f"TO Rate: {s.get('to_rate_per_100_poss')} per 100 poss",
        f"FT Rate: {s.get('ft_rate_fta_per_fga')} | FT%: {s.get('ft_pct')}%",
        f"eFG%: {s.get('effective_fg_pct')} | TS%: {s.get('true_shooting_pct')}",
        f"OR%: {s.get('offensive_rebound_rate_pct')}% | DR%: {s.get('defensive_rebound_rate_pct')}%",
        f"OReb/g: {s.get('offensive_rebounds_per_game')} | DReb/g: {s.get('defensive_rebounds_per_game')}",
        f"Fouls/g: {s.get('fouls_per_game')} | Blocks/g: {s.get('blocks_per_game')} | Steals/g: {s.get('steals_per_game')}",
    ]

    if foul_risks:
        risk_str = ", ".join(f"{p['name']} ({p['position']}) {p['fouls_per_game']}/g" for p in foul_risks)
        lines.append(f"Foul risks (>=2.5/g): {risk_str}")
    else:
        lines.append("Foul risks: none above threshold")

    inj_str = "none" if inj.get("no_known_injuries") else (
        "; ".join(p["name"] for p in inj.get("non_active_players", []))
        or "; ".join(h["headline"] for h in inj.get("injury_news", []))
        or "none"
    )
    lines.append(f"Injuries: {inj_str}")

    last5 = sched.get("last_5_record", "N/A")
    form  = sched.get("last_5_form", "")
    lines.append(f"Last 5: {last5} [{form}]")

    return "\n".join(lines)


def interpret_team_context(team_name: str) -> dict:
    """
    Fetch all stats for a college basketball team and return a 5–7 sentence
    trading-focused analytical summary of game-dynamic implications.

    Stats are collected deterministically in Python; the LLM only handles
    interpretation — it cannot hallucinate numbers because the prompt contains
    the exact verified stats and forbids restating them.

    Args:
        team_name: College basketball team name.

    Returns:
        dict with 'team' (resolved name) and 'context' (5–7 sentence summary),
        or 'error' key on failure.
    """
    # Step 1 — collect data (pure Python, no LLM)
    data = compile_team_context(team_name)
    if "error" in data:
        return data

    stats_block = _build_stats_block(data)
    prompt = _INTERPRETATION_PROMPT.format(
        team=data["team"],
        stats_block=stats_block,
    )

    # Step 2 — interpret (single Gemini call, no tools)
    try:
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        summary = response.text.strip()
    except Exception as e:
        return {"error": f"LLM interpretation failed: {e}"}

    return {
        "team": data["team"],
        "team_id": data["team_id"],
        "context": summary,
        "players": data.get("players", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Game context pipeline
# ─────────────────────────────────────────────────────────────────────────────

_GAME_INTERPRETATION_PROMPT = """\
You are configuring a live VLM that watches: {away_team} @ {home_team}.

The VLM needs to understand how the environment and occasion of THIS game will affect \
how teams play — not their stats, but the context that changes what normal looks like.

Using ONLY the data below, output exactly three labeled sections:

VENUE: Describe how the home arena specifically affects road teams in this game. \
Is it a hostile environment? Does the home team's home vs road record split indicate \
real crowd impact or minimal? What should the VLM expect about home crowd influence \
on away team execution (e.g., FT attempts, early turnovers, slow starts)?

GAME CONTEXT: If this is a rivalry, explain what that means for how the game will play — \
rivalry games historically compress efficiency gaps, elevate underdog intensity, and make \
early leads less predictive. If conference implications exist (standings, seeding), note \
what's at stake and how desperation affects play. If no rivalry, state that the game \
will likely follow the efficiency gap.

EXPECTATION ADJUSTMENT: Tell the VLM how to recalibrate its baseline for this specific game. \
Should it widen its uncertainty band (rivalry, road team with upset history)? \
Should it weight early runs differently (hostile venue, slow-starting road team)? \
Should it treat the underdog's runs as noise or sustained pressure? One to two sentences.

Rules:
- Each section is 1–2 sentences max — be direct
- Every claim must cite a specific number or fact from the data
- No predictions, no speculation, no "expected to" or "likely to" language
- State only what the data confirms, not what might happen

DATA:
{game_block}

Output:"""


def compile_game_context(home_team_name: str, away_team_name: str) -> dict:
    """
    Fetch all data needed to characterize a college basketball game matchup:
    venue, home court advantage, head-to-head, rivalry, efficiency + possession stats.

    Args:
        home_team_name: Name of the home team (e.g. "Purdue", "Illinois").
        away_team_name: Name of the away team.

    Returns:
        dict with home, away, venue, home_advantage, head_to_head, rivalry,
        matchup, home_stats, away_stats keys. 'error' key on failure.
    """
    # ── Resolve both teams ───────────────────────────────────────────────────
    home = find_team_by_name(home_team_name)
    if "error" in home:
        return {"error": f"Home team: {home['error']}"}
    away = find_team_by_name(away_team_name)
    if "error" in away:
        return {"error": f"Away team: {away['error']}"}

    home_id = home["team_id"]
    away_id = away["team_id"]

    # ── Venue (from schedule endpoint — only place ESPN has it) ───────────────
    venue = fetch_venue_info(home_id)

    # ── Home court advantage ──────────────────────────────────────────────────
    def _win_pct(record_str: str) -> float | None:
        try:
            w, l = record_str.split("-")
            total = int(w) + int(l)
            return round(int(w) / total * 100, 1) if total else None
        except Exception:
            return None

    home_pct  = _win_pct(home["home_record"])
    road_pct  = _win_pct(home["road_record"])
    adv_delta = round(home_pct - road_pct, 1) if (home_pct is not None and road_pct is not None) else None

    # ── Barttorvik for both teams (conference comes from here) ────────────────
    home_bart = fetch_barttorvik_stats(home_team_name)
    if "error" in home_bart:
        home_bart = fetch_barttorvik_stats(home_team_name.split()[0])

    away_bart = fetch_barttorvik_stats(away_team_name)
    if "error" in away_bart:
        away_bart = fetch_barttorvik_stats(away_team_name.split()[0])

    def _bart_val(b, key, default=None):
        return b.get(key, default) if "error" not in b else default

    home_net = None
    away_net = None
    if "error" not in home_bart:
        home_net = round(home_bart["adj_offensive_efficiency"] - home_bart["adj_defensive_efficiency"], 1)
    if "error" not in away_bart:
        away_net = round(away_bart["adj_offensive_efficiency"] - away_bart["adj_defensive_efficiency"], 1)

    eff_gap      = round(home_net - away_net, 1) if (home_net is not None and away_net is not None) else None
    matchup_type = compute_matchup_type(home_net, away_net) if eff_gap is not None else None

    # ── ESPN possession stats for both teams ──────────────────────────────────
    home_stats = fetch_team_statistics(home_id)
    away_stats = fetch_team_statistics(away_id)

    # ── Conference (from barttorvik) + rivalry ────────────────────────────────
    home_conf = _bart_val(home_bart, "conference")
    away_conf = _bart_val(away_bart, "conference")
    same_conf = bool(home_conf and away_conf and home_conf == away_conf)

    h2h     = fetch_head_to_head(home_id, away_id, away["display_name"])
    rivalry = detect_rivalry(
        home["display_name"], away["display_name"],
        same_conf, h2h["h2h_wins"] + h2h["h2h_losses"],
    )

    return {
        "home": {
            "name": home["display_name"],
            "record": home["overall_record"],
            "home_record": home["home_record"],
            "road_record": home["road_record"],
            "conference": home_conf,
            "t_rank":     _bart_val(home_bart, "t_rank"),
            "net_eff":    home_net,
            "adj_oe":     _bart_val(home_bart, "adj_offensive_efficiency"),
            "adj_de":     _bart_val(home_bart, "adj_defensive_efficiency"),
            "adj_tempo":  _bart_val(home_bart, "adj_tempo_possessions_per_40min"),
            "barthag":    _bart_val(home_bart, "barthag"),
        },
        "away": {
            "name": away["display_name"],
            "record": away["overall_record"],
            "home_record": away["home_record"],
            "road_record": away["road_record"],
            "conference": away_conf,
            "t_rank":     _bart_val(away_bart, "t_rank"),
            "net_eff":    away_net,
            "adj_oe":     _bart_val(away_bart, "adj_offensive_efficiency"),
            "adj_de":     _bart_val(away_bart, "adj_defensive_efficiency"),
            "adj_tempo":  _bart_val(away_bart, "adj_tempo_possessions_per_40min"),
            "barthag":    _bart_val(away_bart, "barthag"),
        },
        "venue": venue,
        "home_advantage": {
            "home_win_pct": home_pct,
            "road_win_pct": road_pct,
            "delta_pct": adv_delta,
        },
        "home_stats": home_stats if "error" not in home_stats else {},
        "away_stats": away_stats if "error" not in away_stats else {},
        "head_to_head": h2h,
        "rivalry": rivalry,
        "same_conference": same_conf,
        "matchup": {
            "type": matchup_type,
            "eff_gap": eff_gap,
            "favored": (
                home["display_name"] if (eff_gap or 0) > 0
                else away["display_name"] if (eff_gap or 0) < 0
                else "even"
            ),
        },
    }


def _fmt(val, suffix="", prefix="") -> str:
    """Safe formatter for potentially None numeric values."""
    return f"{prefix}{val}{suffix}" if val is not None else "N/A"


def _build_game_block(data: dict) -> str:
    """Format raw game context dict into a compact block for prompt injection."""
    h  = data["home"]
    a  = data["away"]
    v  = data["venue"]
    ha = data["home_advantage"]
    h2h = data["head_to_head"]
    matchup = data["matchup"]
    hs = data["home_stats"]
    as_ = data["away_stats"]

    gap_str = _fmt(matchup["eff_gap"], prefix="+") if (matchup["eff_gap"] or 0) >= 0 else str(matchup["eff_gap"])

    lines = [
        f"HOME: {h['name']} ({h['record']}) | Home: {h['home_record']} | Road: {h['road_record']} | Conf: {h['conference'] or 'N/A'}",
        f"AWAY: {a['name']} ({a['record']}) | Home: {a['home_record']} | Road: {a['road_record']} | Conf: {a['conference'] or 'N/A'}",
        "",
        f"Venue: {v.get('venue_name', 'N/A')}, {v.get('city', '')}, {v.get('state', '')}",
        f"Home court: {ha['home_win_pct']}% home win% vs {ha['road_win_pct']}% road win% (delta: {_fmt(ha['delta_pct'], '%')})",
        "",
        f"{h['name']} — T-Rank: {_fmt(h['t_rank'])} | adjOE: {_fmt(h['adj_oe'])} | adjDE: {_fmt(h['adj_de'])} | Net: {_fmt(h['net_eff'])} | Tempo: {_fmt(h['adj_tempo'])}",
        f"{a['name']} — T-Rank: {_fmt(a['t_rank'])} | adjOE: {_fmt(a['adj_oe'])} | adjDE: {_fmt(a['adj_de'])} | Net: {_fmt(a['net_eff'])} | Tempo: {_fmt(a['adj_tempo'])}",
        f"Efficiency gap: {gap_str} (positive = home favored)",
        "",
        f"{h['name']} possession stats — TO/100: {_fmt(hs.get('to_rate_per_100_poss'))} | OR%: {_fmt(hs.get('offensive_rebound_rate_pct'),'%')} | DR%: {_fmt(hs.get('defensive_rebound_rate_pct'),'%')} | 3PA%: {_fmt(hs.get('three_point_attempt_rate_pct'),'%')} | FT%: {_fmt(hs.get('ft_pct'),'%')} | Stl/g: {_fmt(hs.get('steals_per_game'))}",
        f"{a['name']} possession stats — TO/100: {_fmt(as_.get('to_rate_per_100_poss'))} | OR%: {_fmt(as_.get('offensive_rebound_rate_pct'),'%')} | DR%: {_fmt(as_.get('defensive_rebound_rate_pct'),'%')} | 3PA%: {_fmt(as_.get('three_point_attempt_rate_pct'),'%')} | FT%: {_fmt(as_.get('ft_pct'),'%')} | Stl/g: {_fmt(as_.get('steals_per_game'))}",
        "",
        f"Rivalry: {data['rivalry'] or 'None'} | Same conf: {data['same_conference']}",
        f"H2H this season: {h2h['h2h_record']}",
    ]

    for g in h2h.get("h2h_games", []):
        lines.append(f"  {g['result']} {g['score']} ({g['location']}, {g['date']})")

    return "\n".join(lines)


def interpret_game_context(home_team_name: str, away_team_name: str) -> dict:
    """
    Fetch all matchup data and return a VLM-operational game context:
    venue difficulty, matchup type, rivalry flag, and stat-grounded critical signals.

    Args:
        home_team_name: Home team name.
        away_team_name: Away team name.

    Returns:
        dict with 'home', 'away', and 'context' (formatted game brief), or 'error'.
    """
    data = compile_game_context(home_team_name, away_team_name)
    if "error" in data:
        return data

    game_block = _build_game_block(data)
    prompt = _GAME_INTERPRETATION_PROMPT.format(
        home_team=data["home"]["name"],
        away_team=data["away"]["name"],
        game_block=game_block,
    )

    try:
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        summary = response.text.strip()
    except Exception as e:
        return {"error": f"LLM interpretation failed: {e}"}

    return {
        "home_team": data["home"]["name"],
        "away_team": data["away"]["name"],
        "context": summary,
    }
