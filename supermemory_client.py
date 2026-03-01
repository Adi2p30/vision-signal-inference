"""
Supermemory integration for the Historical Context Agent.

Four containers per game:
  team-{home}              — persistent, updated each game (home team rules)
  team-{away}              — persistent, updated each game (away team rules)
  game-{home}-vs-{away}    — per-game matchup context (venue, rivalry, signals)
  events-{game_tag}        — live game events, newest weighted highest

query_live() hits all four in parallel. Events container is queried with a
higher top_k and its results are placed first so recent game state takes
priority over pre-game context.
"""

import os
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Optional

from supermemory import Supermemory


# ── Container naming ──────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", text.lower().strip())[:80].strip("-")

def team_container(team_name: str) -> str:
    return f"team-{_slug(team_name)}"

def game_container(home: str, away: str, game_date: str | None = None) -> str:
    d = game_date or date.today().strftime("%Y%m%d")
    return f"game-{_slug(home)}-vs-{_slug(away)}-{d}"

def events_container(g_tag: str) -> str:
    return f"events-{g_tag}"


# ── Supermemory client ────────────────────────────────────────────────────────

def _sm() -> Supermemory:
    key = os.environ.get("SUPERMEMORY_API_KEY")
    if not key:
        raise RuntimeError("SUPERMEMORY_API_KEY is not set")
    return Supermemory(api_key=key)


# ── Context fetching ──────────────────────────────────────────────────────────

def _fetch_all(home_team: str, away_team: str) -> tuple[dict, dict, dict]:
    from historical_context_agent.tools.compile_tool import (
        interpret_team_context,
        interpret_game_context,
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        home_f  = pool.submit(interpret_team_context, home_team)
        away_f  = pool.submit(interpret_team_context, away_team)
        game_f  = pool.submit(interpret_game_context, home_team, away_team)
        home_ctx = home_f.result()
        away_ctx = away_f.result()
        game_ctx = game_f.result()

    for label, ctx in [("Home team", home_ctx), ("Away team", away_ctx), ("Game", game_ctx)]:
        if "error" in ctx:
            raise ValueError(f"{label}: {ctx['error']}")

    return home_ctx, away_ctx, game_ctx


# ── Storage helpers ───────────────────────────────────────────────────────────

def _parse_rules(context: str) -> list[str]:
    """Extract individual rule lines, skipping headers and blank lines."""
    rules = []
    for line in context.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line or len(line) < 20:
            continue
        if line.rstrip(":").upper() == line.rstrip(":") and "→" not in line and len(line) < 40:
            continue
        rules.append(line)
    return rules


def _store_rules(sm, context, team_name, container_tags, role):
    """Only store lines containing → (actual event-weighting rules).
    Baseline narrative sentences are excluded — they're descriptions, not rules."""
    slug = _slug(team_name)
    rules = [line for line in _parse_rules(context) if "→" in line]
    for i, line in enumerate(rules):
        try:
            sm.add(
                content=f"{team_name}: {line}",
                container_tags=container_tags,
                custom_id=f"{slug}-rule-{i:02d}",
                metadata={"team": team_name, "role": role, "type": "team_rule"},
            )
        except Exception:
            pass


def _store_player_profiles(sm, team_name: str, players: list, container_tags: list, role: str) -> None:
    """Store each player's stats as a searchable profile document in Supermemory."""
    slug = _slug(team_name)
    for player in players:
        name    = player.get("name", "Unknown")
        pos     = player.get("position") or "?"
        year    = player.get("year") or "?"
        status  = player.get("status", "Active")
        summary = player.get("summary", "")

        content = f"[PLAYER PROFILE] {team_name} | {name} ({pos}, {year})"
        if status != "Active":
            content += f" | STATUS: {status}"
        if summary:
            content += f" | {summary}"

        try:
            sm.add(
                content=content,
                container_tags=container_tags,
                custom_id=f"{slug}-player-{_slug(name)}",
                metadata={
                    "team": team_name,
                    "player_name": name,
                    "role": role,
                    "type": "player_profile",
                    "status": status,
                },
            )
        except Exception:
            pass


def _store_game_rules(sm, context, home_name, away_name, g_tag):
    slug = _slug(f"{home_name}-vs-{away_name}")
    for i, line in enumerate(_parse_rules(context)):
        try:
            sm.add(
                content=line,
                container_tags=[g_tag],
                custom_id=f"{slug}-game-{i:02d}",
                metadata={"home_team": home_name, "away_team": away_name, "type": "game_rule"},
            )
        except Exception:
            pass


def _extract_chunks(result, top_k: int) -> list[str]:
    """Pull text chunks from a profile search_results response."""
    chunks = []
    if result.search_results and result.search_results.results:
        for r in result.search_results.results:
            text = (r.get("memory") or r.get("chunk") or "") if isinstance(r, dict) else (
                getattr(r, "memory", None) or getattr(r, "chunk", None) or ""
            )
            text = text.strip()
            if text and text not in chunks:
                chunks.append(text)
            if len(chunks) >= top_k:
                break
    return chunks


# ── Load game (pre-game) ──────────────────────────────────────────────────────

def load_game(home_team: str, away_team: str, game_date: str | None = None) -> dict:
    """
    Pre-game: fetch all contexts and load into Supermemory.
    Creates all four containers and returns their tags.

    Returns:
        dict with keys: game_tag, h_tag, a_tag, e_tag
    """
    home_ctx, away_ctx, game_ctx = _fetch_all(home_team, away_team)

    sm             = _sm()
    resolved_home  = home_ctx["team"]
    resolved_away  = away_ctx["team"]
    h_tag          = team_container(resolved_home)
    a_tag          = team_container(resolved_away)
    g_tag          = game_container(home_team, away_team, game_date)
    e_tag          = events_container(g_tag)

    # Team rules → both team container (persistent) and game container (scoped)
    _store_rules(sm, home_ctx["context"], resolved_home, [h_tag, g_tag], "home")
    _store_rules(sm, away_ctx["context"], resolved_away, [a_tag, g_tag], "away")

    # Game context rules → game container only
    _store_game_rules(sm, game_ctx["context"], resolved_home, resolved_away, g_tag)

    # Player profiles → both team container (persistent) and game container (scoped)
    home_players = home_ctx.get("players", [])
    away_players = away_ctx.get("players", [])

    def _store_home_players():
        _store_player_profiles(sm, resolved_home, home_players, [h_tag, g_tag], "home")

    def _store_away_players():
        _store_player_profiles(sm, resolved_away, away_players, [a_tag, g_tag], "away")

    with ThreadPoolExecutor(max_workers=2) as pool:
        home_pf = pool.submit(_store_home_players)
        away_pf = pool.submit(_store_away_players)
        home_pf.result()
        away_pf.result()

    print(f"  Players : {len(home_players)} home + {len(away_players)} away profiles stored")

    # Seed the events container so it exists before the first add-event call
    try:
        sm.add(
            content=f"GAME START: {resolved_home} (home) vs {resolved_away} (away)",
            container_tags=[e_tag],
            custom_id=f"{e_tag}-start",
            metadata={"type": "game_event", "sequence": 0, "game_time": "start"},
        )
    except Exception:
        pass

    print(f"Loaded game memory:")
    print(f"  Home    : {h_tag}")
    print(f"  Away    : {a_tag}")
    print(f"  Game    : {g_tag}")
    print(f"  Events  : {e_tag}")

    return {"game_tag": g_tag, "h_tag": h_tag, "a_tag": a_tag, "e_tag": e_tag}


# ── Add live event ────────────────────────────────────────────────────────────

def add_game_event(
    e_tag: str,
    event: str,
    game_time: str | None = None,
    sequence: int = 0,
) -> None:
    """
    Store a live game event in the events container.

    Events are stored with a timestamp so Supermemory's recency bias
    naturally surfaces newer events higher during retrieval. The sequence
    number and game_time are embedded in the content so semantic search
    also reflects event order.

    Args:
        e_tag:      Events container tag (from load_game return dict).
        event:      Description of what happened (e.g. "Kansas made a 3-pointer").
        game_time:  Optional game clock string (e.g. "Q2 14:32").
        sequence:   Monotonically increasing event counter for ordering.
    """
    sm        = _sm()
    ts        = datetime.utcnow().isoformat()
    time_str  = game_time or ts
    content   = f"[{time_str}] #{sequence:04d} {event}"

    sm.add(
        content=content,
        container_tags=[e_tag],
        custom_id=f"{e_tag}-event-{sequence:04d}",
        metadata={
            "type": "game_event",
            "sequence": sequence,
            "game_time": time_str,
            "timestamp": ts,
        },
    )


# ── Momentum synthesis ────────────────────────────────────────────────────────

_MOMENTUM_PROMPT = """\
You are the momentum intelligence layer for a live college basketball trading agent.

GAME: {home_team} (home) vs {away_team} (away)
OBSERVED PLAY: {observed_event}

RECENT GAME FLOW — last {n_recent} plays, chronological (oldest → newest):
{recent_block}

PAST PLAYS SEMANTICALLY RELATED TO OBSERVED PLAY (may be from earlier in game):
{semantic_block}

PLAYER PROFILES (season stats for players involved in or relevant to this play):
{player_block}

APPLICABLE TEAM RULES (pre-game stat-derived event weights):
{rules_block}

GAME CONTEXT (venue / rivalry / matchup):
{context_block}

Output EXACTLY this format with no other text:

MOMENTUM: [home_team or away_team or NEUTRAL] [(+) strong / (~) moderate / (=) neutral]
SCORING RUN: [e.g. "Kansas 9-2 over last 5 plays" — derive from scores visible in recent flow; "Tied / no clear run" if unclear]
SITUATION: [1-2 sentences: what is happening now and what it means for next 2-3 possessions, referencing the player's stats if relevant]
KEY RULE: [exact team rule text that is triggered right now, or "None"]
KEY PLAYER: [player name and the specific stat that makes this play significant, or "None"]

Definitions:
- MOMENTUM team = whoever has scored more in the 4-6 most recent plays; NEUTRAL if tied/unclear
- Use PLAYER PROFILES to assess the significance of the play (e.g. a 20 PPG scorer hitting is high-leverage)
- Never speculate beyond what the plays, profiles, and rules explicitly show\
"""


def _tag_to_name(tag: str, prefix: str = "team-") -> str:
    """team-kansas-jayhawks → Kansas Jayhawks (best-effort; used only for prompt context)."""
    return tag.removeprefix(prefix).replace("-", " ").title()


def _synthesize_momentum(
    observed_event: str,
    home_team: str,
    away_team: str,
    recent_events: list[str],
    semantic_events: list[str],
    player_profiles: list[str],
    team_rules: list[str],
    game_context: list[str],
) -> str:
    """Single Gemini call that produces the structured momentum assessment."""
    import os
    try:
        from google import genai
    except ImportError:
        return _fallback_context(recent_events, semantic_events, player_profiles, team_rules, game_context)

    recent_block   = "\n".join(recent_events) if recent_events else "No recent events recorded yet."
    semantic_block = "\n".join(semantic_events) if semantic_events else "None."
    player_block   = "\n".join(player_profiles) if player_profiles else "No player profiles retrieved."
    rules_block    = "\n".join(team_rules) if team_rules else "None."
    context_block  = "\n".join(game_context) if game_context else "None."

    prompt = _MOMENTUM_PROMPT.format(
        home_team      = home_team,
        away_team      = away_team,
        observed_event = observed_event,
        n_recent       = len(recent_events),
        recent_block   = recent_block,
        semantic_block = semantic_block,
        player_block   = player_block,
        rules_block    = rules_block,
        context_block  = context_block,
    )

    try:
        t0 = _time.time()
        print(f"[supermemory] calling Gemini for momentum synthesis", flush=True)
        client   = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        print(f"[supermemory] Gemini momentum done — {_time.time()-t0:.1f}s", flush=True)
        return response.text.strip()
    except Exception as e:
        print(f"[supermemory] Gemini momentum failed: {e}", flush=True)
        return _fallback_context(recent_events, semantic_events, player_profiles, team_rules, game_context)


def _fallback_context(
    recent_events: list[str],
    semantic_events: list[str],
    player_profiles: list[str],
    team_rules: list[str],
    game_context: list[str],
) -> str:
    """Plain-text fallback if Gemini is unavailable."""
    parts = []
    if recent_events:
        parts.append("RECENT GAME FLOW:\n" + "\n".join(recent_events))
    if semantic_events:
        parts.append("RELATED PAST PLAYS:\n" + "\n".join(semantic_events))
    if player_profiles:
        parts.append("PLAYER PROFILES:\n" + "\n".join(player_profiles))
    if team_rules:
        parts.append("TEAM RULES:\n" + "\n".join(team_rules))
    if game_context:
        parts.append("GAME CONTEXT:\n" + "\n".join(game_context))
    return "\n\n".join(parts) if parts else ""


# ── Live query ────────────────────────────────────────────────────────────────

def retrieve_context(
    game_tag: str,
    observed_event: str,
    h_tag: str,
    a_tag: str,
    e_tag: str,
    recent_events: Optional[list[str]] = None,
    top_k: int = 3,
) -> dict:
    """
    Query all four Supermemory containers in parallel and return raw sections.

    Returns a dict with keys:
        recent_events   — chronological buffer passed in (not from Supermemory)
        semantic_events — past plays semantically similar to observed_event
        player_profiles — per-player stat profiles relevant to this play
        team_rules      — deduplicated stat-derived team rules
        game_context    — venue/rivalry/matchup chunks
        home_team       — display name derived from h_tag
        away_team       — display name derived from a_tag
    """
    sm = _sm()
    t0 = _time.time()
    print(f"[supermemory] querying 4 containers for: {observed_event!r:.60}")

    def _query(tag: str, k: int) -> list[str]:
        t1 = _time.time()
        try:
            result = sm.profile(container_tag=tag, q=observed_event)
            chunks = _extract_chunks(result, k)
            print(f"[supermemory] {tag!r:.40} → {len(chunks)} chunks — {_time.time()-t1:.1f}s", flush=True)
            return chunks
        except Exception as e:
            print(f"[supermemory] {tag!r:.40} → error: {e} — {_time.time()-t1:.1f}s", flush=True)
            return []

    # Use top_k * 2 for team containers so we surface both player profiles and team rules
    with ThreadPoolExecutor(max_workers=4) as pool:
        events_f = pool.submit(_query, e_tag,    top_k * 2)
        game_f   = pool.submit(_query, game_tag, top_k)
        home_f   = pool.submit(_query, h_tag,    top_k * 2)
        away_f   = pool.submit(_query, a_tag,    top_k * 2)

        semantic_events = events_f.result()
        game_chunks     = game_f.result()
        home_chunks     = home_f.result()
        away_chunks     = away_f.result()

    print(f"[supermemory] all 4 queries done — {_time.time()-t0:.1f}s", flush=True)

    # Split team chunks: player profiles (prefixed) vs stat-derived rules
    seen: set[str] = set()
    player_profiles: list[str] = []
    team_rules: list[str] = []
    for chunk in home_chunks + away_chunks:
        if chunk in seen:
            continue
        seen.add(chunk)
        if chunk.startswith("[PLAYER PROFILE]"):
            player_profiles.append(chunk)
        else:
            team_rules.append(chunk)

    return {
        "recent_events":   recent_events or [],
        "semantic_events": semantic_events,
        "player_profiles": player_profiles,
        "team_rules":      team_rules,
        "game_context":    game_chunks,
        "home_team":       _tag_to_name(h_tag),
        "away_team":       _tag_to_name(a_tag),
    }


def synthesize_momentum(
    observed_event: str,
    ctx: dict,
) -> str:
    """
    Run Gemini synthesis on a context dict returned by retrieve_context().

    Args:
        observed_event: What the VLM just observed.
        ctx:            Dict from retrieve_context().

    Returns:
        Structured momentum assessment string.
    """
    return _synthesize_momentum(
        observed_event  = observed_event,
        home_team       = ctx["home_team"],
        away_team       = ctx["away_team"],
        recent_events   = ctx["recent_events"],
        semantic_events = ctx["semantic_events"],
        player_profiles = ctx.get("player_profiles", []),
        team_rules      = ctx["team_rules"],
        game_context    = ctx["game_context"],
    )


def _format_raw_context(observed_event: str, ctx: dict) -> str:
    """Format a retrieve_context() dict into labeled plain-text sections."""
    sections = [f"OBSERVED PLAY: {observed_event}"]

    if ctx["recent_events"]:
        sections.append(
            "RECENT GAME FLOW (chronological, oldest → newest):\n"
            + "\n".join(ctx["recent_events"])
        )
    else:
        sections.append("RECENT GAME FLOW: No events recorded yet.")

    if ctx["semantic_events"]:
        sections.append(
            "PAST PLAYS (semantically related to observed play):\n"
            + "\n".join(ctx["semantic_events"])
        )

    if ctx["team_rules"]:
        sections.append("TEAM RULES:\n" + "\n".join(ctx["team_rules"]))

    if ctx["game_context"]:
        sections.append("GAME CONTEXT:\n" + "\n".join(ctx["game_context"]))

    return "\n\n".join(sections)
