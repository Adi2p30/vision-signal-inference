"""
JSON-backed events store for chronological game event retrieval.
Persists across server restarts via events_store.json.
"""

import json
import threading

_MAX_RECENT  = 15
_EVENTS_FILE = "events_store.json"
_lock        = threading.Lock()


def _load() -> dict[str, list[str]]:
    try:
        with open(_EVENTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(store: dict[str, list[str]]) -> None:
    with open(_EVENTS_FILE, "w") as f:
        json.dump(store, f)


def push_event(e_tag: str, entry: str) -> None:
    """Append one event and trim to the last _MAX_RECENT."""
    with _lock:
        store = _load()
        buf = store.setdefault(e_tag, [])
        buf.append(entry)
        store[e_tag] = buf[-_MAX_RECENT:]
        _save(store)


def push_batch(e_tag: str, entries: list[str]) -> None:
    """Append multiple events at once and trim to the last _MAX_RECENT."""
    if not entries:
        return
    with _lock:
        store = _load()
        buf = store.setdefault(e_tag, [])
        buf.extend(entries)
        store[e_tag] = buf[-_MAX_RECENT:]
        _save(store)


def get_recent(e_tag: str) -> list[str]:
    """Return up to _MAX_RECENT events in chronological order (oldest → newest)."""
    with _lock:
        return _load().get(e_tag, [])
