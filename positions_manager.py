"""
Positions manager — simple read/write of positions.json, no threading.

Schema:
{
  "buying_power": float,
  "positions": {
    "<ticker>": {
      "side":      "yes" | "no",
      "contracts": int,
      "avg_price": float
    }
  },
  "trade_history": [
    { "ts": int, "ticker": str, "action": str, "contracts": int, "price": float }
  ]
}
"""

import datetime
import json
import time

_FILE = "positions.json"


def load() -> dict:
    try:
        with open(_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"buying_power": 0.0, "positions": {}, "trade_history": []}


def _save(state: dict) -> None:
    with open(_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_buying_power() -> float:
    return load().get("buying_power", 0.0)


def format_positions_context(ticker: str = "") -> str:
    """
    Return a plain-text summary of current positions and recent trade history.
    If ticker is provided, highlights whether a position exists for that market.
    """
    state   = load()
    bp      = state.get("buying_power", 0.0)
    pos     = state.get("positions", {})
    history = state.get("trade_history", [])

    lines = [f"Buying power: ${bp:.2f}"]

    # Open positions
    if not pos:
        lines.append("Open positions: none")
    else:
        lines.append("Open positions:")
        for t, p in pos.items():
            side  = p.get("side", "?").upper()
            qty   = p.get("contracts", 0)
            avg   = p.get("avg_price", 0.0)
            tag   = " ← THIS MARKET" if ticker and t == ticker else ""
            lines.append(f"  {t} ({side}): {qty} contracts @ avg ${avg:.2f}{tag}")

    # For this specific ticker: position detail with unrealized P&L
    if ticker:
        if ticker in pos:
            held      = pos[ticker]
            avg_price = held.get("avg_price", 0.0)
            qty       = held.get("contracts", 0)
            side      = held["side"]
            lines.append(
                f"For {ticker}: you hold {qty} {side.upper()} contracts bought at avg ${avg_price:.2f}. "
                f"You may SELL {side.upper()} or BUY more. "
                f"(Unrealized P&L will be shown in signal context.)"
            )
        else:
            lines.append(
                f"For {ticker}: no open position. You may BUY YES or BUY NO. "
                f"You CANNOT sell — you hold nothing."
            )

    # Recent trade history (last 5 trades)
    if history:
        recent = history[-5:]
        lines.append("Recent trades (oldest → newest):")
        for tr in recent:
            ts  = datetime.datetime.utcfromtimestamp(tr["ts"]).strftime("%H:%M:%S")
            lines.append(
                f"  [{ts}] {tr['action']} {tr['contracts']}x {tr['ticker']} @ ${tr['price']:.2f}"
            )
    else:
        lines.append("Recent trades: none")

    return "\n".join(lines)


def record_trade(ticker: str, action: str, contracts: int, price: float) -> str:
    """
    Update positions.json for a trade.
    Returns a simulated order id.
    """
    state = load()
    pos   = state.setdefault("positions", {})
    side  = "yes" if "YES" in action else "no"
    is_buy = action.startswith("BUY")
    cost   = price * contracts

    if is_buy:
        state["buying_power"] = round(state.get("buying_power", 0.0) - cost, 4)
        if ticker in pos and pos[ticker]["side"] == side:
            existing = pos[ticker]
            old_qty  = existing["contracts"]
            new_qty  = old_qty + contracts
            new_avg  = round((existing["avg_price"] * old_qty + price * contracts) / new_qty, 4)
            existing["contracts"] = new_qty
            existing["avg_price"] = new_avg
        else:
            pos[ticker] = {"side": side, "contracts": contracts, "avg_price": round(price, 4)}
    else:
        state["buying_power"] = round(state.get("buying_power", 0.0) + cost, 4)
        if ticker in pos:
            new_qty = pos[ticker]["contracts"] - contracts
            if new_qty <= 0:
                del pos[ticker]
            else:
                pos[ticker]["contracts"] = new_qty

    order_id = f"sim-{int(time.time() * 1000)}"
    state.setdefault("trade_history", []).append({
        "ts": int(time.time()), "ticker": ticker,
        "action": action, "contracts": contracts,
        "price": round(price, 4), "order_id": order_id,
    })

    _save(state)
    return order_id
