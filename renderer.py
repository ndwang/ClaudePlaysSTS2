"""Render GameState into context-appropriate plain text for the LLM."""

from __future__ import annotations
from collections import Counter
from state import (
    GameState, PlayerState, CombatState, Card,
    CardSelectionOverlay, HandSelectionOverlay, RewardsOverlay,
)


def render(gs: GameState) -> str:
    """Produce a decision-focused text briefing from the current game state."""
    parts: list[str] = []

    # Events log (if any)
    if gs.events_log:
        for msg in gs.events_log:
            parts.append(f">> {msg}")
        parts.append("")

    # Dispatch to overlay or context renderer
    if gs.overlay_type == "card_selection":
        parts.append(_render_card_selection(gs))
    elif gs.overlay_type == "hand_select":
        parts.append(_render_hand_selection(gs))
    elif gs.overlay_type == "rewards":
        parts.append(_render_rewards(gs))
    elif gs.context == "combat":
        parts.append(_render_combat(gs))
    elif gs.context == "map":
        parts.append(_render_map(gs))
    elif gs.context == "event":
        parts.append(_render_event(gs))
    elif gs.context == "rest":
        parts.append(_render_rest(gs))
    elif gs.context == "shop":
        parts.append(_render_shop(gs))
    elif gs.context == "treasure":
        parts.append(_render_treasure(gs))
    elif gs.context == "game_over":
        parts.append(_render_game_over(gs))
    elif gs.context == "character_select":
        parts.append(_render_character_select(gs))
    elif gs.context == "main_menu":
        parts.append(_render_main_menu(gs))
    else:
        parts.append(f"Unknown context: {gs.context}")

    # Commands
    parts.append(_render_commands(gs))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Context renderers
# ---------------------------------------------------------------------------

def _render_combat(gs: GameState) -> str:
    c = gs.combat
    p = gs.player
    lines = [f"COMBAT Round {c.round} | HP:{p.hp}/{p.max_hp} Energy:{c.energy} Block:{c.block}"]
    if c.stars:
        lines[0] += f" Stars:{c.stars}"

    # Enemies
    lines.append("\nEnemies:")
    for e in c.enemies:
        intent_parts = []
        for i in e.intents:
            if i.type == "Attack":
                s = f"Attack {i.damage}"
                if i.hits and i.hits > 1:
                    s += f"x{i.hits}"
                intent_parts.append(s)
            else:
                intent_parts.append(i.type)
        intent_str = ", ".join(intent_parts)
        line = f"  [{e.index}] {e.name} {e.hp}/{e.max_hp}"
        if e.block:
            line += f" Block:{e.block}"
        line += f" Intent:[{intent_str}]"
        if e.powers:
            pw_strs = [f"{pw.name} {pw.amount}: {pw.description}" for pw in e.powers]
            line += f" Powers:[{'; '.join(pw_strs)}]"
        lines.append(line)

    # Hand
    lines.append("\nHand:")
    for h in c.hand:
        cost = h.cost if h.cost != -1 else "X"
        mark = " *target*" if h.target_type in ("AnyEnemy", "AnyAlly") else ""
        playable = "" if h.playable else " (unplayable)"
        lines.append(f"  [{h.index}] {h.name}({cost}) {h.description}{mark}{playable}")

    # Orbs (Defect only)
    if c.orb_slots > 0:
        orb_strs = [f"{o.name}(passive:{o.passive_value} evoke:{o.evoke_value})" for o in c.orbs]
        empty = c.orb_slots - len(c.orbs)
        if empty > 0:
            orb_strs.append(f"(empty)x{empty}" if empty > 1 else "(empty)")
        lines.append(f"\nOrbs ({len(c.orbs)}/{c.orb_slots}): {', '.join(orb_strs)}")

    lines.append(f"\nDraw:{c.draw_pile_count} Discard:{c.discard_pile_count} Exhaust:{c.exhaust_pile_count}")

    # Player powers
    if c.powers:
        pw_strs = [f"{pw.name} {pw.amount}: {pw.description}" for pw in c.powers]
        lines.append(f"Powers: {'; '.join(pw_strs)}")

    # Relics (always with descriptions)
    lines.append(_render_relics(p))

    # Potions
    if p.potions:
        lines.append(_render_potions(p))

    return "\n".join(lines)


def _render_card_selection(gs: GameState) -> str:
    ov: CardSelectionOverlay = gs.overlay
    p = gs.player
    lines = ["CARD SELECTION"]
    if ov.can_skip:
        lines[0] += " (can skip)"
    if ov.min_select is not None and ov.max_select is not None:
        lines[0] += f" (pick {ov.min_select}-{ov.max_select})"

    for c in ov.cards:
        cost = f"({c.cost})" if c.cost is not None else ""
        lines.append(f"  [{c.index}] {c.name}{cost} {c.description}")

    lines.append(f"\nHP:{p.hp}/{p.max_hp} Gold:{p.gold}")
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_hand_selection(gs: GameState) -> str:
    ov: HandSelectionOverlay = gs.overlay
    p = gs.player
    lines = [f"HAND SELECTION: {ov.prompt}"]
    lines.append(f"Select {ov.min_select}-{ov.max_select} (selected: {ov.selected_count})")

    for c in ov.cards:
        lines.append(f"  [{c.index}] {c.name} {c.description}")

    lines.append(f"\nHP:{p.hp}/{p.max_hp}")
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_rewards(gs: GameState) -> str:
    ov: RewardsOverlay = gs.overlay
    p = gs.player
    lines = ["REWARDS"]

    for r in ov.rewards:
        lines.append(f"  [{r.index}] {r.reward_type}: {r.description}")

    lines.append(f"\nHP:{p.hp}/{p.max_hp} Gold:{p.gold}")
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_map(gs: GameState) -> str:
    m = gs.map
    p = gs.player
    pos = f"({m.current_coord[0]},{m.current_coord[1]})" if m.current_coord else "start"
    lines = [f"MAP Act {m.act} | HP:{p.hp}/{p.max_hp} Gold:{p.gold}"]
    lines.append(f"Position: {pos}")

    lines.append("\nNodes:")
    for n in m.available_nodes:
        lines.append(f"  [{n.index}] ({n.coord[0]},{n.coord[1]}) {n.node_type}")

    lines.append("")
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_event(gs: GameState) -> str:
    ev = gs.event
    p = gs.player
    lines = [f"EVENT: {ev.title}"]
    lines.append(ev.description)

    if ev.finished:
        lines.append("\n(Event complete)")
    else:
        lines.append("\nOptions:")
        for o in ev.options:
            lock = " [LOCKED]" if o.locked else ""
            desc = f" - {o.description}" if o.description else ""
            lines.append(f"  [{o.index}] {o.label}{desc}{lock}")

    lines.append(f"\nHP:{p.hp}/{p.max_hp} Gold:{p.gold}")
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_rest(gs: GameState) -> str:
    r = gs.rest
    p = gs.player
    lines = [f"REST SITE | HP:{p.hp}/{p.max_hp} Gold:{p.gold}"]

    lines.append("\nOptions:")
    for o in r.options:
        enabled = "" if o.enabled else " [DISABLED]"
        lines.append(f"  [{o.index}] {o.name}: {o.description}{enabled}")

    lines.append("")
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_shop(gs: GameState) -> str:
    s = gs.shop
    p = gs.player
    lines = [f"SHOP | HP:{p.hp}/{p.max_hp} Gold:{p.gold}"]

    if not s.is_open:
        lines.append("(Shop is closed - open it first)")
    else:
        lines.append("\nItems:")
        for it in s.items:
            afford = "" if it.affordable else " [can't afford]"
            desc = f" {it.description}" if it.description else ""
            lines.append(f"  [{it.index}] [{it.item_type}] {it.name} ({it.cost}g){desc}{afford}")

    lines.append("")
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    lines.append(_render_potions(p))
    return "\n".join(lines)


def _render_treasure(gs: GameState) -> str:
    t = gs.treasure
    p = gs.player
    lines = [f"TREASURE | HP:{p.hp}/{p.max_hp} Gold:{p.gold}"]

    for r in t.relics:
        lines.append(f"  [{r.index}] {r.name}: {r.description}")

    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_game_over(gs: GameState) -> str:
    result = "VICTORY" if gs.game_over_victory else "DEFEAT"
    return f"GAME OVER: {result}"


def _render_character_select(gs: GameState) -> str:
    lines = ["CHARACTER SELECT"]
    if gs.selected_character:
        lines.append(f"Currently selected: {gs.selected_character}")
    for ch in gs.characters or []:
        lock = " [LOCKED]" if ch.locked else ""
        lines.append(f"  [{ch.index}] {ch.name}{lock}")
    return "\n".join(lines)


def _render_main_menu(gs: GameState) -> str:
    lines = ["MAIN MENU"]
    if gs.has_saved_run:
        lines.append("Saved run available.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_relics(p: PlayerState) -> str:
    if not p.relics:
        return "Relics: (none)"
    relic_strs = [f"{r.name}: {r.description}" for r in p.relics]
    return "Relics: " + " | ".join(relic_strs)


def _render_potions(p: PlayerState) -> str:
    if not p.potions:
        return "Potions: (none)"
    pot_strs = [f"[{pt.slot}] {pt.name}: {pt.description}" for pt in p.potions]
    return "Potions: " + ", ".join(pot_strs)


def _render_deck(p: PlayerState) -> str:
    """Group duplicate cards, include descriptions."""
    if not p.deck:
        return "Deck(0): (empty)"

    # Group by (name, cost, description)
    counts: Counter[str] = Counter()
    info: dict[str, Card] = {}
    for c in p.deck:
        key = c.name
        counts[key] += 1
        info[key] = c

    parts = []
    for name, count in counts.items():
        c = info[name]
        cost = c.cost if c.cost != -1 else "X"
        entry = f"{name}({cost})x{count}" if count > 1 else f"{name}({cost})"
        entry += f" {c.description}"
        parts.append(entry)

    return f"Deck({len(p.deck)}): " + " | ".join(parts)


def _render_commands(gs: GameState) -> str:
    lines = ["\nCommands:"]
    for i, cmd in enumerate(gs.commands):
        cmd_type = cmd.get("type", "?")
        detail_parts = []
        for k, v in cmd.items():
            if k == "type":
                continue
            detail_parts.append(f"{k}={v}")
        detail = " " + " ".join(detail_parts) if detail_parts else ""
        lines.append(f"  [{i}] {cmd_type}{detail}")
    return "\n".join(lines)
