"""Render GameState into context-appropriate plain text for the LLM."""

from __future__ import annotations
from collections import Counter
from i18n import t
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
        parts.append(t("unknown_context", context=gs.context))

    # Commands
    parts.append(_render_commands(gs))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Context renderers
# ---------------------------------------------------------------------------

def _render_combat(gs: GameState) -> str:
    c = gs.combat
    p = gs.player
    header = t("combat.header", round=c.round, hp=p.hp, max_hp=p.max_hp, energy=c.energy, block=c.block)
    if c.stars:
        header += t("combat.stars", stars=c.stars)
    for comp in p.companions:
        header += t("combat.companion", name=comp.name, hp=comp.hp, max_hp=comp.max_hp)
        if comp.block:
            header += t("combat.companion_block", block=comp.block)
    lines = [header]

    # Enemies
    lines.append(t("combat.enemies"))
    for e in c.enemies:
        intent_parts = []
        for i in e.intents:
            if i.type == "Attack":
                dmg = i.damage // i.hits if i.hits and i.hits > 1 else i.damage
                s = t("combat.attack", damage=dmg)
                if i.hits and i.hits > 1:
                    s += t("combat.hits", hits=i.hits)
                intent_parts.append(s)
            else:
                intent_parts.append(i.type)
        intent_str = ", ".join(intent_parts)
        line = t("combat.enemy_line", index=e.index, name=e.name, hp=e.hp, max_hp=e.max_hp)
        if e.block:
            line += t("combat.block", block=e.block)
        line += t("combat.intent", intent=intent_str)
        if e.powers:
            pw_strs = [f"{pw.name} {pw.amount}: {pw.description}" for pw in e.powers]
            line += t("combat.powers_suffix", powers='; '.join(pw_strs))
        lines.append(line)

    # Hand
    lines.append(t("combat.hand"))
    for h in c.hand:
        cost = h.cost if h.cost != -1 else "X"
        mark = t("combat.target") if h.target_type in ("AnyEnemy", "AnyAlly") else ""
        playable = "" if h.playable else t("combat.unplayable")
        lines.append(t("combat.card_line", index=h.index, name=h.name, cost=cost, desc=h.description, mark=mark, playable=playable))

    # Orbs (Defect only)
    if c.orb_slots > 0:
        orb_strs = [f"{o.name}(passive:{o.passive_value} evoke:{o.evoke_value})" for o in c.orbs]
        empty = c.orb_slots - len(c.orbs)
        if empty > 0:
            orb_strs.append(t("combat.empty_orbs", count=empty) if empty > 1 else t("combat.empty_orb"))
        lines.append(t("combat.orbs", count=len(c.orbs), slots=c.orb_slots, orbs=', '.join(orb_strs)))

    lines.append(t("combat.piles", draw=c.draw_pile_count, discard=c.discard_pile_count, exhaust=c.exhaust_pile_count))

    # Player powers
    if c.powers:
        pw_strs = [f"{pw.name} {pw.amount}: {pw.description}" for pw in c.powers]
        lines.append(t("combat.powers", powers='; '.join(pw_strs)))

    # Relics (always with descriptions)
    lines.append(_render_relics(p))

    # Potions
    if p.potions:
        lines.append(_render_potions(p))

    return "\n".join(lines)


def _render_card_selection(gs: GameState) -> str:
    ov: CardSelectionOverlay = gs.overlay
    p = gs.player
    header = t("card_sel.header")
    if ov.can_skip:
        header += t("card_sel.can_skip")
    if ov.min_select is not None and ov.max_select is not None:
        header += t("card_sel.pick_range", min=ov.min_select, max=ov.max_select)
    lines = [header]

    for c in ov.cards:
        cost = f"({c.cost})" if c.cost is not None else ""
        lines.append(f"  [{c.index}] {c.name}{cost} {c.description}")

    lines.append(t("card_sel.hp_gold", hp=p.hp, max_hp=p.max_hp, gold=p.gold))
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_hand_selection(gs: GameState) -> str:
    ov: HandSelectionOverlay = gs.overlay
    p = gs.player
    lines = [t("hand_sel.header", prompt=ov.prompt)]
    lines.append(t("hand_sel.select_range", min=ov.min_select, max=ov.max_select, selected=ov.selected_count))

    for c in ov.cards:
        lines.append(f"  [{c.index}] {c.name} {c.description}")

    lines.append(t("hand_sel.hp", hp=p.hp, max_hp=p.max_hp))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_rewards(gs: GameState) -> str:
    ov: RewardsOverlay = gs.overlay
    p = gs.player
    lines = [t("rewards.header")]

    for r in ov.rewards:
        lines.append(f"  [{r.index}] {r.reward_type}: {r.description}")

    lines.append(t("rewards.hp_gold", hp=p.hp, max_hp=p.max_hp, gold=p.gold))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_map(gs: GameState) -> str:
    m = gs.map
    p = gs.player
    pos = f"({m.current_coord[0]},{m.current_coord[1]})" if m.current_coord else t("map.start")
    lines = [t("map.header", act=m.act, hp=p.hp, max_hp=p.max_hp, gold=p.gold)]
    lines.append(t("map.position", pos=pos))

    lines.append(t("map.nodes"))
    for n in m.available_nodes:
        lines.append(t("map.node_line", index=n.index, x=n.coord[0], y=n.coord[1], type=n.node_type))

    lines.append("")
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_event(gs: GameState) -> str:
    ev = gs.event
    p = gs.player
    lines = [t("event.header", title=ev.title)]
    lines.append(ev.description)

    if ev.finished:
        lines.append(t("event.complete"))
    else:
        lines.append(t("event.options"))
        for o in ev.options:
            lock = t("event.locked") if o.locked else ""
            desc = t("event.option_desc", desc=o.description) if o.description else ""
            lines.append(f"  [{o.index}] {o.label}{desc}{lock}")

    lines.append(t("card_sel.hp_gold", hp=p.hp, max_hp=p.max_hp, gold=p.gold))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_rest(gs: GameState) -> str:
    r = gs.rest
    p = gs.player
    lines = [t("rest.header", hp=p.hp, max_hp=p.max_hp, gold=p.gold)]

    lines.append(t("rest.options"))
    for o in r.options:
        enabled = "" if o.enabled else t("rest.disabled")
        lines.append(f"  [{o.index}] {o.name}: {o.description}{enabled}")

    lines.append("")
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_shop(gs: GameState) -> str:
    s = gs.shop
    p = gs.player
    lines = [t("shop.header", hp=p.hp, max_hp=p.max_hp, gold=p.gold)]

    if not s.is_open:
        lines.append(t("shop.closed"))
    else:
        lines.append(t("shop.items"))
        for it in s.items:
            afford = "" if it.affordable else t("shop.cant_afford")
            desc = f" {it.description}" if it.description else ""
            lines.append(f"  [{it.index}] [{it.item_type}] {it.name} ({it.cost}g){desc}{afford}")

    lines.append("")
    lines.append(_render_deck(p))
    lines.append(_render_relics(p))
    lines.append(_render_potions(p))
    return "\n".join(lines)


def _render_treasure(gs: GameState) -> str:
    tr = gs.treasure
    p = gs.player
    lines = [t("treasure.header", hp=p.hp, max_hp=p.max_hp, gold=p.gold)]

    for r in tr.relics:
        lines.append(f"  [{r.index}] {r.name}: {r.description}")

    lines.append(_render_relics(p))
    return "\n".join(lines)


def _render_game_over(gs: GameState) -> str:
    go = gs.game_over
    result = t("gameover.victory") if go.victory else t("gameover.defeat")
    lines = [t("gameover.header", result=result)]
    lines.append(t("gameover.character", character=go.character, score=go.score))
    lines.append(t("gameover.floor", floor=go.floor_reached, seed=go.seed, ascension=go.ascension))
    minutes = int(go.run_time) // 60
    seconds = int(go.run_time) % 60
    lines.append(t("gameover.time", minutes=minutes, seconds=seconds, deck_size=go.deck_size, relic_count=go.relic_count))
    if go.killed_by:
        lines.append(t("gameover.killed_by", killed_by=go.killed_by))
    return "\n".join(lines)


def _render_character_select(gs: GameState) -> str:
    lines = [t("charsel.header")]
    if gs.selected_character:
        lines.append(t("charsel.selected", name=gs.selected_character))
    for ch in gs.characters or []:
        lock = t("event.locked") if ch.locked else ""
        lines.append(f"  [{ch.index}] {ch.name}{lock}")
    return "\n".join(lines)


def _render_main_menu(gs: GameState) -> str:
    lines = [t("menu.header")]
    if gs.has_saved_run:
        lines.append(t("menu.saved_run"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_relics(p: PlayerState) -> str:
    if not p.relics:
        return t("relics.none")
    relic_strs = [f"{r.name}: {r.description}" for r in p.relics]
    return t("relics.header") + " | ".join(relic_strs)


def _render_potions(p: PlayerState) -> str:
    if not p.potions:
        return t("potions.none")
    pot_strs = [f"[{pt.slot}] {pt.name}: {pt.description}" for pt in p.potions]
    return t("potions.header") + ", ".join(pot_strs)


def _render_deck(p: PlayerState) -> str:
    """Group duplicate cards, include descriptions."""
    if not p.deck:
        return t("deck.empty")

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

    return t("deck.header", count=len(p.deck)) + " | ".join(parts)


def _render_commands(gs: GameState) -> str:
    lines = [t("commands.header")]
    for i, cmd in enumerate(gs.commands):
        lines.append(f"  [{i}] {format_command(cmd)}")
    return "\n".join(lines)


def _enemy_name(enemies: list | None, index: int) -> str:
    """Look up enemy name by index, falling back to the raw index."""
    if enemies:
        for e in enemies:
            if e.index == index:
                return f"{e.name}[{index}]"
    return f"[{index}]"


def format_command(cmd: dict, enemies: list | None = None) -> str:
    """Format a single command into a human-readable string."""
    cmd_type = cmd.get("type", "?")

    if cmd_type == "play_card":
        name = cmd.get("card", "?")
        if "targetIndex" in cmd:
            target_name = _enemy_name(enemies, cmd["targetIndex"])
            return t("cmd.play_card_targeted", label=name, target=target_name)
        if cmd.get("requiresTarget"):
            return t("cmd.play_card_target", label=name)
        return t("cmd.play_card", label=name)

    if cmd_type == "end_turn":
        return t("cmd.end_turn")

    if cmd_type == "use_potion":
        name = cmd.get("potion", "?")
        if "targetIndex" in cmd:
            target_name = _enemy_name(enemies, cmd["targetIndex"])
            return t("cmd.use_potion_targeted", label=name, target=target_name)
        if cmd.get("requiresTarget"):
            return t("cmd.use_potion_target", label=name)
        return t("cmd.use_potion", label=name)

    if cmd_type == "select_map_node":
        coord = cmd.get("coord", {})
        node_type = cmd.get("nodeType", "?")
        return t("cmd.select_map_node", type=node_type, row=coord.get("row", "?"), col=coord.get("col", "?"))

    if cmd_type == "select_event_option":
        return t("cmd.select_event_option", label=cmd.get("label", "?"))

    if cmd_type == "proceed":
        return t("cmd.proceed")

    if cmd_type == "continue":
        return t("cmd.continue")

    if cmd_type == "rest_option":
        return cmd.get("name", cmd.get("option", "?"))

    if cmd_type == "shop_open":
        return t("cmd.shop_open")

    if cmd_type == "shop_buy":
        return t("cmd.shop_buy", label=cmd.get("name", "?"))

    if cmd_type == "shop_leave":
        return t("cmd.shop_leave")

    if cmd_type == "select_card":
        return t("cmd.select_card", label=cmd.get("card", "?"))

    if cmd_type == "skip":
        return t("cmd.skip")

    if cmd_type == "choose_hand_cards":
        return t("cmd.choose_hand_cards", label=cmd.get("card", "?"))

    if cmd_type == "confirm_selection":
        return t("cmd.confirm_selection")

    if cmd_type == "select_reward":
        return t("cmd.select_reward", label=cmd.get("reward", "?"))

    if cmd_type == "select_character":
        return t("cmd.select_character", label=cmd.get("name", "?"))

    if cmd_type == "embark":
        return t("cmd.embark")

    if cmd_type == "start_run":
        return t("cmd.start_run")

    if cmd_type == "continue_run":
        return t("cmd.continue_run")

    if cmd_type == "abandon_run":
        return t("cmd.abandon_run")

    # Fallback: raw key dump for unknown command types
    detail_parts = [f"{k}={v}" for k, v in cmd.items() if k != "type"]
    detail = " " + " ".join(detail_parts) if detail_parts else ""
    return f"{cmd_type}{detail}"
