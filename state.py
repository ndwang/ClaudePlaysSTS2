"""Internal state model for an STS2 run."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Relic:
    name: str
    description: str


@dataclass
class Potion:
    slot: int
    name: str
    description: str


@dataclass
class Card:
    name: str
    description: str
    cost: int | str  # int or "X"


@dataclass
class HandCard:
    index: int
    name: str
    description: str
    cost: int | str
    target_type: str  # "Self", "AnyEnemy", "AllEnemy", etc.
    playable: bool


@dataclass
class Power:
    name: str
    amount: int
    description: str


@dataclass
class Intent:
    type: str  # "Attack", "Debuff", "Buff", "Defend", "Unknown", etc.
    damage: int | None = None
    hits: int | None = None


@dataclass
class Enemy:
    index: int
    name: str
    hp: int
    max_hp: int
    block: int
    powers: list[Power]
    intents: list[Intent]


@dataclass
class Orb:
    index: int
    name: str
    passive_value: int
    evoke_value: int


@dataclass
class CombatState:
    round: int
    energy: int
    stars: int
    block: int
    powers: list[Power]
    hand: list[HandCard]
    enemies: list[Enemy]
    draw_pile_count: int
    discard_pile_count: int
    exhaust_pile_count: int
    orb_slots: int
    orbs: list[Orb]


@dataclass
class MapNode:
    index: int
    coord: tuple[int, int]
    node_type: str


@dataclass
class MapState:
    act: int
    current_coord: tuple[int, int] | None
    available_nodes: list[MapNode]


@dataclass
class EventOption:
    index: int
    label: str
    description: str
    locked: bool


@dataclass
class EventState:
    title: str
    description: str
    options: list[EventOption]
    finished: bool


@dataclass
class RestOption:
    index: int
    id: str
    name: str
    description: str
    enabled: bool


@dataclass
class RestState:
    options: list[RestOption]


@dataclass
class ShopItem:
    index: int
    item_type: str  # "card", "relic", "potion", "card_removal"
    name: str
    description: str
    cost: int
    affordable: bool


@dataclass
class ShopState:
    is_open: bool
    gold: int
    items: list[ShopItem]


@dataclass
class TreasureRelic:
    index: int
    name: str
    description: str


@dataclass
class TreasureState:
    relics: list[TreasureRelic]
    proceed_available: bool


@dataclass
class OverlayCard:
    index: int
    name: str
    description: str
    cost: int | str | None = None


@dataclass
class Reward:
    index: int
    reward_type: str  # "gold", "card", "potion", "relic", "card_removal"
    description: str


@dataclass
class CardSelectionOverlay:
    cards: list[OverlayCard]
    can_skip: bool
    min_select: int | None = None
    max_select: int | None = None


@dataclass
class HandSelectionOverlay:
    prompt: str
    min_select: int
    max_select: int
    selected_count: int
    cards: list[OverlayCard]


@dataclass
class RewardsOverlay:
    rewards: list[Reward]
    can_proceed: bool


@dataclass
class GameOverState:
    victory: bool
    seed: str
    ascension: int
    run_time: float
    floor_reached: int
    killed_by: str | None
    score: int
    character: str
    deck_size: int
    relic_count: int


@dataclass
class CharacterInfo:
    index: int
    name: str
    locked: bool


@dataclass
class PlayerState:
    hp: int = 0
    max_hp: int = 0
    gold: int = 0
    relics: list[Relic] = field(default_factory=list)
    potions: list[Potion] = field(default_factory=list)
    deck: list[Card] = field(default_factory=list)


@dataclass
class GameState:
    """Root state for the entire run. Updated from server snapshots."""

    context: str = ""
    player: PlayerState = field(default_factory=PlayerState)
    combat: CombatState | None = None
    map: MapState | None = None
    event: EventState | None = None
    rest: RestState | None = None
    shop: ShopState | None = None
    treasure: TreasureState | None = None
    overlay: CardSelectionOverlay | HandSelectionOverlay | RewardsOverlay | None = None
    overlay_type: str | None = None
    game_over: GameOverState | None = None
    commands: list[dict[str, Any]] = field(default_factory=list)
    events_log: list[str] = field(default_factory=list)

    # Pre-run
    characters: list[CharacterInfo] | None = None
    selected_character: str | None = None
    has_saved_run: bool = False

    def update(self, raw: dict[str, Any]) -> None:
        """Parse a server state snapshot into the typed model."""
        self.context = raw.get("context", "")
        self.commands = raw.get("available_commands", [])
        self.events_log = [e.get("message", "") for e in raw.get("events", [])]

        # Clear transient state
        self.combat = None
        self.map = None
        self.event = None
        self.rest = None
        self.shop = None
        self.treasure = None
        self.overlay = None
        self.overlay_type = None
        self.game_over = None
        self.characters = None
        self.selected_character = None

        # Player
        if "player" in raw:
            p = raw["player"]
            self.player.hp = p["hp"]
            self.player.max_hp = p["maxHp"]
            self.player.gold = p["gold"]
            self.player.relics = [
                Relic(r["name"], r["description"]) for r in p.get("relics", [])
            ]
            self.player.potions = [
                Potion(pt["slot"], pt["name"], pt["description"])
                for pt in p.get("potions", [])
            ]
            self.player.deck = [
                Card(c["name"], c["description"], c["cost"])
                for c in p.get("deck", [])
            ]

        # Context-specific state
        if "combat" in raw:
            self._parse_combat(raw["combat"])
        if "map" in raw:
            self._parse_map(raw["map"])
        if "event" in raw:
            self._parse_event(raw["event"])
        if "rest" in raw:
            self._parse_rest(raw["rest"])
        if "shop" in raw:
            self._parse_shop(raw["shop"])
        if "treasure" in raw:
            self._parse_treasure(raw["treasure"])
        if "game_over" in raw:
            go = raw["game_over"]
            self.game_over = GameOverState(
                victory=go.get("victory", False),
                seed=go.get("seed", ""),
                ascension=go.get("ascension", 0),
                run_time=go.get("run_time", 0),
                floor_reached=go.get("floor_reached", 0),
                killed_by=go.get("killed_by"),
                score=go.get("score", 0),
                character=go.get("character", ""),
                deck_size=go.get("deck_size", 0),
                relic_count=go.get("relic_count", 0),
            )
        if "character_select" in raw:
            self._parse_character_select(raw["character_select"])
        if "main_menu" in raw:
            self.has_saved_run = raw["main_menu"].get("has_saved_run", False)

        # Overlay
        if "overlay" in raw and raw["overlay"]:
            self._parse_overlay(raw["overlay"])

    def _parse_combat(self, c: dict) -> None:
        enemies = []
        for e in c.get("enemies", []):
            powers = [
                Power(pw["name"], pw["amount"], pw["description"])
                for pw in e.get("powers", [])
            ]
            intents = [
                Intent(
                    i["type"],
                    i.get("damage"),
                    i.get("hits"),
                )
                for i in e.get("intents", [])
            ]
            enemies.append(
                Enemy(e["index"], e["name"], e["hp"], e["maxHp"], e["block"], powers, intents)
            )

        hand = [
            HandCard(
                h["index"], h["name"], h["description"],
                h["cost"], h.get("targetType", "Self"), h.get("playable", False),
            )
            for h in c.get("hand", [])
        ]

        player_powers = [
            Power(pw["name"], pw["amount"], pw["description"])
            for pw in c.get("playerPowers", [])
        ]

        orbs = [
            Orb(o["index"], o["name"], o["passiveValue"], o["evokeValue"])
            for o in c.get("orbs", [])
        ]

        self.combat = CombatState(
            round=c.get("round", 1),
            energy=c.get("energy", 0),
            stars=c.get("stars", 0),
            block=c.get("playerBlock", 0),
            powers=player_powers,
            hand=hand,
            enemies=enemies,
            draw_pile_count=c.get("drawPileCount", 0),
            discard_pile_count=c.get("discardPileCount", 0),
            exhaust_pile_count=c.get("exhaustPileCount", 0),
            orb_slots=c.get("orbSlots", 0),
            orbs=orbs,
        )

    def _parse_map(self, m: dict) -> None:
        coord = m.get("currentCoord")
        current = (coord["row"], coord["col"]) if coord else None
        nodes = [
            MapNode(i, (n["coord"]["row"], n["coord"]["col"]), n["type"])
            for i, n in enumerate(m.get("availableNodes", []))
        ]
        self.map = MapState(
            act=m.get("act", 1),
            current_coord=current,
            available_nodes=nodes,
        )

    def _parse_event(self, ev: dict) -> None:
        has_proceed = any(c.get("type") == "proceed" for c in self.commands)
        options = [
            EventOption(o["index"], o["label"], o.get("description", ""), o.get("locked", False))
            for o in ev.get("options", [])
        ]
        self.event = EventState(
            title=ev.get("title", ""),
            description=ev.get("description", ""),
            options=options,
            finished=has_proceed,
        )

    def _parse_rest(self, r: dict) -> None:
        options = [
            RestOption(o["index"], o["id"], o["name"], o.get("description", ""), o.get("enabled", True))
            for o in r.get("options", [])
        ]
        self.rest = RestState(options=options)

    def _parse_shop(self, s: dict) -> None:
        items = [
            ShopItem(
                it["index"], it["type"], it["name"],
                it.get("description", ""), it["cost"], it.get("affordable", False),
            )
            for it in s.get("items", [])
        ]
        self.shop = ShopState(
            is_open=s.get("isOpen", False),
            gold=s.get("gold", 0),
            items=items,
        )

    def _parse_treasure(self, t: dict) -> None:
        relics = [
            TreasureRelic(r["index"], r["name"], r.get("description", ""))
            for r in t.get("relics", [])
        ]
        self.treasure = TreasureState(
            relics=relics,
            proceed_available=t.get("proceedAvailable", False),
        )

    def _parse_character_select(self, cs: dict) -> None:
        self.characters = [
            CharacterInfo(c["index"], c["name"], c.get("locked", False))
            for c in cs.get("characters", [])
        ]
        self.selected_character = cs.get("selected")

    def _parse_overlay(self, ov: dict) -> None:
        self.overlay_type = ov.get("type")

        if self.overlay_type == "card_selection":
            cards = [
                OverlayCard(c["index"], c["name"], c.get("description", ""), c.get("cost"))
                for c in ov.get("cards", [])
            ]
            self.overlay = CardSelectionOverlay(
                cards=cards,
                can_skip=ov.get("canSkip", False),
                min_select=ov.get("minSelect"),
                max_select=ov.get("maxSelect"),
            )

        elif self.overlay_type == "hand_select":
            cards = [
                OverlayCard(c["index"], c["name"], c.get("description", ""))
                for c in ov.get("cards", [])
            ]
            self.overlay = HandSelectionOverlay(
                prompt=ov.get("prompt", ""),
                min_select=ov.get("minSelect", 1),
                max_select=ov.get("maxSelect", 1),
                selected_count=ov.get("selectedCount", 0),
                cards=cards,
            )

        elif self.overlay_type == "rewards":
            rewards = [
                Reward(r["index"], r["type"], r.get("description", ""))
                for r in ov.get("rewards", [])
            ]
            self.overlay = RewardsOverlay(
                rewards=rewards,
                can_proceed=ov.get("canProceed", False),
            )
