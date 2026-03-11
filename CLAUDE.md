# STS2 Client — LLM Agent for Slay the Spire 2

## Project Overview

The goal is to build an LLM agent that can play Slay the Spire 2 well. A companion C# mod (`../sts2agent/`) exposes game state and accepts actions via an HTTP API. This Python client implements the agent logic — the game loop, state parsing, and LLM decision-making. The random agent exists as a baseline; the real work is developing the LLM agent (prompts, strategy, context management) to play competently.

The agent's gameplay is livestreamed, showing both the game and the agent's internal reasoning. The OBS overlay integration supports this by displaying stats (score, rounds, timer) on stream.

The API reference is at `../sts2agent/API.md`. The agent architecture design is in `AGENT_DESIGN.md`.

## Architecture

```
client.py       Main loop: wait for state → render → decide → act
api.py          HTTP client wrapping the mod's REST API (localhost:57541)
state.py        Typed dataclass model parsed from JSON state snapshots
renderer.py     Renders GameState into plain-text briefings for LLM consumption
llm.py          Agent interface + RandomAgent + LLMAgent (Anthropic Claude)
prompts.py      System prompt for the LLM agent
obs.py          OBS overlay integration (score, rounds, timer via obs-websocket)
obs/            OBS browser source assets (overlay.html, state.json)
```

## Key Concepts

- **Game loop**: `client.py` polls `/state/wait` for decision points. Each decision point yields a state snapshot with `available_commands`. The agent picks a command index (and optional target), which is POSTed to `/action`.
- **Agent interface** (`llm.py`): `Agent.decide(gs, briefing) -> {"action": idx, "target"?: idx}`. `reset()` is called between runs.
- **RandomAgent**: Picks random commands, preferring non-terminal actions (plays cards before ending turn).
- **LLMAgent**: Sends the rendered briefing as a user message to Claude, expects a JSON response with action index.
- **Renderer** (`renderer.py`): Converts typed `GameState` into concise text. Each context (combat, map, event, shop, etc.) has its own renderer. Output is used both for console display and as LLM input.
- **State model** (`state.py`): `GameState.update(raw)` parses server JSON into typed dataclasses. Transient fields are cleared on each update.
- **Overlays**: Card selection, hand selection, and rewards appear on top of room contexts. When active, `overlay_type` is set and commands belong to the overlay.

## Game Contexts

Main menu → Character select → Map → (Combat | Event | Rest | Shop | Treasure) → Map → ... → Game Over

Overlays (card_selection, hand_select, rewards) appear on top of combat/other contexts.

## Running

```bash
# Random agent (for testing)
uv run python client.py --agent random --speed fast

# LLM agent
uv run python client.py --agent llm --model claude-sonnet-4-20250514

# With OBS overlay and logging
uv run python client.py --agent llm --obs-password SECRET --log sts2agent.log
```

Requires the STS2 mod to be running (game must be open with mod loaded).

## Development Notes

- Python 3.12+, managed with `uv`
- Dependencies: `anthropic`, `requests`, `obsws-python`
- The LLM agent uses the Anthropic SDK directly with conversation history (cleared on `reset()`)
- `ANTHROPIC_API_KEY` env var must be set for LLM agent
- The briefing text doubles as both human-readable console output and LLM prompt input — keep it concise and information-dense
- Card `cost` can be int or `"X"` (or -1 internally for X-cost cards)
- Enemy `targetIndex` indexes into the alive-enemies array
- The mod server is in Chinese locale — some strings (character names, etc.) are localized
