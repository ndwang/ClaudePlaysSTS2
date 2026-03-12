# Agent Architecture Design

## Goal

Build an LLM agent that plays Slay the Spire 2 competently, with visible reasoning for a livestream audience.

## Inspiration

Follows the same philosophy as Claude Plays Pokemon: simplicity over complexity, agent autonomy over rigid structure. The agent decides what to remember rather than the developer imposing a memory schema.

## Game Loop

Game state is delivered as tool results, not pushed as separate user messages.

```
1. Initial prompt contains first state briefing (from /state/wait)
2. Agent calls play_action(index=2, target=0)
3. Client executes action via POST /action, then polls /state/wait
4. New state briefing returned as tool_result
5. Agent calls play_action(...) again
6. ...repeat
```

If the agent responds without calling a tool (possible when thinking is enabled and `tool_choice` is `auto`), the response is kept in history and a "you must call a tool" message is appended, then retried up to `MAX_RETRIES` times.

## Context Management

Two mechanisms:

### Conversation History + Summarization

Full conversation history accumulates as the agent makes decisions. When the message count exceeds a threshold (~60 messages / ~30 exchanges), the agent is asked to write a summary. The history is then cleared and replaced with the compressed summary, which is prepended to the next briefing.

STS2 state snapshots are nearly complete — each briefing includes the full hand, enemies, energy, deck, relics, etc. — so the agent needs less conversational memory than you'd expect. It can reason about the current tactical situation from a single snapshot. Memory matters most for strategic continuity across the run.

### Two-Level Knowledge Base

**In-run KB**: A key-value dictionary the agent manages via tool calls during a run. Tracks deck strategy, fight takeaways, pathing plans — whatever the agent finds useful for the current run. Embedded in every system prompt, survives summarization. Cleared between runs.

**Cross-run KB**: Persistent strategic knowledge that carries across runs. Stored to disk (`knowledge.json`). The agent writes to it during post-run reflection (victory or defeat) and can also update it mid-run. Loaded into every system prompt. Examples: "Strength builds need block by Act 2", "avoid elites below 30 HP."

Both are agent-controlled via an `update_knowledge_base` tool with `set` and `delete` operations. No hardcoded schema imposed by the developer.

## Multi-Provider Support

The agent is not locked to Claude. The `backends.py` module abstracts the LLM call behind an `LLMBackend` interface with two implementations:

- **AnthropicBackend** — for `claude-*` models, using the Anthropic SDK with prompt caching and streaming
- **OpenAIBackend** — for `gpt-*`, `o*` models, and any OpenAI-compatible endpoint (Ollama, vLLM, Together, LM Studio, etc.)

Provider-specific features (extended thinking, reasoning effort, prompt caching, message format) are handled per-backend while the agent logic in `llm.py` stays provider-agnostic.

## Extended Thinking for Stream

Where supported, the model's thinking/reasoning is streamed to the console and OBS overlay in real time:
- **Anthropic**: extended thinking (`thinking` blocks), enabled via `--thinking-budget`
- **OpenAI**: reasoning content from o-series and GPT-5 models

When thinking is enabled, `tool_choice` must be set to `auto` (Anthropic doesn't allow forced tool use with thinking). The agent retries if the model doesn't call a tool. For models without thinking, the agent's text output serves as visible reasoning instead.

## Prompt Assembly

Each LLM call receives:

```
System prompt:
  - Game rules + strategy guide (static, from i18n)
  - Cross-run knowledge base contents
  - In-run knowledge base contents

Messages:
  - [Summary of older history — if summarization has occurred]
  - [Recent conversation history — tool_use/tool_result pairs]
  - [Current state briefing — as tool_result or initial user message]
```

## Tools Available to Agent

| Tool | Purpose |
|------|---------|
| `play_action` | Submit the chosen action index + optional target. Returns the next game state as tool_result. |
| `update_knowledge_base` | Set or delete entries in the in-run or cross-run knowledge base. |

### Tool Calling Strategy

Without thinking: `tool_choice: any` (Anthropic) / `required` (OpenAI) — the agent must call a tool every turn.

With thinking enabled: `tool_choice: auto` — the agent may respond with text only. If it does, the response is kept in history with a retry prompt until it calls `play_action`.

If the agent calls `update_knowledge_base` but not `play_action`, the KB update is processed, the result returned, and the loop continues until `play_action` is called.

## Auto-Resolve Layer

A rule-based filter that handles trivial decisions without calling the LLM. Runs in a tight loop — execute, poll next state, check again — so multiple trivial steps chain automatically.

### Auto-resolve (no LLM call)

- `proceed` / `continue` — forced advancement, game over screens
- `main_menu` — auto-selects start or continue run
- `shop_open` — mechanical prerequisite before agent sees the shop
- Gold rewards — always claim
- Relic rewards — always claim

### Agent decides

- Combat (play_card, end_turn, use_potion)
- Card reward selection (pick vs skip, which card)
- Potion rewards (slot management — slots are limited)
- Map node selection
- Event options
- Rest site options
- Shop purchases (buy vs leave)
- Hand selection (discard/exhaust)
- Character selection

## Crash Recovery

The full conversation state (message history, pending tool results, in-run KB, summary) is persisted to `run_state.json` after each successful action. On startup, if a saved state exists, the agent resumes from where it left off. Old state files are archived (not deleted) when a new run begins.

## Stream Output

Agent reasoning is streamed to OBS via obs-websocket as it's generated. The `on_reasoning_delta` callback pushes thinking text to both the console and the OBS overlay in real time. Stats (cost, rounds, model name, knowledge base) are also pushed to the overlay.

## Post-Run Reflection

At game over, the agent is prompted to reflect on the run and update the cross-run KB via `update_knowledge_base` tool calls. Over multiple runs the agent builds up strategic knowledge from experience. Reflection uses `tool_choice: auto` so the agent can mix text responses with KB updates.
