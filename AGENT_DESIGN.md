# Agent Architecture Design

## Goal

Build an LLM agent that plays Slay the Spire 2 competently, with visible reasoning for a livestream audience.

## Inspiration

Follows the same philosophy as Claude Plays Pokemon: simplicity over complexity, agent autonomy over rigid structure. The agent decides what to remember rather than the developer imposing a memory schema.

## Game Loop

The loop mirrors Claude Plays Pokemon's structure. Game state is delivered as tool results, not pushed as separate user messages.

```
1. Initial prompt contains first state briefing (from /state/wait)
2. Claude calls play_action(index=2, target=0)
3. Client executes action via POST /action, then polls /state/wait
4. New state briefing returned as tool_result
5. Claude calls play_action(...) again
6. ...repeat
```

If Claude responds with text but no tool call, the response is not added to history and the same context is retried — identical to the Pokemon approach. This works because observations and actions are coupled: you only get a new state by calling a tool.

## Context Management

Two mechanisms:

### Conversation History + Summarization

Full conversation history accumulates as the agent makes decisions. When it gets long (~30 exchanges, or at natural game boundaries like combat end), the agent is asked to write a summary. The history is then cleared and replaced with the compressed summary. Older summaries get recursively compressed, creating a telescoping effect: recent actions in full detail, older ones condensed.

STS2 state snapshots are nearly complete — each briefing includes the full hand, enemies, energy, deck, relics, etc. — so the agent needs less conversational memory than you'd expect. It can reason about the current tactical situation from a single snapshot. Memory matters most for strategic continuity across the run.

### Two-Level Knowledge Base

**In-run KB**: A key-value dictionary the agent manages via tool calls during a run. Tracks deck strategy, fight takeaways, pathing plans — whatever the agent finds useful for the current run. Embedded in every prompt, survives summarization. Cleared between runs.

**Cross-run KB**: Persistent strategic knowledge that carries across runs. Stored to disk. The agent writes to it during post-run reflection (victory or defeat) and can also update it mid-run. Loaded into every run's system prompt. Examples: "Strength builds need block by Act 2", "avoid elites below 30 HP."

Both are agent-controlled via an `update_knowledge_base` tool with add/edit/delete operations. No hardcoded schema imposed by the developer.

## Model-Agnostic Design

The agent should not be locked to Claude. The core loop (tool-use, forced tool calling, knowledge base) works across providers. The LLM call should be abstracted behind an interface so provider-specific features (like extended thinking) are handled per-provider while the rest is shared.

## Extended Thinking for Stream

Where supported, use the model's thinking/reasoning feature to surface reasoning on stream:
- Anthropic: extended thinking (`thinking` blocks)
- OpenAI: reasoning via o-series models
- Gemini: thinking mode
- Open-source (vLLM): not yet available

Thinking blocks are extracted and displayed on stream for viewers. The action output remains clean and parseable. For models without thinking support, the agent's text output (alongside tool calls) can serve as visible reasoning instead.

## Prompt Assembly

Each LLM call receives:

```
System: game rules + strategy guide (static)

[Knowledge base contents — persistent, agent-controlled]
[Summarized older history — if any]
[Recent conversation history — tool_use/tool_result pairs]
[Current state — either initial briefing or latest tool_result]
```

## Tools Available to Agent

| Tool | Purpose |
|------|---------|
| `play_action` | Submit the chosen action index + optional target. Returns the next game state as tool_result. |
| `update_knowledge_base` | Add, edit, or delete entries in the in-run or cross-run knowledge base. |

### Forced Tool Calling

The API is called with forced tool use so the agent must always call at least one tool (no text-only responses):

| Provider | Parameter |
|---|---|
| Anthropic | `tool_choice: {"type": "any"}` |
| OpenAI | `tool_choice: "required"` |
| Gemini | `function_calling_config: {mode: "ANY"}` |
| vLLM (open-source) | `tool_choice: "required"` (OpenAI-compatible) |

If the agent calls `update_knowledge_base` but not `play_action`, we process the KB update, return the result, and loop until `play_action` is called.

## Auto-Resolve Layer

A rule-based filter that handles trivial decisions without calling the LLM. Runs in a tight loop — execute, poll next state, check again — so multiple trivial steps chain automatically.

### Auto-resolve (no LLM call)

- `proceed` — forced advancement
- `continue` — game over screens
- `shop_open` — mechanical prerequisite before agent sees the shop
- Gold rewards — always claim
- Relic rewards — always claim (log it so the agent sees what it got)
- Treasure proceed — relics already claimed

### Consolidate (one LLM call for multiple steps)

- Main menu → character select → embark: agent picks character, rest is automated

### Agent decides

- Combat (play_card, end_turn, use_potion)
- Card reward selection (pick vs skip, which card)
- Potion rewards (slot management — slots are limited)
- Map node selection
- Event options
- Rest site options
- Shop purchases (buy vs leave)
- Hand selection (discard/exhaust)

## Summarization Triggers

- Conversation history exceeds ~30 exchanges
- Natural game boundaries (combat end, room transition) when history is non-trivial

On trigger:
1. Agent writes a summary of the recent history
2. History is cleared, summary is prepended to next prompt
3. Older summaries are recursively compressed

## Crash Recovery

The full conversation state (message history, in-run KB, summarization state) is persisted to disk after each action. On startup, if a saved state exists, the agent resumes from where it left off rather than starting fresh.

## Stream Output

Agent reasoning is pushed to OBS via the existing obs-websocket integration as browser source events — no terminal capture needed. The game state is already visible on screen; only the agent's thinking/reasoning text needs to be overlaid.

## Post-Run Reflection

At the end of each run (victory or defeat), the agent is prompted to reflect and update the cross-run KB. Over multiple runs the agent builds up strategic knowledge from experience.

## What Changes

- `llm.py` — Tool-use loop, conversation history with summarization, knowledge base, extended thinking
- `prompts.py` — System prompt with knowledge base and summarization instructions
- `client.py` — Simplified: hands control to the agent loop, extracts thinking for stream display
- `obs.py` — Display agent reasoning on stream
