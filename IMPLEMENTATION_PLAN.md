# Implementation Plan

## Phase 1: Tool-Use Game Loop + Auto-Resolve

The foundation. Convert the game loop from push-state/parse-JSON to the tool-use pattern, and add the auto-resolve layer to skip trivial decisions.

- Define `play_action` tool schema (params: `index`, `target?`)
- Abstract LLM provider behind an interface (Anthropic first, OpenAI/Gemini/vLLM later)
- Restructure the loop: initial state in first user message, subsequent states as `tool_result`
- Forced tool calling (`tool_choice: any`) with inner loop until `play_action` is called
- Auto-resolve layer: proceed, continue, shop_open, gold/relic rewards, treasure
- Consolidate main menu → character select → embark into one agent decision
- Keep RandomAgent working as a baseline

**Test**: Run against the game with `--agent llm`, verify it plays through a full run with auto-resolved steps chaining correctly.

## Phase 2: Extended Thinking + Stream Output

- Enable extended thinking in the API call
- Extract thinking blocks from responses
- Push reasoning to OBS via obs-websocket browser source events
- Build/update the browser source overlay to render reasoning text
- Fallback for models without thinking support: use text output as reasoning

**Test**: Watch the stream, verify reasoning appears in the OBS overlay and updates each decision.

## Phase 3: Knowledge Base (Two-Level)

- Define `update_knowledge_base` tool schema (params: `store` (in-run|cross-run), `operation`, `key`, `value?`)
- In-run KB: dict in memory, embedded in prompt, cleared between runs
- Cross-run KB: persisted to disk (JSON file), loaded into system prompt
- Process KB tool calls in the inner loop alongside `play_action`
- Post-run reflection: after game over, prompt the agent to reflect and update cross-run KB

**Test**: Play multiple runs, inspect KB files, verify cross-run knowledge accumulates and in-run KB resets.

## Phase 4: Summarization

- Track conversation history length (message count or token estimate)
- Detect triggers: ~30 exchanges or natural game boundaries (combat end, room transition)
- Summarization call: separate API call asking the agent to summarize recent history
- Replace history with compressed summary
- Recursive compression of older summaries

**Test**: Run a long game, verify history stays bounded, agent retains strategic awareness across summarization boundaries.

## Phase 5: Crash Recovery

- Persist full conversation state to disk after each action (message history, in-run KB, summarization state)
- On startup, detect and load saved state to resume mid-run
- Clean up saved state on run completion

**Test**: Kill the client mid-combat, restart, verify it resumes from the last action.

## Phase 6: Prompt Engineering

- Refine system prompt with STS2-specific strategy knowledge
- Tune summarization prompts
- Tune KB usage instructions (what to store, when to update, in-run vs cross-run)
- Iterate based on gameplay results and cross-run KB quality

**Test**: Compare game scores and decision quality across prompt versions.

## Dependencies

```
Phase 1 (core loop + auto-resolve)
  ├── Phase 2 (thinking + stream)
  ├── Phase 3 (knowledge base)
  │     └── Phase 5 (crash recovery — needs KB state to persist)
  └── Phase 4 (summarization — needs stable loop)
Phase 6 (prompt engineering — ongoing, benefits from all prior phases)
```
