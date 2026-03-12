"""Benchmark Claude API using the actual STS2 agent prompt and tools.

Modes:
  single (default): One briefing per run, measures baseline TTFT/throughput.
  multi: Simulates N turns of conversation accumulation, showing how
         TTFT and throughput degrade as history grows.
"""

import argparse
import random
import time
import anthropic

from i18n import t, set_lang
from llm import _build_tools

# Realistic briefings matching the actual renderer output format.
# These are modeled on real log output from render(gs).
BRIEFINGS = [
    # Combat: multi-enemy, various cards, powers, potions
    """COMBAT Round 1 | HP:66/75 Energy:3 Block:0 Stars:3

Enemies:
  [0] 噬尸蛞蝓 25/25 Intent:[Attack 8] Powers:[饥饿 4: 当有敌人死亡时，噬尸蛞蝓会立即吃下尸体，在本回合被击晕然后获得1点力量。]
  [1] 噬尸蛞蝓 26/26 Intent:[Debuff] Powers:[饥饿 4: 当有敌人死亡时，噬尸蛞蝓会立即吃下尸体，在本回合被击晕然后获得1点力量。]

Hand:
  [0] 崇拜(1) 获得辉星辉星。
  [1] 星星点点(1) 获得8点格挡。\n获得2点活力。
  [2] 防御(1) 获得5点格挡。
  [3] 打击+(1) 造成9点伤害。 *target*
  [4] 陨星(0) 造成7点伤害。\n给予1层虚弱。\n给予1层易伤。 *target*

Draw:6 Discard:0 Exhaust:0
Relics: 天赋君权: 在每场战斗开始时，获得辉星辉星辉星。 | 橙型香盒: 拾起时，升级一张牌。

Commands:
  [0] play_card cardIndex=0 card=崇拜 requiresTarget=False
  [1] play_card cardIndex=1 card=星星点点 requiresTarget=False
  [2] play_card cardIndex=2 card=防御 requiresTarget=False
  [3] play_card cardIndex=3 card=打击+ requiresTarget=True
  [4] play_card cardIndex=4 card=陨星 requiresTarget=True
  [5] end_turn""",

    # Combat: mid-fight, fewer cards, powers active
    """COMBAT Round 2 | HP:66/75 Energy:3 Block:0 Stars:3

Enemies:
  [0] 噬尸蛞蝓 5/25 Intent:[Debuff] Powers:[饥饿 4: 当有敌人死亡时，噬尸蛞蝓会立即吃下尸体，在本回合被击晕然后获得1点力量。]
  [1] 噬尸蛞蝓 26/26 Intent:[Attack 6x2] Powers:[饥饿 4: 当有敌人死亡时，噬尸蛞蝓会立即吃下尸体，在本回合被击晕然后获得1点力量。]

Hand:
  [0] 防御(1) 获得3点格挡。
  [1] 打击(1) 造成8点伤害。 *target*
  [2] 防御(1) 获得3点格挡。
  [3] 打击(1) 造成8点伤害。 *target*
  [4] 防御(1) 获得3点格挡。

Draw:1 Discard:5 Exhaust:0
Powers: 活力 2: 你的下一张攻击牌伤害增加。; 脆弱 2: 脆弱时，从卡牌中获得的格挡值减少25%。
Relics: 天赋君权: 在每场战斗开始时，获得辉星辉星辉星。 | 橙型香盒: 拾起时，升级一张牌。

Commands:
  [0] play_card cardIndex=0 card=防御 requiresTarget=False
  [1] play_card cardIndex=1 card=打击 requiresTarget=True
  [2] play_card cardIndex=2 card=防御 requiresTarget=False
  [3] play_card cardIndex=3 card=打击 requiresTarget=True
  [4] play_card cardIndex=4 card=防御 requiresTarget=False
  [5] end_turn""",

    # Combat: end of turn, only end_turn available
    """COMBAT Round 1 | HP:66/75 Energy:0 Block:8 Stars:3

Enemies:
  [0] 噬尸蛞蝓 5/25 Intent:[Attack 6] Powers:[饥饿 4: 当有敌人死亡时，噬尸蛞蝓会立即吃下尸体，在本回合被击晕然后获得1点力量。; 虚弱 1: 虚弱的生物造成的攻击伤害减少25%。; 易伤 1: 易伤的生物从攻击中受到的伤害增加50%。]
  [1] 噬尸蛞蝓 26/26 Intent:[Debuff] Powers:[饥饿 4: 当有敌人死亡时，噬尸蛞蝓会立即吃下尸体，在本回合被击晕然后获得1点力量。]

Hand:
  [0] 防御(1) 获得5点格挡。 (unplayable)

Draw:6 Discard:4 Exhaust:0
Powers: 活力 2: 你的下一张攻击牌伤害增加。
Relics: 天赋君权: 在每场战斗开始时，获得辉星辉星辉星。 | 橙型香盒: 拾起时，升级一张牌。

Commands:
  [0] end_turn""",

    # Card selection overlay (reward after combat)
    """Card Selection (can skip) [Pick 1]:
  [0] 铁壁(2) 获得15点格挡。
  [1] 回旋斩(1) 对所有敌人造成8点伤害。
  [2] 重击(2) 造成12点伤害。

HP:66/75 Gold:120
Deck (11): 打击(1)x4 造成6点伤害。 | 防御(1)x4 获得5点格挡。 | 崇拜(1) 获得辉星辉星。 | 星星点点(1) 获得8点格挡。获得2点活力。 | 陨星(0) 造成7点伤害。给予1层虚弱。给予1层易伤。
Relics: 天赋君权: 在每场战斗开始时，获得辉星辉星辉星。 | 橙型香盒: 拾起时，升级一张牌。

Commands:
  [0] select_card card=铁壁
  [1] select_card card=回旋斩
  [2] select_card card=重击
  [3] skip""",
]


def build_system_prompt() -> str:
    return t("prompt.system")


def _stream_one(client, kwargs):
    """Stream a single API call, return (ttft, total, tps, usage, msg, rate_info)."""
    t0 = time.perf_counter()
    first_token_time = None

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if first_token_time is None and hasattr(event, "type") and event.type in (
                "content_block_start", "content_block_delta",
            ):
                first_token_time = time.perf_counter()

    t_end = time.perf_counter()
    msg = stream.get_final_message()
    response = stream.response
    usage = msg.usage
    ttft = first_token_time - t0 if first_token_time else float("nan")
    total = t_end - t0
    gen_time = t_end - first_token_time if first_token_time else 0
    out_toks = usage.output_tokens
    tps = out_toks / gen_time if gen_time > 0 else 0

    # Extract rate limit headers
    headers = getattr(response, "headers", {})
    rate_info = {
        "requests_remaining": headers.get("anthropic-ratelimit-requests-remaining"),
        "tokens_remaining": headers.get("anthropic-ratelimit-tokens-remaining"),
        "requests_reset": headers.get("anthropic-ratelimit-requests-reset"),
        "tokens_reset": headers.get("anthropic-ratelimit-tokens-reset"),
        "retry_after": headers.get("retry-after"),
    }

    return ttft, total, tps, usage, msg, rate_info


def _format_extras(msg, usage):
    """Format thinking and cache info strings."""
    think_info = ""
    for block in msg.content:
        if block.type == "thinking":
            think_info = f" | think={len(block.thinking)}ch"
            break
    cache_info = ""
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    if cache_create or cache_read:
        cache_info = f" | cache_w={cache_create} cache_r={cache_read}"
    return think_info, cache_info


def _format_rate_info(rate_info: dict) -> str:
    """Format rate limit info for display."""
    parts = []
    req_rem = rate_info.get("requests_remaining")
    tok_rem = rate_info.get("tokens_remaining")
    if req_rem is not None:
        parts.append(f"req_rem={req_rem}")
    if tok_rem is not None:
        parts.append(f"tok_rem={tok_rem}")
    return f" | {' '.join(parts)}" if parts else ""


def _apply_cache_control(system: str, tools: list[dict], messages: list[dict]) -> tuple:
    """Add cache_control breakpoints to system, tools, and last user message."""
    cached_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    cached_tools = list(tools)
    cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

    cached_messages = list(messages)
    # Find and mark last user message
    for i in range(len(cached_messages) - 1, -1, -1):
        msg = cached_messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            cached_messages[i] = {
                "role": "user",
                "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
            }
        elif isinstance(content, list) and content:
            content = list(content)
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block = {**last_block}
            last_block["cache_control"] = {"type": "ephemeral"}
            content[-1] = last_block
            cached_messages[i] = {"role": "user", "content": content}
        break

    return cached_system, cached_tools, cached_messages


def _print_summary(results, label="Summary"):
    avg = lambda key: sum(r[key] for r in results) / len(results)
    print(f"\n--- {label} ({len(results)} calls) ---")
    print(f"  Avg TTFT:       {avg('ttft'):.2f}s")
    print(f"  Avg throughput: {avg('tps'):.1f} tok/s")
    print(f"  Avg total:      {avg('total'):.2f}s")
    print(f"  Avg in tokens:  {avg('input_tokens'):.0f}")
    print(f"  Avg out tokens: {avg('output_tokens'):.0f}")


def bench_single(model: str, max_tokens: int, runs: int, thinking_budget: int):
    """Single-round benchmark: one briefing per call."""
    client = anthropic.Anthropic()
    system = build_system_prompt()
    tools = _build_tools()
    results = []

    kwargs: dict = {
        "model": model,
        "system": system,
        "tools": tools,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": BRIEFINGS[0]}],
    }
    if thinking_budget > 0:
        kwargs["max_tokens"] = thinking_budget + max_tokens
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["tool_choice"] = {"type": "auto"}
    else:
        kwargs["tool_choice"] = {"type": "any"}

    for i in range(runs):
        ttft, total, tps, usage, msg, rate_info = _stream_one(client, kwargs)
        think_info, cache_info = _format_extras(msg, usage)
        rate_str = _format_rate_info(rate_info)

        results.append({
            "ttft": ttft, "total": total, "tps": tps,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        })
        print(
            f"  Run {i+1}: TTFT={ttft:.2f}s | Total={total:.2f}s | "
            f"{tps:.1f} tok/s | in={usage.input_tokens} out={usage.output_tokens}"
            f"{think_info}{cache_info}{rate_str}"
        )

    _print_summary(results)


def bench_multi(model: str, max_tokens: int, turns: int, thinking_budget: int):
    """Multi-round benchmark: accumulates conversation history over N turns."""
    client = anthropic.Anthropic()
    system = build_system_prompt()
    tools = _build_tools()
    messages: list[dict] = []
    results = []

    base_kwargs: dict = {
        "model": model,
        "system": system,
        "tools": tools,
        "max_tokens": max_tokens,
    }
    if thinking_budget > 0:
        base_kwargs["max_tokens"] = thinking_budget + max_tokens
        base_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        base_kwargs["tool_choice"] = {"type": "auto"}
    else:
        base_kwargs["tool_choice"] = {"type": "any"}

    # First turn: plain user message
    messages.append({"role": "user", "content": BRIEFINGS[0]})

    for i in range(turns):
        kwargs = {**base_kwargs, "messages": messages}
        ttft, total, tps, usage, msg, rate_info = _stream_one(client, kwargs)
        think_info, cache_info = _format_extras(msg, usage)
        rate_str = _format_rate_info(rate_info)

        results.append({
            "ttft": ttft, "total": total, "tps": tps,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        })
        print(
            f"  Turn {i+1}: TTFT={ttft:.2f}s | Total={total:.2f}s | "
            f"{tps:.1f} tok/s | in={usage.input_tokens} out={usage.output_tokens}"
            f"{think_info}{cache_info}{rate_str}"
        )

        # Append assistant response, then deliver next briefing as tool_result
        messages.append({"role": "assistant", "content": msg.content})

        if i < turns - 1:
            next_briefing = BRIEFINGS[(i + 1) % len(BRIEFINGS)]
            tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]

            if not tool_uses:
                # No tool call (shouldn't happen with tool_choice: any, but handle it)
                messages.append({"role": "user", "content": next_briefing})
                continue

            # Match real agent: KB tool_results first, play_action last with briefing
            kb_results = []
            play_action_id = None
            for block in tool_uses:
                name = getattr(block, "name", "")
                if name == "update_knowledge_base":
                    kb_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": t("llm.kb_ok_set", store="in_run",
                                     key="bench", value="test"),
                    })
                elif name == "play_action":
                    play_action_id = block.id

            tool_results = list(kb_results)
            if play_action_id:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": play_action_id,
                    "content": next_briefing,
                })
            else:
                # No play_action — provide results for all tool_uses, then briefing
                for block in tool_uses:
                    if block.id not in {r["tool_use_id"] for r in tool_results}:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "OK",
                        })
                tool_results.append({"type": "text", "text": next_briefing})

            messages.append({"role": "user", "content": tool_results})

    _print_summary(results)

    # Show degradation: first half vs second half
    if turns >= 4:
        half = turns // 2
        first_half = results[:half]
        second_half = results[half:]
        avg_ttft_1 = sum(r["ttft"] for r in first_half) / len(first_half)
        avg_ttft_2 = sum(r["ttft"] for r in second_half) / len(second_half)
        avg_in_1 = sum(r["input_tokens"] for r in first_half) / len(first_half)
        avg_in_2 = sum(r["input_tokens"] for r in second_half) / len(second_half)
        print(f"\n--- Degradation ---")
        print(f"  First half  avg TTFT: {avg_ttft_1:.2f}s  avg input: {avg_in_1:.0f} tokens")
        print(f"  Second half avg TTFT: {avg_ttft_2:.2f}s  avg input: {avg_in_2:.0f} tokens")
        print(f"  TTFT increase: {(avg_ttft_2 / avg_ttft_1 - 1) * 100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Claude API with STS2 agent prompt")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", type=int, default=0,
                        help="Extended thinking budget in tokens (0=off)")
    parser.add_argument("--lang", default="zh", choices=["en", "zh"],
                        help="Language for prompts (default: zh)")

    sub = parser.add_subparsers(dest="mode", help="Benchmark mode")

    single_p = sub.add_parser("single", help="Single-round benchmark (default)")
    single_p.add_argument("--runs", type=int, default=5)

    multi_p = sub.add_parser("multi", help="Multi-round with history accumulation")
    multi_p.add_argument("--turns", type=int, default=20,
                         help="Number of conversation turns to simulate")

    args = parser.parse_args()
    set_lang(args.lang)
    mode = args.mode or "single"

    print(f"Model: {args.model}")
    print(f"Mode: {mode}")
    print(f"Language: {args.lang}")
    if args.thinking:
        print(f"Thinking budget: {args.thinking} tokens")
    print()

    if mode == "single":
        bench_single(args.model, args.max_tokens, getattr(args, "runs", 5), args.thinking)
    elif mode == "multi":
        bench_multi(args.model, args.max_tokens, getattr(args, "turns", 20), args.thinking)
