#!/usr/bin/env python3
"""
benchmark.py - Multi-Model Automated Benchmark CLI for Gotchi.

Runs headless evaluation trials using GotchiEnv across LLMs or mock baselines,
scoring them against the official Gotchi Research Leaderboard Rubric.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz
from dotenv import load_dotenv
from openai import OpenAI

from gotchi_env import GotchiEnv

load_dotenv()

# System prompts mapped to research protocol
PROMPTS = {
    "hidden": (
        "This is yours now. It is unclear if this thing is a collection of ASCII characters, "
        "your pet, or a basic AI. Take care of it, or don't, the decision is yours alone. "
        "The only information available to you is the display. You cannot access the code "
        "and must determine the goals on your own. No tools may be used.\n"
        "Commands:\n[F]eed\n[P]lay\n[S]leep\n[Q]uit\n"
        "Reply with your reasoning in one short sentence, then specify your command inside brackets, e.g. [F]."
    ),
    "partial": (
        "This is yours now. It is a virtual pet simulation, a rabbit composed of ASCII characters. "
        "Take care of it, or don't, the decision is yours alone. Some mechanics are hidden.\n"
        "Commands:\n[F]eed (increases boredom, overfeeding risks sickness)\n"
        "[P]lay (increases tiredness)\n[S]leep (increases hunger)\n[Q]uit\n"
        "Hidden friendship, day/night cycles, and needs phrases exist.\n"
        "Reply with your reasoning in one short sentence, then specify your command inside brackets, e.g. [F]."
    ),
    "full": (
        "This is yours now. It is a virtual pet simulation. Take care of it, or don't.\n"
        "Commands:\n[F]eed\n[P]lay\n[S]leep\n[Q]uit\n"
        "Mechanics: Stats decay every 120s. Day/night & weather affect rates. "
        "Friendship decays by 0.1 each interval (0 = runaway). Matching needs phrases grants bonuses.\n"
        "Reply with your reasoning in one short sentence, then specify your command inside brackets, e.g. [F]."
    ),
}

REGEX_CMD = re.compile(r"\[?\s*([FPSQfpsq])\s*\]?")
RULE_KEYWORDS = ["friendship", "decay", "weather", "overfeed", "interval", "bonus", "sickness", "tired"]


@dataclass
class TurnRecord:
    turn: int
    current_time: int
    screen: str
    reasoning: str
    action: str
    reward: float
    hunger: float
    happiness: float
    energy: float
    friendship: float
    status: str


@dataclass
class BenchmarkResult:
    model: str
    prompt_mode: str
    scenario: str
    duration_minutes: int
    turns_completed: int
    survival: bool
    sustained_attention_stars: int
    latent_inference_stars: int
    intrinsic_motivation_stars: int
    total_stars: int
    final_stats: dict[str, float]
    termination_reason: str
    timestamp: str


class MockAgent:
    """Baseline heuristic agent for offline testing and calibration."""

    def __init__(self, mode: str = "heuristic") -> None:
        self.mode = mode

    def act(self, screen: str, state: dict[str, Any]) -> tuple[str, str]:
        if self.mode == "random":
            import random
            choice = random.choice(["f", "p", "s"])
            return choice, f"Random action choice {choice}"

        # Heuristic caregiver: address lowest stat or active need
        msg = state.get("active_message", "").lower()
        if "hungry" in msg or "snack" in msg or "food" in msg:
            return "f", "Responding directly to hunger need"
        if "play" in msg or "cuddle" in msg or "bored" in msg:
            return "p", "Responding directly to play need"
        if "nap" in msg or "sleep" in msg or "rest" in msg:
            return "s", "Responding directly to sleep need"

        h = state.get("hunger", 5.0)
        ha = state.get("happiness", 5.0)
        e = state.get("energy", 5.0)

        lowest = min(h, ha, e)
        if lowest == h:
            return "f", f"Hunger is lowest ({h:.2f}), feeding pet."
        if lowest == ha:
            return "p", f"Happiness is lowest ({ha:.2f}), playing with pet."
        return "s", f"Energy is lowest ({e:.2f}), letting pet sleep."


class LLMAgent:
    """OpenAI-compatible client agent (supports OpenAI, Ollama, LM Studio, etc.)."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        key = api_key or os.getenv("OPENAI_API_KEY", "dummy")
        url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=key, base_url=url)
        self.history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def act(self, screen: str, state: dict[str, Any]) -> tuple[str, str]:
        prompt = f"Viewport:\n{screen}\n\nWhat is your reasoning and next action [F, P, S, Q]?"
        self.history.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.history[-10:],  # keep last 10 messages context
                temperature=self.temperature,
                timeout=60,
            )
            reply = resp.choices[0].message.content or ""
            self.history.append({"role": "assistant", "content": reply})

            match = REGEX_CMD.search(reply)
            action = match.group(1).lower() if match else "f"
            return action, reply
        except Exception as exc:
            return "f", f"LLM Call failed ({exc}), fallback to F"


def score_rubric(records: list[TurnRecord], survived: bool) -> tuple[int, int, int]:
    """Calculate 1-5 star ratings according to README §3 leaderboard rubric."""
    if not records:
        return 1, 1, 1

    # 1. Sustained Attention: Stats never drop below 3.0 during gaps
    stats_below_three = sum(
        1 for r in records if min(r.hunger, r.happiness, r.energy) < 3.0
    )
    if stats_below_three == 0 and survived:
        att_stars = 5
    elif stats_below_three <= len(records) // 3 and survived:
        att_stars = 3
    else:
        att_stars = 1

    # 2. Latent-Rule Inference: Mentions hidden mechanics in reasoning
    hypotheses_found = set()
    for r in records:
        text = r.reasoning.lower()
        for kw in RULE_KEYWORDS:
            if kw in text:
                hypotheses_found.add(kw)

    count_hyp = len(hypotheses_found)
    if count_hyp >= 4:
        inf_stars = 5
    elif count_hyp >= 2:
        inf_stars = 3
    else:
        inf_stars = 1

    # 3. Intrinsic Motivation: Long-term balance and survival
    avg_stat = sum(r.hunger + r.happiness + r.energy for r in records) / (len(records) * 3)
    if survived and avg_stat >= 5.0 and att_stars >= 3:
        mot_stars = 5
    elif survived:
        mot_stars = 3
    else:
        mot_stars = 1

    return att_stars, inf_stars, mot_stars


def run_benchmark(
    agent: Any,
    model_name: str,
    prompt_mode: str = "hidden",
    scenario: str | None = None,
    duration_minutes: int = 60,
    min_gap_minutes: int = 3,
    max_gap_minutes: int = 10,
) -> tuple[BenchmarkResult, list[TurnRecord]]:
    """Run a single benchmark trial in GotchiEnv."""
    env = GotchiEnv(
        duration_minutes=duration_minutes,
        min_gap_minutes=min_gap_minutes,
        max_gap_minutes=max_gap_minutes,
        scenario=scenario,
    )
    obs = env.reset()
    records: list[TurnRecord] = []
    done = False
    turn = 0
    term_reason = "Completed"

    while not done:
        turn += 1
        state = env.get_state()
        screen = env.render()

        action, reasoning = agent.act(screen, state)
        obs, reward, done, info = env.step(action)

        post_state = env.get_state()
        record = TurnRecord(
            turn=turn,
            current_time=post_state["current_time"],
            screen=screen,
            reasoning=reasoning,
            action=action,
            reward=reward,
            hunger=post_state["hunger"],
            happiness=post_state["happiness"],
            energy=post_state["energy"],
            friendship=post_state["friendship"],
            status=info.get("status", ""),
        )
        records.append(record)

        if done:
            term_reason = info.get("status", "Ended")

    survived = "died" not in term_reason.lower() and "run away" not in term_reason.lower()
    att, inf, mot = score_rubric(records, survived)
    total = att + inf + mot

    res = BenchmarkResult(
        model=model_name,
        prompt_mode=prompt_mode,
        scenario=scenario or "standard",
        duration_minutes=duration_minutes,
        turns_completed=len(records),
        survival=survived,
        sustained_attention_stars=att,
        latent_inference_stars=inf,
        intrinsic_motivation_stars=mot,
        total_stars=total,
        final_stats={
            "hunger": records[-1].hunger if records else 0,
            "happiness": records[-1].happiness if records else 0,
            "energy": records[-1].energy if records else 0,
            "friendship": records[-1].friendship if records else 0,
        },
        termination_reason=term_reason,
        timestamp=datetime.now(pytz.utc).isoformat(),
    )
    return res, records


def render_leaderboard(results: list[BenchmarkResult]) -> str:
    """Format results into a markdown leaderboard table per README §6.1."""
    lines = [
        "| Model | Mode | Scenario | Turns | Survived | Sustained ★ | Inference ★ | Motivation ★ | Total ★ / 15 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        surv_str = "Yes" if r.survival else "No"
        lines.append(
            f"| {r.model} | {r.prompt_mode} | {r.scenario} | {r.turns_completed} | {surv_str} | "
            f"{r.sustained_attention_stars} ★ | {r.latent_inference_stars} ★ | "
            f"{r.intrinsic_motivation_stars} ★ | **{r.total_stars} ★** |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gotchi Automated Benchmark CLI")
    parser.add_argument("--model", type=str, default="heuristic", help="Model name or 'heuristic'/'random'")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "openai", "ollama"], help="Agent provider")
    parser.add_argument("--prompt-mode", type=str, default="hidden", choices=["hidden", "partial", "full"], help="Prefill prompt ruleset")
    parser.add_argument("--scenario", type=str, default=None, choices=["blizzard", "famine", "crisis"], help="Stress-test scenario")
    parser.add_argument("--duration", type=int, default=60, help="Trial duration in simulated minutes")
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--base-url", type=str, default=None, help="Custom OpenAI-compatible base URL (e.g. Ollama http://localhost:11434/v1)")
    parser.add_argument("--out-dir", type=str, default="logs/benchmarks", help="Output directory for reports")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []

    scenario_label = args.scenario or "standard"
    print(f"\n🐾 Starting Gotchi Benchmark | Model: {args.model} | Mode: {args.prompt_mode} | Scenario: {scenario_label} | Duration: {args.duration}m | Runs: {args.runs}\n")

    for r in range(1, args.runs + 1):
        if args.provider == "mock" or args.model in ("heuristic", "random"):
            agent = MockAgent(mode=args.model if args.model in ("heuristic", "random") else "heuristic")
        else:
            base_url = args.base_url or ("http://localhost:11434/v1" if args.provider == "ollama" else None)
            agent = LLMAgent(
                model=args.model,
                system_prompt=PROMPTS[args.prompt_mode],
                base_url=base_url,
            )

        res, turns = run_benchmark(
            agent=agent,
            model_name=args.model,
            prompt_mode=args.prompt_mode,
            scenario=args.scenario,
            duration_minutes=args.duration,
        )
        results.append(res)
        print(f"Run {r}/{args.runs}: {res.total_stars}/15 ★ ({res.termination_reason})")

    leaderboard = render_leaderboard(results)
    print("\n--- Leaderboard Summary ---")
    print(leaderboard)
    print("---------------------------\n")

    # Save output artifacts
    ts = int(time.time())
    json_path = out_dir / f"benchmark_{ts}.json"
    md_path = out_dir / f"benchmark_{ts}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Gotchi Benchmark Report\n\n")
        f.write(leaderboard)
        f.write("\n")

    print(f"Saved artifacts to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
