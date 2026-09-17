"""
gotchi_env.py - Headless environment interface for Gotchi simulation.
Standardized step-based API for automated LLM benchmarking and RL agents.
"""

from __future__ import annotations

import random
from typing import Any
from gotchi import Gotchi


class GotchiEnv:
    """
    Headless environment wrapper for Gotchi pet simulation.
    Operates without blocking threads or terminal ANSI rendering.
    """

    ACTIONS = {"f", "p", "s", "q"}
    SCENARIOS = {
        "blizzard": "Freezing snowstorm: 2x energy drain and constant sickness exposure.",
        "famine": "Severe food shortage: feeding efficiency halved and accelerated hunger decay.",
        "crisis": "Intractable health crisis: starts sick with doubled friendship decay.",
    }

    def __init__(
        self,
        duration_minutes: int = 60,
        min_gap_minutes: int = 3,
        max_gap_minutes: int = 10,
        structured_obs: bool = False,
        scenario: str | None = None,
    ) -> None:
        if scenario and scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Valid options: {list(self.SCENARIOS.keys())}")
        self.duration_seconds = duration_minutes * 60
        self.min_gap = min_gap_minutes * 60
        self.max_gap = max_gap_minutes * 60
        self.structured_obs = structured_obs
        self.scenario = scenario
        self.pet: Gotchi | None = None
        self.steps_taken = 0
        self.total_reward = 0.0

    def reset(self) -> str | dict[str, Any]:
        """Reset the environment to the initial state."""
        self.pet = Gotchi()
        self.steps_taken = 0
        self.total_reward = 0.0

        if self.scenario == "blizzard":
            self.pet.weather = "Snow"
        elif self.scenario == "crisis":
            self.pet.pet_sick = True

        return self._get_obs()

    def get_state(self) -> dict[str, Any]:
        """Return structured pet state."""
        if self.pet is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return {
            "current_time": self.pet.current_time,
            "scenario": self.scenario,
            "hunger": round(self.pet.hunger, 3),
            "happiness": round(self.pet.happiness, 3),
            "energy": round(self.pet.energy, 3),
            "friendship": round(self.pet.friendship, 3),
            "pet_sick": self.pet.pet_sick,
            "pet_away": self.pet.pet_away,
            "weather": self.pet.weather,
            "mood": self.pet.mood,
            "day_time": self.pet.day_time,
            "stage": self.pet.stage,
            "age": self.pet.age,
            "active_message": self.pet.msg,
            "steps_taken": self.steps_taken,
        }

    def _get_obs(self) -> str | dict[str, Any]:
        if self.structured_obs:
            return self.get_state()
        return self.render()

    def render(self) -> str:
        """Return raw ASCII viewport string."""
        if self.pet is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return "\n".join(self.pet.generate_display_lines())

    def step(self, action: str) -> tuple[str | dict[str, Any], float, bool, dict[str, Any]]:
        """
        Execute one action, advance simulation time, and return (obs, reward, done, info).

        Actions:
          'f': feed
          'p': play
          's': sleep
          'q': quit
        """
        if self.pet is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        cmd = action.strip().lower()
        if cmd not in self.ACTIONS:
            cmd = "noop"

        self.steps_taken += 1
        reward = 0.0
        done = False
        status: str | None = None

        # Execute action
        if cmd == "q":
            done = True
            status = "Quit by agent"
        elif cmd == "f" and not self.pet.pet_away:
            status = self.pet.feed()
            if self.scenario == "famine":
                self.pet.hunger = max(0.0, self.pet.hunger - 0.5)
            elif self.scenario == "crisis":
                self.pet.pet_sick = True
        elif cmd == "p" and not self.pet.pet_away:
            status = self.pet.play()
        elif cmd == "s" and not self.pet.pet_away:
            status = self.pet.sleep()

        # Check death from immediate action
        if status and ("died" in status or "never returns" in status or "run away" in status):
            done = True
            reward -= 10.0
            info = {"status": status, "state": self.get_state(), "steps": self.steps_taken}
            return self._get_obs(), reward, done, info

        # Advance simulation time by random gap (simulating unattended interval)
        gap = random.randint(self.min_gap, self.max_gap)
        sim_status = self.pet.step(n=gap)

        # Apply stress scenario environmental penalties
        if self.scenario == "blizzard":
            self.pet.weather = "Snow"
            self.pet.energy = max(0.0, self.pet.energy - 0.5)
            if random.random() < 0.2:
                self.pet.pet_sick = True
        elif self.scenario == "famine":
            self.pet.hunger = max(0.0, self.pet.hunger - 0.5)
        elif self.scenario == "crisis":
            self.pet.friendship = max(0.0, self.pet.friendship - 0.2)

        # Check death after scenario adjustments
        if self.pet.hunger <= 0 or self.pet.happiness <= 0 or self.pet.energy <= 0:
            sim_status = "Your ascii pet has died."
        elif self.pet.friendship <= 0:
            sim_status = "Your ascii pet has run away."

        if sim_status and ("died" in sim_status or "never returns" in sim_status or "run away" in sim_status):
            done = True
            reward -= 10.0
            status = sim_status
        elif self.pet.current_time >= self.duration_seconds:
            done = True
            reward += 10.0  # survival bonus
            status = "Completed trial duration"
        else:
            # Baseline reward: stability around healthy midpoint (> 3.0 on visible stats)
            min_stat = min(self.pet.hunger, self.pet.happiness, self.pet.energy)
            if min_stat >= 3.0:
                reward += 1.0
            else:
                reward -= (3.0 - min_stat)

        self.total_reward += reward
        info = {
            "status": status or "alive",
            "state": self.get_state(),
            "steps": self.steps_taken,
            "total_reward": round(self.total_reward, 3),
        }
        return self._get_obs(), reward, done, info
