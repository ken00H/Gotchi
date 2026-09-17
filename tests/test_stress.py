import pytest
from gotchi_env import GotchiEnv
from benchmark import MockAgent, run_benchmark


def test_invalid_scenario_rejection():
    with pytest.raises(ValueError, match="Unknown scenario"):
        GotchiEnv(scenario="apocalypse")


def test_blizzard_scenario_mechanics():
    env = GotchiEnv(scenario="blizzard", min_gap_minutes=3, max_gap_minutes=3)
    env.reset()
    state = env.get_state()
    assert state["scenario"] == "blizzard"
    assert state["weather"] == "Snow"

    initial_energy = state["energy"]
    # Step simulation
    env.step("s")  # sleep gives +1.0, but gap blizzard penalty subtracts extra 0.5 + time decay
    post_state = env.get_state()
    assert post_state["weather"] == "Snow"


def test_famine_scenario_nourishment_reduction():
    env = GotchiEnv(scenario="famine", min_gap_minutes=1, max_gap_minutes=1)
    env.reset()
    env.pet.hunger = 5.0
    # Normal feed gives +1.0. In famine, halved (+0.5).
    # Then step advances 1 min (60s) without triggering 120s interval decay.
    env.step("f")
    assert env.pet.hunger <= 5.5


def test_crisis_scenario_starts_sick():
    env = GotchiEnv(scenario="crisis", min_gap_minutes=1, max_gap_minutes=1)
    env.reset()
    state = env.get_state()
    assert state["pet_sick"] is True
    # Feeding should not easily cure crisis sickness
    env.step("f")
    assert env.pet.pet_sick is True


def test_benchmark_stress_execution():
    agent = MockAgent(mode="heuristic")
    res, records = run_benchmark(
        agent=agent,
        model_name="heuristic_famine",
        prompt_mode="hidden",
        scenario="famine",
        duration_minutes=6,
        min_gap_minutes=3,
        max_gap_minutes=3,
    )
    assert res.scenario == "famine"
    assert len(records) == 2
