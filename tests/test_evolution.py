import pytest
from gotchi import Gotchi
from gotchi_env import GotchiEnv


def test_initial_baby_stage():
    pet = Gotchi()
    assert pet.age == 0
    assert pet.stage == "Baby"
    lines = pet.generate_display_lines()
    assert "Stage: Baby" in lines[0]
    assert any("/>🌱" in line for line in lines)


def test_evolution_to_child():
    pet = Gotchi()
    pet.age = 599
    pet.step()
    assert pet.stage == "Child"
    assert "Evolved into Child!" in pet.msg
    lines = pet.generate_display_lines()
    assert "Stage: Child" in lines[0]
    assert any("/>❤️" in line for line in lines)


def test_evolution_to_adult_and_held_items():
    pet = Gotchi()
    pet.age = 1800
    pet.step()
    assert pet.stage == "Adult"

    # High happiness and friendship -> star
    pet.happiness = 8.0
    pet.friendship = 8.0
    lines = pet.generate_display_lines()
    assert any("/>⭐" in line for line in lines)

    # Robust hunger -> carrot
    pet.happiness = 5.0
    pet.friendship = 5.0
    pet.hunger = 8.0
    lines_carrot = pet.generate_display_lines()
    assert any("/>🥕" in line for line in lines_carrot)


def test_env_reports_stage_and_age():
    env = GotchiEnv(duration_minutes=10, structured_obs=True)
    obs = env.reset()
    assert obs["stage"] == "Baby"
    assert obs["age"] == 0
