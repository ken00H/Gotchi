import pytest
from gotchi import Gotchi
from auto_gotchi import parse_command, capture_screen, is_pet_dead


def test_initial_state():
    pet = Gotchi()
    assert pet.hunger == 5.0
    assert pet.happiness == 5.0
    assert pet.energy == 5.0
    assert pet.friendship == 5.0
    assert not pet.pet_sick
    assert not pet.pet_away


def test_feed_action():
    pet = Gotchi()
    initial_energy = pet.energy
    initial_friendship = pet.friendship
    pet.feed()
    assert pet.hunger == 6.0
    assert pet.energy == initial_energy - 0.25
    assert pet.friendship == initial_friendship + 0.2


def test_play_action():
    pet = Gotchi()
    initial_energy = pet.energy
    pet.play()
    assert pet.happiness == 6.0
    assert pet.energy == initial_energy - 0.25


def test_play_while_sick():
    pet = Gotchi()
    pet.pet_sick = True
    pet.play()
    # Happiness should not increase when sick
    assert pet.happiness == 5.0
    assert pet.energy == 4.75


def test_sleep_action():
    pet = Gotchi()
    initial_hunger = pet.hunger
    pet.sleep()
    assert pet.energy == 6.0
    assert pet.hunger == initial_hunger - 0.25


def test_sickness_trigger_at_max_hunger():
    pet = Gotchi()
    pet.hunger = 10.0
    pet.energy = 5.0
    # Step simulation multiple times to trigger the 10% sickness roll
    for _ in range(100):
        pet.step()
        if pet.pet_sick:
            break
    assert pet.pet_sick is True


def test_death_when_hunger_zero():
    pet = Gotchi()
    pet.hunger = 0.0
    res = pet.step()
    assert res == "Your ascii pet has died."


def test_death_when_happiness_zero():
    pet = Gotchi()
    pet.happiness = 0.0
    res = pet.step()
    assert res == "Your ascii pet has died."


def test_death_when_energy_zero():
    pet = Gotchi()
    pet.energy = 0.0
    res = pet.step()
    assert res == "Your ascii pet has died."


def test_runaway_when_friendship_zero():
    pet = Gotchi()
    pet.friendship = 0.05
    pet.current_time = 0
    pet.last_needs_update = 0
    pet.needs_interval = 1
    pet.step()
    # friendship decayed to 0
    assert pet.friendship == 0.0


def test_needs_matching_bonus():
    pet = Gotchi()
    pet.happiness = 5.0
    pet.active_phrase_data = ("Feeling playful!", "happiness", -2.0)
    pet.play()
    # 5.0 + 1.0 (play) - (-2.0) (bonus) = 8.0
    assert pet.happiness == 8.0
    assert pet.msg == "Thank you for playing with me!"
    assert pet.active_phrase_data is None


def test_wandering_away_and_return():
    pet = Gotchi()
    pet.current_time = 0
    pet.last_input_time = 0
    # Advance past 300 steps of inactivity
    pet.step(301)
    assert pet.pet_away is True

    # Advance another 300 steps away -> returns
    pet.step(300)
    assert pet.pet_away is False


def test_auto_gotchi_command_parser():
    assert parse_command("[F]") == "f"
    assert parse_command("  p  ") == "p"
    assert parse_command("[S]") == "s"
    assert parse_command("Q") == "q"
    assert parse_command("invalid text") is None


def test_auto_gotchi_death_detector():
    pet = Gotchi()
    screen_alive = capture_screen(pet)
    assert not is_pet_dead(pet, screen_alive)

    screen_dead = "Your ascii pet has died."
    assert is_pet_dead(pet, screen_dead)
