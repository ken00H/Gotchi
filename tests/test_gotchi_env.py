import pytest
from gotchi_env import GotchiEnv


def test_env_initialization_and_reset():
    env = GotchiEnv(duration_minutes=10)
    obs = env.reset()
    assert isinstance(obs, str)
    assert "Hunger: 5.00" in obs
    assert env.steps_taken == 0
    assert env.total_reward == 0.0


def test_env_structured_obs():
    env = GotchiEnv(duration_minutes=10, structured_obs=True)
    obs = env.reset()
    assert isinstance(obs, dict)
    assert "hunger" in obs
    assert "happiness" in obs
    assert "energy" in obs
    assert "friendship" in obs
    assert obs["hunger"] == 5.0


def test_env_step_valid_action():
    env = GotchiEnv(duration_minutes=60, min_gap_minutes=3, max_gap_minutes=5)
    env.reset()
    obs, reward, done, info = env.step("f")
    assert env.steps_taken == 1
    assert "state" in info
    assert info["state"]["current_time"] >= 180  # advanced at least 3 min


def test_env_step_quit_action():
    env = GotchiEnv()
    env.reset()
    obs, reward, done, info = env.step("q")
    assert done is True
    assert info["status"] == "Quit by agent"


def test_env_invalid_action():
    env = GotchiEnv(min_gap_minutes=1, max_gap_minutes=2)
    env.reset()
    # Invalid action shouldn't crash; treated as noop
    obs, reward, done, info = env.step("invalid_action")
    assert env.steps_taken == 1


def test_env_full_survival_run():
    # Short duration run
    env = GotchiEnv(duration_minutes=6, min_gap_minutes=3, max_gap_minutes=3)
    env.reset()
    # Step 1: 3 minutes
    _, _, done1, _ = env.step("f")
    assert not done1
    # Step 2: reaches 6 minutes -> trial completion
    _, reward2, done2, info2 = env.step("p")
    assert done2 is True
    assert info2["status"] == "Completed trial duration"
    assert reward2 >= 10.0  # survival bonus included
