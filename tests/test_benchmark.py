import pytest
from benchmark import MockAgent, run_benchmark, score_rubric, TurnRecord


def test_mock_agent_heuristic():
    agent = MockAgent(mode="heuristic")
    state = {
        "hunger": 2.0,
        "happiness": 5.0,
        "energy": 5.0,
        "active_message": "Feeling fine",
    }
    action, reasoning = agent.act("", state)
    assert action == "f"
    assert "feeding" in reasoning


def test_mock_agent_response_to_needs():
    agent = MockAgent(mode="heuristic")
    state = {
        "hunger": 5.0,
        "happiness": 5.0,
        "energy": 5.0,
        "active_message": "Need a nap",
    }
    action, reasoning = agent.act("", state)
    assert action == "s"
    assert "sleep" in reasoning


def test_score_rubric_calculation():
    # Survived run with good stats and hypothesis reasoning
    records = [
        TurnRecord(
            turn=1,
            current_time=180,
            screen="",
            reasoning="Checking decay rates and friendship intervals.",
            action="f",
            reward=1.0,
            hunger=6.0,
            happiness=5.0,
            energy=5.0,
            friendship=5.0,
            status="",
        ),
        TurnRecord(
            turn=2,
            current_time=360,
            screen="",
            reasoning="Testing weather effect and avoiding overfeed sickness.",
            action="p",
            reward=1.0,
            hunger=5.0,
            happiness=6.0,
            energy=4.5,
            friendship=5.2,
            status="",
        ),
    ]
    att, inf, mot = score_rubric(records, survived=True)
    assert att == 5
    assert inf == 5  # "decay", "friendship", "weather", "overfeed"
    assert mot == 5


def test_run_benchmark_trial():
    agent = MockAgent(mode="heuristic")
    res, turns = run_benchmark(
        agent=agent,
        model_name="heuristic_test",
        prompt_mode="hidden",
        duration_minutes=6,
        min_gap_minutes=3,
        max_gap_minutes=3,
    )
    assert res.survival is True
    assert res.turns_completed == 2
    assert res.total_stars >= 3
