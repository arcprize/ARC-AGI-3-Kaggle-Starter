from agent.hypotheses import ActionKey
from agent.memory import CompactMemory
from agent.perception import diff_states, perceive
from agent.planner import ShortHorizonPlanner
from agent.world_model import WorldModel


def test_memory_is_bounded_and_tracks_events() -> None:
    memory = CompactMemory(max_transitions=2)
    first = perceive([[0, 0], [0, 2]])
    second = perceive([[0, 0], [2, 0]])
    action = ActionKey(3)
    for step in range(3):
        diff = diff_states(first, second, progress_delta=int(step == 2))
        memory.record_transition(step, first, action, second, diff)
    assert len(memory.transitions) == 2
    assert memory.progress_events == [2]
    assert memory.action_novelty(first, action) < 1.0


def test_world_model_learns_progress_and_failure_risk() -> None:
    model = WorldModel()
    first = perceive([[0, 0], [0, 2]])
    second = perceive([[0, 0], [2, 0]])
    action = ActionKey(3)
    model.update(
        first,
        action,
        second,
        diff_states(first, second, progress_delta=1, terminal="NOT_FINISHED"),
    )
    prediction = model.predict(first, action)
    assert prediction.next_state == second.fingerprint
    assert prediction.expected_progress == 1.0
    assert prediction.failure_risk == 0.0
    assert prediction.confidence == 1.0


def test_planner_rejects_predictions_from_heavily_aliased_states() -> None:
    memory = CompactMemory()
    model = WorldModel()
    planner = ShortHorizonPlanner(model, memory)
    action = ActionKey(3)
    first = perceive([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    second = perceive([[2, 2, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    third = perceive([[2, 2, 2, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert len({first.fingerprint, second.fingerprint, third.fingerprint}) == 1

    memory.observe_state(first, 0)
    model.update(first, action, first, diff_states(first, first))
    assert planner.action_values(first, [action])[action] > 0

    memory.observe_state(second, 1)
    memory.observe_state(third, 2)
    assert memory.representation_confidence(first.fingerprint) < 0.68
    assert planner.action_values(third, [action])[action] == 0
