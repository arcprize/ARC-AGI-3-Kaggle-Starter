from agent.hypotheses import ActionKey, HypothesisEngine, OutcomePattern
from agent.state import ObjectMotion, StateDiff


def test_hypotheses_strengthen_and_contradictions_falsify() -> None:
    engine = HypothesisEngine()
    action = ActionKey(2)
    seeded = engine.applicable(action)
    no_effect = next(h for h in seeded if h.prediction.changed is False)
    progress = next(h for h in seeded if h.prediction.progress == 1)

    engine.update(action, StateDiff(), 1)
    assert no_effect.support == [1]
    assert progress.contradictions == [1]
    update = engine.update(action, StateDiff(), 2)
    assert progress.hypothesis_id in update.falsified
    assert progress.rejected


def test_unexplained_transition_generates_explicit_hypothesis() -> None:
    engine = HypothesisEngine()
    action = ActionKey(5)
    diff = StateDiff(progress_delta=2, terminal="WIN")
    update = engine.update(action, diff, 3)
    assert update.generated
    generated = next(h for h in engine.hypotheses if h.hypothesis_id == update.generated[0])
    assert generated.prediction.progress == 2
    assert generated.prediction.terminal == "WIN"


def test_generic_change_does_not_suppress_specific_motion_rule() -> None:
    engine = HypothesisEngine()
    action = ActionKey(4)
    motion = ObjectMotion("old", "new", 2, (1, 0), (1.0, 1.0), (2.0, 1.0))
    update = engine.update(action, StateDiff(motions=(motion,)), 1)
    assert update.generated
    generated = next(h for h in engine.hypotheses if h.hypothesis_id == update.generated[0])
    assert generated.prediction.motion == (1, 0)


def test_prediction_disagreement_is_nonzero_for_competing_rules() -> None:
    engine = HypothesisEngine()
    assert 0.5 < engine.disagreement(ActionKey(1)) <= 1.0


def test_shared_motion_of_multiple_objects_forms_directional_prediction() -> None:
    motion = ObjectMotion("old", "new", 2, (1, 0), (1.0, 1.0), (2.0, 1.0))
    pattern = OutcomePattern.from_diff(StateDiff(motions=(motion, motion)))
    assert pattern.motion == (1, 0)


def test_coordinate_actions_share_general_priors() -> None:
    engine = HypothesisEngine()
    first = ActionKey(6, target_color=2, region=(0, 0))
    second = ActionKey(6, target_color=3, region=(2, 2))
    engine.applicable(first)
    assert len(engine.applicable(second)) == 4
    assert all(h.action == ActionKey(6) for h in engine.applicable(second))
