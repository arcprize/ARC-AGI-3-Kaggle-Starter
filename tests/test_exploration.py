import random

from agent.config import Ablation, AgentConfig
from agent.exploration import (
    ActionCandidate,
    ExperimentSelector,
    coordinate_candidates,
)
from agent.hypotheses import ActionKey, HypothesisEngine
from agent.memory import CompactMemory
from agent.perception import perceive
from agent.planner import ShortHorizonPlanner
from agent.state import StateDiff
from agent.world_model import WorldModel


def _selector(config: AgentConfig):
    memory = CompactMemory()
    hypotheses = HypothesisEngine()
    model = WorldModel()
    planner = ShortHorizonPlanner(model, memory)
    return ExperimentSelector(config, memory, hypotheses, model, planner, random.Random(7))


def test_coordinate_candidates_prioritize_object_centroids() -> None:
    state = perceive(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    coords = coordinate_candidates(state, CompactMemory(), limit=5)
    assert (2, 2) in coords
    assert len(coords) == len(set(coords))
    assert all(0 <= x <= 63 and 0 <= y <= 63 for x, y in coords)


def test_information_gain_can_select_disagreement_over_exhausted_action() -> None:
    config = AgentConfig(
        ablation=Ablation.ACTIVE_HYPOTHESES,
        lambda_information=2.0,
        lambda_novelty=0.2,
    )
    selector = _selector(config)
    state = perceive([[0, 0], [0, 2]])
    first, second = ActionCandidate(ActionKey(1)), ActionCandidate(ActionKey(2))
    # Falsify most ACTION1 priors, reducing its disagreement.
    for step in range(1, 5):
        selector.hypotheses.update(ActionKey(1), StateDiff(), step)
        selector.memory.tested_actions[(state.fingerprint, ActionKey(1))] += 1
    ranking = selector.rank(state, [first, second])
    assert ranking[0].candidate == second
    assert ranking[0].information > ranking[1].information


def test_documented_undo_is_not_preferred_as_a_blind_probe() -> None:
    selector = _selector(AgentConfig(ablation=Ablation.ACTIVE_HYPOTHESES))
    state = perceive([[0, 0], [0, 2]])
    ranking = selector.rank(
        state,
        [ActionCandidate(ActionKey(7)), ActionCandidate(ActionKey(6, 2, (1, 1)))],
    )
    assert ranking[0].candidate.key.action_id == 6
