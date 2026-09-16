from types import SimpleNamespace

from agent.perception import diff_states, extract_objects, latest_grid, perceive


def test_latest_grid_uses_final_animation_frame() -> None:
    assert latest_grid([[[0, 0]], [[1, 1]]]) == ((1, 1),)


def test_extracts_connected_objects_and_shape() -> None:
    grid = (
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 3),
        (0, 2, 0, 0, 3),
        (0, 0, 0, 0, 0),
    )
    objects = extract_objects(grid)
    assert [(obj.color, obj.size) for obj in objects] == [(2, 3), (3, 2)]
    assert objects[0].bbox == (1, 1, 2, 2)
    assert objects[0].shape_signature == ((0, 0), (0, 1), (1, 0))


def test_frame_diff_detects_motion_and_changed_region() -> None:
    before = perceive([[0, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0]])
    after = perceive([[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]])
    diff = diff_states(before, after)
    assert diff.changed_cells == frozenset({(1, 1), (2, 1)})
    assert diff.changed_bbox == (1, 1, 2, 1)
    assert len(diff.motions) == 1
    assert diff.motions[0].delta == (1, 0)


def test_frame_diff_detects_appearance_and_disappearance() -> None:
    before = perceive([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    after = perceive([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    diff = diff_states(before, after)
    # Same location is matched as a transformation instead of double-counting.
    assert len(diff.transformed) == 1
    assert diff.changed


def test_border_counter_changes_do_not_fragment_abstract_state() -> None:
    first = perceive([[2, 0, 0], [0, 3, 0], [0, 0, 0]])
    second = perceive([[2, 2, 0], [0, 3, 0], [0, 0, 0]])
    assert first.raw_fingerprint != second.raw_fingerprint
    assert first.fingerprint == second.fingerprint


def test_interior_motion_and_level_progress_change_abstract_state() -> None:
    first = perceive(
        SimpleNamespace(
            frame=[[[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0]]],
            levels_completed=0,
        )
    )
    moved = perceive(
        SimpleNamespace(
            frame=[[[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0]]],
            levels_completed=0,
        )
    )
    progressed = perceive(
        SimpleNamespace(
            frame=[[[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0]]],
            levels_completed=1,
        )
    )
    assert first.fingerprint != moved.fingerprint
    assert first.fingerprint != progressed.fingerprint
