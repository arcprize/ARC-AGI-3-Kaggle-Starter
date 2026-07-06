"""
Debug script: Run AR25 level 2 through the ORCH-OR analyzer.
Constructs the board grid from the level definition sprite data.
"""
from pprint import pprint

from agi_core.orch_or.analyzer import build_game_model
from agi_core.orch_or.reducer import reduce, select_leaf
from agi_core.orch_or.plan_generator import generate_plan

# AR25 constants
BACKGROUND_COLOR = 9
PADDING_COLOR = 5  # letterbox
GRID_SIZE = 21

# Sprite pixel data
# 0001sruqbuvukh: single pixel, color 11 (nfcdkcnytd)
# 0026ptolirjxrb: clickable obstacle, color 5 (avprkivezt/bzphplunby)
# Shape: 4 rows x 5 cols
#  [5, 5, 5, -, -]   row 0
#  [-, -, 5, -, -]   row 1
#  [-, -, 5, -, -]   row 2
#  [-, -, 5, 5, 5]   row 3
# 0054kgxrvfihgm: mirror axis, 1x41 pixels color 10 (zdyigzwohq/htgujuotrd) at (12, -3)

# Level 2 sprite placements
# 0001sruqbuvukh (single pixels, color 11) at these positions:
SINGLE_PIXELS = [
    (5, 14), (1, 17), (4, 14), (3, 14),
    (3, 15), (3, 16), (3, 17), (2, 17),
]

OBSTACLE_X, OBSTACLE_Y = 15, 6  # 0026ptolirjxrb position
MIRROR_X = 12  # 0054kgxrvfihgm is at (12, -3), vertical mirror at x=12

# 0026ptolirjxrb pixel shape (4 rows, 5 cols, -1 = transparent)
OBSTACLE_PIXELS = [
    [5, 5, 5, -1, -1],
    [-1, -1, 5, -1, -1],
    [-1, -1, 5, -1, -1],
    [-1, -1, 5, 5, 5],
]


def construct_board() -> list[list[int]]:
    board = [[BACKGROUND_COLOR for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    # Place the reflected obstacle copy (mirrored across x=12, color 4 = pbtdgroplk)
    for dy, row in enumerate(OBSTACLE_PIXELS):
        for dx, color in enumerate(row):
            if color != -1:
                src_x = OBSTACLE_X + dx
                src_y = OBSTACLE_Y + dy
                refl_x = 2 * MIRROR_X - src_x
                refl_y = src_y
                if 0 <= refl_x < GRID_SIZE and 0 <= refl_y < GRID_SIZE:
                    board[refl_y][refl_x] = 4  # REFLECTION color

    # Place the original obstacle (color 5)
    for dy, row in enumerate(OBSTACLE_PIXELS):
        for dx, color in enumerate(row):
            if color != -1:
                x = OBSTACLE_X + dx
                y = OBSTACLE_Y + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    board[y][x] = color  # 5 = clickable obstacle

    # Place single-pixel markers (color 11)
    for x, y in SINGLE_PIXELS:
        board[y][x] = 11

    return board


def print_board(board: list[list[int]]) -> None:
    print(f"Board {len(board)}x{len(board[0])}:")
    for y, row in enumerate(board):
        line = ""
        for x, val in enumerate(row):
            if val == BACKGROUND_COLOR:
                line += "."
            elif val == 4:
                line += "R"
            elif val == 5:
                line += "O"
            elif val == 11:
                line += "P"
            else:
                line += str(val)
        # Print with row number
        print(f"{y:2d}: {line}")


def main():
    board = construct_board()
    print_board(board)
    print()

    gm = build_game_model(board)
    print("=== GAME MODEL ===")
    print(f"Board size: {gm.width}x{gm.height}")
    print(f"Player pos: {gm.player_pos}")
    print(f"Goal positions: {gm.goal_positions}")
    print(f"Components ({len(gm.components)}):")
    for i, c in enumerate(gm.components):
        print(f"  [{i}] color={c.dominant_color}, size={c.size}, "
              f"bbox={c.bbox}, touches_border={c.touches_border}")
        print(f"       cells={c.cells}")
    print(f"Obstacles ({len(gm.obstacles)}):")
    for ox, oy, otype in gm.obstacles:
        print(f"  ({ox},{oy}) type={otype}")
    print(f"Walkable cells: {len(gm.walkable)}")
    print(f"Features:")
    print(f"  background_color: {gm.features.background_color}")
    print(f"  player_detectable: {gm.features.player_detectable}")
    print(f"  goal_detectable: {gm.features.goal_detectable}")
    print(f"  board_type: {gm.features.board_type}")
    print(f"  tile_size: {gm.features.tile_size}")
    print(f"  has_gravity: {gm.features.has_gravity}")
    print(f"  has_click_interaction: {gm.features.has_click_interaction}")
    print(f"  has_breakable_blocks: {gm.features.has_breakable_blocks}")
    print(f"  path_to_goal_possible: {gm.features.path_to_goal_possible}")
    print(f"  player_color: {gm.features.player_color}")
    print(f"  goal_color: {gm.features.goal_color}")
    print(f"  movement_axes: {gm.features.movement_axes}")

    print()
    print("=== REDUCER ===")
    root = reduce(gm)
    print(f"Root objective: {root.description} (priority={root.priority})")
    print("Sub-objectives:")
    def print_obj(obj, indent=2):
        print(" " * indent + f"{obj.description} (priority={obj.priority})")
        for sub in obj.sub_objectives:
            print_obj(sub, indent + 2)
    print_obj(root)

    leaf = select_leaf(root)
    print(f"Leaf objective: {leaf.description}")

    print()
    print("=== PLAN GENERATOR ===")
    plan = generate_plan(gm)
    if plan:
        print(f"Plan objective: {plan.objective_id}")
        print(f"Steps ({len(plan.steps)}):")
        for i, step in enumerate(plan.steps):
            print(f"  [{i}] action={step.action}, params={step.params}, "
                  f"source={step.source}, expected_effect={step.expected_effect}")
    else:
        print("No plan generated")


if __name__ == "__main__":
    main()
