import copy
from typing import Tuple
from board import Board
from color import Color

def calculate_reward(board: Board, move: Tuple[int, Tuple[int, int]], color: Color) -> int:
    board_copy = copy.deepcopy(board)
    paw_index, destination = move
    copy_all_paws = [paw for paw_list in board_copy.paws_coverage.values() for paw in paw_list]
    copy_agent_paws = board_copy.get_unicolor_list(copy_all_paws, color)
    selected_paw_copy = copy_agent_paws[paw_index]
    board_copy.move_paw(selected_paw_copy, destination)
    if board_copy.check_win(destination) == color:
        return 100
    return 0
