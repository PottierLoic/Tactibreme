from typing import Tuple
from board import Board
from color import Color

def calculate_reward(board: Board, move: Tuple[int, Tuple[int, int]], color: Color) -> int:
    paw_index, destination = move
    all_paws = [paw for paw_list in board.paws_coverage.values() for paw in paw_list]
    agent_paws = board.get_unicolor_list(all_paws, color)
    selected_paw = agent_paws[paw_index]
    board.move_paw(selected_paw, destination)
    if board.check_win(move[1]) == color:
        return 100
    return 0
    