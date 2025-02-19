from typing import Tuple
from board import Board
from color import Color


def calculate_reward(
    board: Board, move: Tuple[int, Tuple[int, int]], color: Color
) -> int:
    total_reward = 0
    paw, destination = move
    if board.check_win(destination):
        total_reward += 100
    return total_reward
