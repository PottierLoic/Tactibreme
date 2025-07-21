import random
import torch
from collections import deque
from typing import Deque, List, Tuple
from logger import get_logger
from torch import Tensor, nn
from board import Board
from color import Color
from ai.base_agent import AgentBase

class GameAgent(AgentBase):
    def __init__(
        self,
        color: Color,
        network: nn.Module,
        epsilon: float = 1,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
        epsilon_off: bool = False
    ) -> None:
        super().__init__(color, network, epsilon, gamma, learning_rate, buffer_size, epsilon_off)

    def create_mask(self, valid_moves: List[Tuple[int, Tuple[int, int]]]) -> Tensor:
        """
        Create a mask for valid moves.
        Args:
            valid_moves (list): List of valid (paw_index, destination) pairs.
        Returns:
            Tensor: A mask of size 100 with 1 for valid moves, 0 otherwise.
        """
        mask = torch.zeros(100, dtype=torch.float32)
        for paw_index, (row, col) in valid_moves:
            flat_index = paw_index * 25 + row * 5 + col
            mask[flat_index] = 1
        return mask

    def encode_board(self, board: Board, reverse: bool = False) -> Tensor:
        """
        Encode the board state into a tensor.
        Args:
            board (Board): The current board state.
            reverse (bool): If True, rotate the board and swap colors.
        Returns:
            Tensor: Encoded board state of shape (1, 8, 5, 5).
        """
        state = torch.zeros((8, 5, 5), dtype=torch.float32)
        for pos, paws in board.paws_coverage.items():
            row, col = pos
            if reverse:
                row = 4 - row
                col = 4 - col
            for paw in paws:
                idx = paw.paw_type.value - 1
                channel = idx + 4 if ((not reverse and paw.color == Color.RED) or (reverse and paw.color == Color.BLUE)) else idx
                state[channel, row, col] = 1
        return state.unsqueeze(0)

    def select_action(
        self, board: Board, valid_moves: List[Tuple[int, Tuple[int, int]]], reverse: bool = False
    ) -> Tuple[int, Tuple[int, int]]:
        """
        Select an action based on the current policy.
        Args:
            board (Board): The current board state.
            valid_moves (list): List of valid moves (paw_index, destination). # TODO
            reverse (bool): If True, rotate the board and swap colors.
        Returns:
            tuple: Selected (paw_index, destination).
        """
        state_tensor = self.encode_board(board, reverse)
        if not self.epsilon_off:
            if random.random() < self.epsilon:
                random_move = random.choice(valid_moves)
                return self.encode_action(random_move)

        q_values = self.network(state_tensor).detach().squeeze()
        mask = self.create_mask(valid_moves).to(self.network.device)
        q_values[mask == 0] = -1e9
        q_values = torch.exp(q_values) * mask
        if q_values.sum() == 0:
            probabilities = mask / mask.sum()
        else:
            probabilities = q_values / q_values.sum()
        move_index = torch.multinomial(probabilities, 1).item()
        return move_index

    def encode_action(self, move: tuple[int, tuple[int, int]]) -> int:
        """
        Convert (paw_index, (row, col)) into a single integer 0..99.
        """
        paw_index, (row, col) = move
        return paw_index * 25 + row * 5 + col

    def decode_action(self, action_idx: int) -> tuple[int, tuple[int, int]]:
        """
        Convert a single integer 0..99 back to (paw_index, (row, col)).
        """
        paw_index = action_idx // 25
        destination_index = action_idx % 25
        row = destination_index // 5
        col = destination_index % 5
        return paw_index, (row, col)
