import random
import torch
from torch import nn, Tensor
from typing import List, Tuple, Deque
from collections import deque
from board import Board
from color import Color

class Agent:
    def __init__(
        self,
        color: Color,
        network: nn.Module,
        epsilon: float = 0.1,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
    ) -> None:
        """
        Initialize the Agent.
        Args:
            color (Color): The agent's color (RED or BLUE).
            network (nn.Module): The neural network for decision-making.
            epsilon (float): Exploration rate for epsilon-greedy policy.
            gamma (float): Discount factor for future rewards.
            learning_rate (float): Learning rate for the optimizer.
            buffer_size (int): Maximum size of the replay buffer.
        """
        self.color: Color = color
        self.network: nn.Module = network
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory: Deque[Tuple[Tensor, int, float, Tensor, bool]] = deque(maxlen=buffer_size)
        self.criterion: nn.Module = nn.MSELoss()

    def encode_board(self, board: Board) -> Tensor:
        """
        Encode the board state into a tensor.
        Args:
            board (Board): The current board state.
        Returns:
            Tensor: Encoded board state of shape (1, 8, 5, 5).
        """
        state = torch.zeros((8, 5, 5), dtype=torch.float32)
        for pos, paws in board.paws_coverage.items():
            row, col = pos
            for paw in paws:
                idx = paw.paw_type.value - 1
                if paw.color == Color.BLUE:
                    state[idx, row, col] = 1
                elif paw.color == Color.RED:
                    state[idx + 4, row, col] = 1
        return state.unsqueeze(0)

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

    def select_action(self, board: Board, valid_moves: List[Tuple[int, Tuple[int, int]]]) -> Tuple[int, Tuple[int, int]]:
        pass

    def store_transition(self, state: Tensor, action: int, reward: float, next_state: Tensor, done: bool) -> None:
        pass

    def train(self, batch_size: int = 32) -> None:
        pass
