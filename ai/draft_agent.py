import random
import torch
from collections import deque
from typing import Deque, List, Tuple
from torch import Tensor, nn
from board import Board
from color import Color
from paw import Paw
from ai.base_agent import AgentBase

class DraftAgent(AgentBase):
  def __init__(
    self,
    color: Color,
    network: nn.Module,
    epsilon: float = 1,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    buffer_size: int = 10000,
    epsilon_off: bool = False,
    ) -> None:
      """
      Initialize the draft Agent.
      Args:
        color (Color): The draft agent color (RED or BLUE).
        network (nn.Module): The neural network for decision-making.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        gamma (float): Discount factor for future rewards.
        learning_rate (float): Learning rate for the optimizer.
        buffer_size (int): Maximum size of the replay buffer.
      """
      super().__init__(color, network, epsilon, gamma, learning_rate, buffer_size, epsilon_off)
      self.reward_counter = 0
      
  def encode_board(self, board: Board, reverse: bool = False) -> Tensor:
        """
        Encode the board state into a tensor.
        Args:
            board (Board): The current board state.
            reverse (bool): If True, rotate the board and swap colors.
        Returns:
            Tensor: Encoded board state of shape (1, 4, 2, 5).
        """
        state = torch.zeros((4, 2, 5), dtype=torch.float32)
        for pos, paws in board.paws_coverage.items():
            row, col = pos
            if row != 0 and row != 4:
              continue
            if reverse:
                col = 4 - col
            for paw in paws:
                idx = paw.paw_type.value
                row = 0 if paw.color == Color.RED else 1
                state[idx, row, col] = 1
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
                rand_paw = random.randint(0, 3)
                rand_dest = (0 if reverse else 4, random.randint(0, 4))
                return self.encode_action((rand_paw, rand_dest))

        q_values = self.network(state_tensor).detach().squeeze()
        q_values = torch.exp(q_values)
        probabilities = q_values / q_values.sum()
        move_index = torch.multinomial(probabilities, 1).item()
        return move_index

  def encode_action(self, move: tuple[int, tuple[int, int]]) -> int:
    """
    Convert (paw_index, (row, col)) into a single integer 0..20.
    """
    paw_index, (row, col) = move
    return paw_index * 5 + col

  def decode_action(self, action_idx: int) -> tuple[int, tuple[int, int]]:
    """
    Convert a single integer 0..20 back to (paw_index, (row, col)).
    """
    paw_index = action_idx // 5
    row = 0
    col = action_idx % 5
    return paw_index, (row, col)
