import random
from collections import deque
from typing import Deque, List, Tuple

import torch
from torch import Tensor, nn

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
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate
        )
        self.memory: Deque[Tuple[Tensor, int, float, Tensor, bool]] = deque(
            maxlen=buffer_size
        )
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

    def select_action(
        self, board: Board, valid_moves: List[Tuple[int, Tuple[int, int]]]
    ) -> Tuple[int, Tuple[int, int]]:
        """
        Select an action based on the current policy.
        Args:
            board (Board): The current board state.
            valid_moves (list): List of valid moves (paw_index, destination). # TODO
        Returns:
            tuple: Selected (paw_index, destination).
        """
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        encoded_board = self.encode_board(board)
        output = self.network(encoded_board.unsqueeze(0)).detach().squeeze()
        mask = self.create_mask(valid_moves)

        masked_output = output * mask
        masked_output[mask == 0] = -float("inf")

        best_move_index = torch.argmax(masked_output).item()
        paw_index = best_move_index // 25
        destination_index = best_move_index % 25
        row, col = destination_index // 5, destination_index % 5
        return paw_index, (row, col)

    def store_transition(
        self, state: Tensor, action: int, reward: float, next_state: Tensor, done: bool
    ) -> None:
        """
        Store a transition in the replay buffer.
        Args:
            state (Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (Tensor): Next state.
            done (bool): Whether the game is over.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size: int = 32) -> None:
        """
        Train the network using experience replay.
        Args:
            batch_size (int): Number of transitions to sample from the replay buffer.
        """
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.network(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze()

        next_q_values = self.network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
