import random
import torch
from collections import deque
from typing import Deque, List, Tuple
from logger import get_logger
from torch import Tensor, nn
from board import Board
from color import Color


class AgentBase:
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
        """s
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
        self.learning_rate: float = learning_rate
        self.buffer_size: int = buffer_size
        self.epsilon_off: bool = epsilon_off
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate
        )
        self.memory: Deque[Tuple[Tensor, int, float, Tensor, bool]] = deque(
            maxlen=buffer_size
        )
        self.criterion: nn.Module = nn.MSELoss()

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save the current model and hyperparameters to a file.
        Args:
            filepath (str): Path where the model will be saved.
        """
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
        }
        torch.save(checkpoint, filepath)
        get_logger(__name__).info(f"Model and hyperparameters saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a model and hyperparameters from a checkpoint file.
        Args:
            filepath (str): Path to the model checkpoint file.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.gamma = checkpoint["gamma"]
        self.learning_rate = checkpoint["learning_rate"]
        self.buffer_size = checkpoint["buffer_size"]
        self.network.eval()
        get_logger(__name__).info(f"Model and hyperparameters loaded from {filepath}")

    def create_mask(self, valid_moves: List[Tuple[int, Tuple[int, int]]]) -> Tensor:
        raise NotImplementedError("create_mask method not implemented.")

    def encode_board(self, board: Board, reverse: bool = False) -> Tensor:
        raise NotImplementedError("encode_board method not implemented.")

    def select_action(
        self, board: Board, valid_moves: List[Tuple[int, Tuple[int, int]]], reverse: bool = False
    ) -> Tuple[int, Tuple[int, int]]:
        raise NotImplementedError("select_action method not implemented.")

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

        device = self.network.device

        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.cat(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        q_values = self.network(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze()

        next_q_values = self.network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_epsilon()

    def update_epsilon(self, min_epsilon: float = 0.2, decay_amount: float = 0.001) -> None:
        """
        Subtract decay_amount from epsilon until it reach min_epsilon
        """
        self.epsilon = max(min_epsilon, self.epsilon - decay_amount)

    def encode_action(self, move: tuple[int, tuple[int, int]]) -> int:
        raise NotImplementedError("encode_action method not implemented.")

    def decode_action(self, action_idx: int) -> tuple[int, tuple[int, int]]:
        raise NotImplementedError("decode_action method not implemented.")
