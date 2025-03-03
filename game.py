import random
from tqdm import tqdm
from enum import Enum
from logger import get_logger
from ai.agent import Agent, agent_decode_action, agent_encode_action
from ai.network import Network
from ai.draft_network import Draft_network
from ai.placeur import Placeur, placeur_decode_action, placeur_encode_action
from board import Board, GameFinished
from color import Color
from paw import Paw
from reward import calculate_reward
from stats import Stats
from writerBuffer import init_writer, WriterBuffer
import time
from paw import *

class Game:
    def __init__(self, agent1_path=None, agent2_path=None, num_games=1000, mode="train", model_name=None, **agent_params):
        """
        Initializes the game with different modes:

        - "train": Loads two agent checkpoints (or starts fresh if none are provided) and trains them.
        - "ai_vs_ai": Runs a fixed number of games with two AIs facing each other (without training).
        - "play_vs_ai": Allows a real player to play against an AI (without training).

        Args:
            agent1_path (str, optional): Path to the saved model of the first agent.
            agent2_path (str, optional): Path to the saved model of the second agent.
            num_games (int, optional): Number of games to play/train. Default is 1000.
            mode (str, optional): Mode of operation: "train", "ai_vs_ai", or "play_vs_ai". Default is "train".
            model_name (str, optional): Name used to save the trained models. Required for train mode.
            agent_params (dict, optional): Additional hyperparameters for initializing agents.
        """
        self.board = Board()
        self.current_turn = Color.BLUE
        self.retreat_activated = False
        self.retreat_position = None
        self.real_player = mode == "play_vs_ai"
        self.num_games = num_games
        self.mode = mode
        self.model_name = model_name
        self.stats = Stats()
        self.placeur_buffer : dict[Color, old] = {} # Color -> [(tensor_t0, move_idx, is_valid, tensor_t1)]

        if mode in ["train", "ai_vs_ai"]:
            self.agent1 = Agent(
                color=Color.BLUE,
                network=Network(),
                **agent_params
            )
            self.agent2 = Agent(
                color=Color.RED,
                network=Network(),
                **agent_params
            )
            self.placeur1 = Placeur(
                color=Color.BLUE,
                network=Draft_network(),
                # **placeur_params
            )
            self.placeur2 = Placeur(
                color=Color.RED,
                network=Draft_network(),
                # **placeur_params
            )
            if agent1_path:
                self.agent1.load_checkpoint(agent1_path)
                get_logger(__name__).info(f"Loaded Agent 1 from {agent1_path}")
            if agent2_path:
                self.agent2.load_checkpoint(agent2_path)
                get_logger(__name__).info(f"Loaded Agent 2 from {agent2_path}")

    def draft(self, STOP_EVENT) -> None:
        self.placeur_buffer[Color.BLUE] = []
        self.placeur_buffer[Color.RED] = []
        for turn in range(8):
            if STOP_EVENT.is_set():
                STOP_EVENT.set()
                get_logger(__name__).info("Draft aborted")
                return
            is_move_valid = False
            is_red_turn = turn % 2 == 1
            placeur = self.placeur1 if not is_red_turn else self.placeur2
            state_tensor = placeur.encode_board(self.board, reverse=(is_red_turn))
            while not is_move_valid:
                move_idx = placeur.select_action(self.board, [], reverse=(is_red_turn)) # pourquoi on donne board et pas le tenseur ?
                paw_idx, pos = placeur_decode_action(move_idx)
                row, col = pos
                row = 0 if is_red_turn else 4
                pos = (row, col)
                paw = Paw(PawType(paw_idx), placeur.color, pos)
                is_move_valid = self.board.init_paw(paw, pos)
                next_state_tensor = placeur.encode_board(self.board, reverse=(is_red_turn))
                self.placeur_buffer[placeur.color].append((state_tensor, move_idx, is_move_valid, next_state_tensor))
                # reward = 5 if is_move_valid else -10
                # placeur.store_transition(state_tensor, move_idx, reward, next_state_tensor, done=False)
                # placeur.train(batch_size=32)
        self.placeur1.encode_board(self.board)
        return

    def train_agents(self, STOP_EVENT) -> None:
        """
        Train the AI agents for a specified number of games.
        """
        if self.mode == "train" and not self.model_name:
            raise ValueError("model_name is required for training mode")

        filename = "train_rec_{}_{}.csv".format(self.num_games, self.model_name)
        wrt, _ = init_writer(filename)
        self.writer = WriterBuffer(wrt)

        progress_bar = tqdm(range(self.num_games), desc="Training Games", unit="game", dynamic_ncols=True)
        for _ in range(self.num_games):
            self.reset_game()
            self.draft(STOP_EVENT)
            game_finished = False
            while (not game_finished) and (not STOP_EVENT.is_set()):
                agent = self.agent1 if self.current_turn == Color.BLUE else self.agent2
                if agent == self.agent1:
                    self.writer.set_agent(0)
                else:
                    self.writer.set_agent(1)
                self.writer.set_epsilon(agent.epsilon)
                state_tensor = agent.encode_board(self.board, reverse=(self.current_turn == Color.RED))
                valid_moves = self.get_valid_moves(self.current_turn)
                if not valid_moves:
                    get_logger(__name__).debug("No valid moves, ending game.")
                    break
                move_idx = agent.select_action(self.board, valid_moves, reverse=(self.current_turn == Color.RED))
                paw_index, destination = agent_decode_action(move_idx)
                all_paws = [paw for paw_list in self.board.paws_coverage.values() for paw in paw_list]
                agent_paws = self.board.get_unicolor_list(all_paws, self.current_turn)
                selected_paw = agent_paws[paw_index]

                reward = calculate_reward(self.board, (paw_index, destination), self.current_turn)
                self.writer.set_reward(reward)
                m = self.process_move(selected_paw, destination)
                self.writer.set_win(m)
                self.writer.push()
                self.writer.reset_line()
                if (m == 1):
                    for color in self.placeur_buffer:
                        placeur = self.placeur1 if color == self.placeur1.color else self.placeur2
                        for tensor_t0, move_idx, is_valid, tensor_t1 in self.placeur_buffer[color]:
                            reward = 0
                            if is_valid:
                                if color == self.current_turn:
                                    reward = 100
                                else:
                                    reward = -100
                            else:
                                reward = -5
                            reward = 100 if color == self.current_turn and is_valid else -10
                            placeur.store_transition(state_tensor, move_idx, reward, next_state_tensor, done=False)
                            placeur.train()
                    game_finished = True
                next_state_tensor = agent.encode_board(self.board, reverse=(self.current_turn == Color.RED))
                agent.store_transition(state_tensor, move_idx, reward, next_state_tensor, done=False)
                agent.train(batch_size=32)
                self.current_turn = Color.RED if self.current_turn == Color.BLUE else Color.BLUE
            if STOP_EVENT.is_set():
                STOP_EVENT.set()
                get_logger(__name__).info("Training aborted")
                return
            progress_bar.update(1)
        self.agent1.save_checkpoint(f"{self.model_name}_blue.pth")
        self.agent2.save_checkpoint(f"{self.model_name}_red.pth")
        get_logger(__name__).info(f"Training complete! Models saved as {self.model_name}_blue.pth and {self.model_name}_red.pth")
        progress_bar.close()

    def reset_game(self) -> None:
        """
        Resets the game to its initial state to start a new round.
        """
        self.board = Board()
        self.current_turn = Color.BLUE
        self.retreat_activated = False
        self.retreat_position = None
        self.stats = Stats()
        get_logger(__name__).debug("Game has been reset. A new game starts!")

    def process_move(self, selected_paw: Paw, destination: tuple[int, int]) -> None:
        """
        Handles the logic for moving a pawn and checking for retreat activation.

        Args:
            selected_paw (Paw): The pawn to move.
            destination (tuple[int, int]): The target position.
        """
        self.retreat_activated = False
        self.stats.moves_counter += 1
        possible_moves = self.board.possible_movements(selected_paw)
        if destination not in possible_moves:
            self.stats.invalid_moves += 1
            return -1
        self.writer.set_color(selected_paw.color.value)
        self.writer.set_paw(selected_paw.paw_type.value)
        self.writer.set_dest(destination[0], destination[1])
        if self.board.move_paw(selected_paw, destination) == 1:
            get_logger(__name__).debug(f"{self.current_turn} activated the retreat.")
            self.retreat_position = destination
            self.retreat_activated = True
        if self.board.check_win(destination):
            get_logger(__name__).debug(f"The winner is {selected_paw.color}!")
            return 1
        return 0

    def get_valid_moves(self, color: Color) -> list[tuple[int, tuple[int, int]]]:
        """
        Retrieve all possible moves for the given color.

        Args:
            color (Color): The color to get moves from.

        Returns:
            list[tuple[int, tuple[int, int]]]: A list of (paw_index, destination).
        """
        valid_moves = []

        all_paws = [
            paw for paw_list in self.board.paws_coverage.values() for paw in paw_list
        ]

        paws = self.board.get_unicolor_list(all_paws, color)

        for index, paw in enumerate(paws):
            possible_moves = self.board.possible_movements(paw)
            for destination in possible_moves:
                valid_moves.append((index, destination))
        get_logger(__name__).debug(valid_moves)
        return valid_moves

    def record_games(self, STOP_EVENT) -> None:
        """
        Run AI vs AI matches without training.
        """
        filename = "record_rec_{}_{}.csv".format(self.num_games, self.model_name)
        wrt, _ = init_writer(filename)
        self.writer = WriterBuffer(wrt)
        agent1_wins = 0
        progress_bar = tqdm(range(self.num_games), desc="Playing Games", unit="game")
        for _ in range(self.num_games):
            if STOP_EVENT.is_set():
                break
            self.reset_game()
            self.draft(STOP_EVENT)
            self.current_turn = random.choice([Color.BLUE, Color.RED])
            game_finished = False
            while not game_finished and not STOP_EVENT.is_set():
                agent = self.agent1 if self.current_turn == Color.BLUE else self.agent2
                if agent == self.agent1:
                    self.writer.set_agent(0)
                else:
                    self.writer.set_agent(1)
                state_tensor = agent.encode_board(self.board, reverse=(self.current_turn == Color.RED))
                valid_moves = self.get_valid_moves(self.current_turn)
                if not valid_moves:
                    get_logger(__name__).debug("No valid moves, ending game.")
                    break
                move_idx = agent.select_action(self.board, valid_moves, reverse=(self.current_turn == Color.RED))
                paw_index, destination = decode_action(move_idx)
                all_paws = [paw for paw_list in self.board.paws_coverage.values() for paw in paw_list]
                agent_paws = self.board.get_unicolor_list(all_paws, self.current_turn)
                selected_paw = agent_paws[paw_index]
                m = self.process_move(selected_paw, destination)
                self.writer.set_win(m)
                self.writer.push()
                self.writer.reset_line()
                if (m == 1):
                    game_finished = True
                    if agent == self.agent1:
                        agent1_wins += 1
                self.current_turn = Color.RED if self.current_turn == Color.BLUE else Color.BLUE
            progress_bar.update(1)
        progress_bar.close()
        print(f"Agent1 win rate: {float(agent1_wins) / self.num_games * 100:.2f}%")
        if STOP_EVENT.is_set():
            get_logger(__name__).info("Recording aborted")
        else:
            get_logger(__name__).info("Recording complete!")
