from tqdm import tqdm
from enum import Enum
from logger import get_logger
from ai.agent import Agent, decode_action, encode_action
from ai.network import Network
from board import Board, GameFinished
from color import Color
from paw import Paw
from reward import calculate_reward


class Game:
    def __init__(self, agent1_path=None, agent2_path=None, num_games=1000, mode="train", **agent_params):
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
            agent_params (dict, optional): Additional hyperparameters for initializing agents.
        """
        self.board = Board()
        self.current_turn = Color.BLUE
        self.retreat_activated = False
        self.retreat_position = None
        self.real_player = mode == "play_vs_ai"
        self.num_games = num_games
        self.mode = mode

        # TODO: get name of the models from main for saving purposes later
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

        if mode == "train":
            if agent1_path:
                self.agent1.load_checkpoint(agent1_path)
                get_logger(__name__).info(f"Loaded Agent 1 from {agent1_path}")
            if agent2_path:
                self.agent2.load_checkpoint(agent2_path)
                get_logger(__name__).info(f"Loaded Agent 2 from {agent2_path}")

    def train_agents(self) -> None:
        """
        Train the AI agents for a specified number of games.
        """
        progress_bar = tqdm(range(self.num_games), desc="Training Progress", unit="game")
        for game in progress_bar:
            self.reset_game()
            done = False
            while not done:
                agent = self.agent1 if self.current_turn == Color.BLUE else self.agent2
                state_tensor = agent.encode_board(self.board)
                valid_moves = self.get_valid_moves(self.current_turn)
                if not valid_moves:
                    get_logger(__name__).debug("No valid moves, ending game.")
                    break
                move_idx = agent.select_action(self.board, valid_moves)
                paw_index, destination = decode_action(move_idx)
                all_paws = [paw for paw_list in self.board.paws_coverage.values() for paw in paw_list]
                agent_paws = self.board.get_unicolor_list(all_paws, self.current_turn)
                selected_paw = agent_paws[paw_index]

                reward = calculate_reward(self.board, (paw_index, destination), self.current_turn)
                self.process_move(selected_paw, destination)

                next_state_tensor = agent.encode_board(self.board)
                agent.store_transition(state_tensor, move_idx, reward, next_state_tensor, done=False)
                agent.train(batch_size=32)

                self.current_turn = Color.RED if self.current_turn == Color.BLUE else Color.BLUE

            if (game + 1) % 10 == 0:
                self.agent1.save_checkpoint("agent1_checkpoint.pth")
                self.agent2.save_checkpoint("agent2_checkpoint.pth")
                get_logger(__name__).info(f"Checkpoint saved at game {game + 1}")
            progress_bar.set_description(f"Training Game {game + 1}/{self.num_games}")

        self.agent1.save_checkpoint("agent1_checkpoint.pth")
        self.agent2.save_checkpoint("agent2_checkpoint.pth")
        get_logger(__name__).info("Training complete!")

    def reset_game(self) -> None:
        """
        Resets the game to its initial state to start a new round.
        """
        self.board = Board()
        self.current_turn = Color.BLUE
        self.retreat_activated = False
        self.retreat_position = None
        get_logger(__name__).debug("Game has been reset. A new game starts!")

    def process_move(self, selected_paw: Paw, destination: tuple[int, int]) -> None:
        """
        Handles the logic for moving a pawn and checking for retreat activation.

        Args:
            selected_paw (Paw): The pawn to move.
            destination (tuple[int, int]): The target position.
        """
        self.retreat_activated = False
        possible_moves = self.board.possible_movements(selected_paw)
        if destination not in possible_moves:
            return f"Invalid move. {destination} is not a valid destination."
        if self.board.move_paw(selected_paw, destination) == 1:
            get_logger(__name__).debug(f"{self.current_turn} activated the retreat.")
            self.retreat_position = destination
            self.retreat_activated = True

        if self.board.check_win(destination):
            get_logger(__name__).debug(f"The winner is {selected_paw.color}!")
            self.reset_game()
            return

    def play_turn(
        self, selected_paw: Paw = None, destination: tuple[int, int] = None
    ) -> None:
        """
        Execute a turn in the game. If real_player is False, AI agents take turns automatically.
        Otherwise, the human player controls their pieces.

        Args:
            selected_paw (Paw, optional): The paw the player wants to move (only used if real_player is True).
            destination (tuple[int, int], optional): The target position for the paw (same).
        """
        if self.real_player:
            if selected_paw is None or destination is None:
                return "Invalid move. Please select a pawn and destination."
            if selected_paw.color != self.current_turn:
                return f"It's not {selected_paw.color}'s turn."
            if self.retreat_activated and not self.board.valid_retreat_move(
                selected_paw, destination, self.retreat_position
            ):
                return "This move is not valid during retreat."
            self.process_move(selected_paw, destination)
        else:
            agent = self.agent1 if self.current_turn == Color.BLUE else self.agent2
            state_tensor = agent.encode_board(self.board)
            agent.color = self.current_turn
            valid_moves = self.board.get_valid_moves(self.current_turn)
            if not valid_moves:
                get_logger(__name__).debug("AI has no valid moves.")
                return
            move_idx = agent.select_action(self.board, valid_moves)
            paw_index, destination = decode_action(move_idx)
            all_paws = [
                paw
                for paw_list in self.board.paws_coverage.values()
                for paw in paw_list
            ]
            agent_paws = self.board.get_unicolor_list(all_paws, self.current_turn)
            selected_paw = agent_paws[paw_index]
            reward = calculate_reward(
                self.board, (paw_index, destination), self.current_turn
            )
            self.process_move(selected_paw, destination)
            next_state_tensor = agent.encode_board(self.board)
            agent.store_transition(
                state_tensor, move_idx, reward, next_state_tensor, done=False
            )
            agent.train(batch_size=32)
        self.current_turn = Color.RED if self.current_turn == Color.BLUE else Color.BLUE

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
