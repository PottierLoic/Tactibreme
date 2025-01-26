from enum import Enum

from ai.agent import Agent
from ai.network import Network
from board import Board, GameFinished
from paw import Paw
from color import Color
from reward import calculate_reward


class Game:
    def __init__(self):
        self.board = Board()
        self.current_turn = Color.BLUE
        self.retreat_activated = False
        self.retreat_position = None

        self.real_player = False

        self.network1 = Network()
        self.network2 = Network()
        self.agent1 = Agent(
            color=Color.BLUE,
            network=self.network1,
            epsilon=0.1,
            gamma=0.99,
            learning_rate=1e-3,
            buffer_size=10000,
        )
        self.agent2 = Agent(
            color=Color.RED,
            network=self.network2,
            epsilon=0.1,
            gamma=0.99,
            learning_rate=1e-3,
            buffer_size=10000,
        )

    def reset_game(self) -> None:
        """
        Resets the game to its initial state to start a new round.
        """
        self.board = Board()
        self.current_turn = Color.BLUE
        self.retreat_activated = False
        self.retreat_position = None
        print("Game has been reset. A new game starts!")

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
            print(f"{self.current_turn} activated the retreat.")
            self.retreat_position = destination
            self.retreat_activated = True

        winner = self.board.check_win(destination)
        if winner != -1:
            print(f"The winner is {winner}! Restarting the game...")
            self.reset_game()
            return

    def play_turn(self, selected_paw: Paw = None, destination: tuple[int, int] = None) -> None:
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
            current_agent = self.agent1 if self.current_turn == Color.BLUE else self.agent2
            valid_moves = self.board.get_valid_moves(self.current_turn)
            if not valid_moves:
                print("AI has no valid moves.")
                return
            state_tensor = current_agent.encode_board(self.board)
            move = current_agent.select_action(self.board, valid_moves)
            if move:
                paw_index, destination = move
                all_paws = [paw for paw_list in self.board.paws_coverage.values() for paw in paw_list]
                agent_paws = self.board.get_unicolor_list(all_paws, self.current_turn)
                selected_paw = agent_paws[paw_index]
                reward = calculate_reward(self.board, move, self.current_turn)
                self.process_move(selected_paw, destination)
                next_state_tensor = current_agent.encode_board(self.board)
                current_agent.store_transition(state_tensor, move, reward, next_state_tensor, done=False)
                current_agent.train(batch_size=32)
        self.current_turn = Color.RED if self.current_turn == Color.BLUE else Color.BLUE
