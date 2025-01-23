from enum import Enum

from ai.agent import Agent
from ai.network import Network
from board import Board, GameFinished
from color import Color


class Game:
    def __init__(self):
        self.board = Board()
        self.current_turn = Color.BLUE
        self.agent1 = Color.BLUE
        self.agent2 = Color.RED
        self.retreat_activated = False
        self.retreat_position = None

        self.network = Network()
        self.agent_red = Agent(
            color=Color.RED,
            network=self.network,
            epsilon=0.1,
            gamma=0.99,
            learning_rate=1e-3,
            buffer_size=10000,
        )

    def play_turn(self, selected_paw, destination):
        """
        Execute the player's turn by moving the selected paw to the destination.
        Args:
            selected_paw (Paw): The paw the player wants to move.
            destination (tuple[int, int]): The target position for the paw.
        """
        if selected_paw.color != self.current_turn:
            return f"It's not {selected_paw.color}'s turn."
        if self.retreat_activated and not self.board.valid_retreat_move(
            selected_paw, destination, self.retreat_position
        ):
            return f"This move is not valid during retreat."
        try:
            self.retreat_activated = False
            possible_moves = self.board.possible_movements(selected_paw)
            if destination not in possible_moves:
                return f"Invalid move. {destination} is not a valid destination."
            if self.board.move_paw(selected_paw, destination) == 1:
                print(f"{self.current_turn} activated the retreat.")
                self.retreat_position = destination
                self.retreat_activated = True
        except ValueError as e:
            return str(e)
        if self.board.check_win(destination) != -1:
            raise GameFinished(self.current_turn)
        self.current_turn = Color.RED if self.current_turn == Color.BLUE else Color.BLUE

        if self.current_turn == Color.RED:
            self.play_ai_turn()

    def play_ai_turn(self):
        valid_moves = self.board.get_valid_moves(Color.RED)

        if not valid_moves:
            print("AI has no valid moves.")
            return

        state_tensor = self.agent_red.encode_board(self.board)
        print(f"State Tensor Shape: {state_tensor.shape}")

        move = self.agent_red.select_action(self.board, valid_moves)
        print(f"AI selected move: {move}")

        if move:
            paw_index, destination = move
            all_paws = [
                paw
                for paw_list in self.board.paws_coverage.values()
                for paw in paw_list
            ]
            red_paws = self.board.get_unicolor_list(all_paws, Color.RED)

            selected_paw = red_paws[paw_index]

            result = self.play_turn(selected_paw, destination)

            reward = 0 # palceholder
            next_state_tensor = self.agent_red.encode_board(self.board)
            self.agent_red.store_transition(
                state_tensor, move, reward, next_state_tensor, done=False
            )
            self.agent_red.train(batch_size=32)

            if isinstance(result, str):
                print(result)
        else:
            print("AI could not make a move.")
