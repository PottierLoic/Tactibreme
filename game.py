from enum import Enum
from board import Board, GameFinished
from color import Color

class Game:
    def __init__(self):
        self.board = Board()
        self.current_turn = Color.BLUE
        self.agent1 = Color.BLUE
        self.agent2 = Color.RED
        self.retreat_activated = False

    def play_turn(self, selected_paw, destination):
        """
        Execute the player's turn by moving the selected paw to the destination.
        Args:
            selected_paw (Paw): The paw the player wants to move.
            destination (tuple[int, int]): The target position for the paw.
        """
        if selected_paw.color != self.current_turn:
            return f"It's not {selected_paw.color}'s turn."
        if self.retreat_activated and not self.board.valid_retreat_move(selected_paw, destination):
            return f"This move is not valid during retreat."
        try:
            self.retreat_activated = False
            possible_moves = self.board.possible_movements(selected_paw)
            if destination not in possible_moves:
              return f"Invalid move. {destination} is not a valid destination."
            if self.board.move_paw(selected_paw, destination) == 1:
              print(f"{self.current_turn} activated the retreat.")
              self.retreat_activated = True
        except ValueError as e:
            return str(e)
        if self.board.check_win(destination) != -1:
          raise GameFinished(self.current_turn)
        self.current_turn = Color.RED if self.current_turn == Color.BLUE else Color.BLUE
