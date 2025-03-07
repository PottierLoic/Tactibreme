import random
from color import Color
from paw import Paw, PawType
from logger import get_logger
from writerBuffer import WriterBuffer

class GameFinished(Exception):
    def __init__(self, winner_color):
        self.winner_color = winner_color


class Retreat:
    def __init__(self):
        self.is_activated = False
        self.position = None
        self.activator_color = None
        self.paw_to_move = None

class Board:
    def __init__(self) -> None:
        self.paws_coverage: dict[tuple[int, int], list[Paw]] = {}
        self.retreat_status = Retreat()

    def init_paw(self, paw: Paw, start_position: tuple[int, int]) -> bool:
        """
        Initialise a paw to a position.
        Args:
            paw (Paw): The paw to move.
            start_position (tuple[int, int]): The start position (row, col) where the row should be
                either 0 (Blue) or 4 (Red).
        Returns:
            bool: False if the start position is invalid, True otherwise.
        """
        if not self.is_valid_position(start_position):
            raise ValueError(f"Invalid start position: {start_position}")
        if not self.is_valid_start(paw.color, start_position):
            return False
        if self.is_paw_duplicated(paw):
            return False
        self.paws_coverage[start_position] = [paw]
        return True

    def move_paw(self, paw: Paw, destination: tuple[int, int], writer) -> bool:
        """
        Move a paw to a new position.
        Args:
            paw (Paw): The paw to move.
            destination (tuple[int, int]): The new position (row, col).
        Returns:
            bool: False if the move is invalid, True otherwise.
        """
        if not self.is_valid_position(destination):
            raise ValueError(f"Invalid destination: {destination}")
        if destination not in self.possible_movements(paw):
            return False
        self.retreat_status.is_activated = False
        origin_pos = paw.position
        paw_at_origin = self.find_paw_at(origin_pos)
        pawns_to_move = [
            p
            for p in paw_at_origin
            if p is paw or p.paw_type.value > paw.paw_type.value
        ]
        dog_present = any(p.paw_type.value == 1 and p is not paw for p in pawns_to_move)
        cat_present = any(p.paw_type.value == 2 and p is not paw for p in pawns_to_move)
        rooster_present = any(p.paw_type.value == 3 and p is not paw for p in pawns_to_move)
        writer.set_pile(dog_present, cat_present, rooster_present)
        self.paws_coverage[origin_pos] = [
            p for p in paw_at_origin if p not in pawns_to_move
        ]
        if not self.paws_coverage[origin_pos]:
            del self.paws_coverage[origin_pos]
        for p in pawns_to_move:
            p.position = destination
        if destination not in self.paws_coverage:
            self.paws_coverage[destination] = []
        self.paws_coverage[destination].extend(pawns_to_move)
        if (
            (destination[0] == 0 and paw.color == Color.RED)
            or (destination[0] == 4 and paw.color == Color.BLUE)
        ) and self.is_blended(self.paws_coverage[destination], paw.color):
            get_logger(__name__).debug(f"retraite activée par {paw.color}")
            self.retreat_status.is_activated = True
            self.retreat_status.position = destination
            self.retreat_status.activator_color = paw.color
            self.retreat_status.paw_to_move = self.get_biggest_paw(
                self.other_color(paw.color), self.paws_coverage[destination]
            )
            return True
        return False

    def other_color(self, color: Color) -> Color:
        return Color((color.value + 1) % 2)

    def get_biggest_paw(self, color: Color, paws: list[Paw]) -> Paw:
        return sorted(
            self.get_unicolor_list(paws, color), key=lambda paw: paw.paw_type.value
        )[0]

    def valid_retreat_move(
        self, paw: Paw, destination: tuple[int, int], retreat_position: tuple[int, int]
    ) -> int:
        """
        Check if a move is a valid retreat move.
        Args:
            paw (Paw): The paw to move.
            destination (tuple[int, int]): The destination to check.
            retreat_position (tuple[int, int]): The position where the retreat appears.
        Returns:
            Int: 0 -> movement not valid
                 1 -> movement valid
                -1 -> movement valid BUT no moves possible
        """
        if paw.position == retreat_position:
            unicolor_list = self.get_unicolor_list(
                self.paws_coverage[retreat_position], paw.color
            )
            if unicolor_list[0] == paw:
                if len(self.possible_movements(paw)) != 0:
                    return 1
                return -1
        return 0

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        """
        Check if a position is within the board boundaries.
        Args:
            position (tuple[int, int]): The position to check (row, col).
        Returns:
            bool: True if the position is valid, False otherwise.
        """
        row, col = position
        return 0 <= row < 5 and 0 <= col < 5

    def is_valid_start(self, color: Color, position: tuple[int, int]) -> bool:
        """
        Check if a position is a valid start.
        The position should be empty.
        The blue row should be 4 and the red one should be 0.
        Args:
            color (Color): The color of the moving paw.
            position (tuple[int, int]): The position to check (row, col).
        Returns:
            bool: True if the start position is valid, False otherwise.
        """
        row, col = position
        empty_position = not (position in self.paws_coverage)
        match_color_row = (color.value == 0 and row == 4) or (color.value == 1 and row == 0)
        return empty_position and match_color_row

    def is_paw_duplicated(self, paw: Paw) -> bool:
        """
        Check if the paw is already on the board
        Args:
            paw (Paw): The paw to check.
        Returns:
            bool: True if the paw is already on the board, False otherwise.
        """
        row = 4 if paw.color == Color.BLUE else 0
        for col in range (5):
            position = (row, col)
            if position in self.paws_coverage:
                for p in self.paws_coverage[position]:
                    if p.paw_type == paw.paw_type:
                        return True
        return False

    def find_paw_at(self, position: tuple[int, int]) -> list[Paw]:
        """
        Find all paws at the given position.
        Args:
            position (tuple[int, int]): The position from which to retrieve paws.
        Returns:
            list[Paw]: A list of paws at the given position, or an empty list if none.
        """
        if position in self.paws_coverage:
            return self.paws_coverage[position]
        else:
            return []

    def get_movements(self, paw: Paw) -> list[tuple[int, int]]:
        directions = {
            PawType.DONKEY: [(1, 0), (-1, 0), (0, 1), (0, -1)],
            PawType.DOG: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            PawType.ROOSTER: [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ],
        }
        moves = []
        if paw.paw_type in directions:
            for dx, dy in directions[paw.paw_type]:
                step = 1
                while True:
                    nx, ny = paw.position[0] + step * dx, paw.position[1] + step * dy
                    if not self.is_valid_position((nx, ny)):
                        break
                    occupant = self.find_paw_at((nx, ny))
                    if occupant:
                        if occupant[-1].paw_type.value < paw.paw_type.value:
                            moves.append((nx, ny))
                        break
                    moves.append((nx, ny))
                    step += 1
        elif paw.paw_type == PawType.CAT:
            potential_moves = [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ]
            for dx, dy in potential_moves:
                nx, ny = paw.position[0] + dx, paw.position[1] + dy
                if self.is_valid_position((nx, ny)):
                    occupant = self.find_paw_at((nx, ny))
                    if not occupant or occupant[-1].paw_type.value < paw.paw_type.value:
                        moves.append((nx, ny))
        return moves

    def possible_movements(self, paw: Paw) -> list[tuple[int, int]]:
        """
        Return a list of all possible movements for a given paw.
        Args:
            paw (Paw): The paw to check.
        Returns:
            list[tuple[int, int]]: A list of valid destinations for this paw.
        """
        if self.retreat_status.is_activated:
            if paw.color != self.retreat_status.activator_color:
                if paw == self.retreat_status.paw_to_move:
                    moves = self.get_movements(paw)
                    if moves == []:
                        self.retreat_status.is_activated = False
                        return self.possible_movements(paw)
                    return moves
                return []
        return self.get_movements(paw)

    def get_unicolor_list(self, paws: list[Paw], color: Color) -> list[Paw]:
        """
        Get a list of paws that all have the specified color.
        Args:
            paws (list[Paw]): The list of paws to filter.
            color (Color): The color to match.
        Returns:
            list[Paw]: A list of paws matching the specified color.
        """
        return [paw for paw in paws if paw.color == color]

    def is_blended(self, paws: list[Paw], color: Color) -> bool:
        """
        Check if a set of paws is 'blended' relative to a given color.
        Args:
            paws (list[Paw]): The list of paws to check.
            color (Color): The reference color for checking.
        Returns:
            bool: True if there's at least one paw of a different color, False otherwise
        """
        return len(self.get_unicolor_list(paws, color)) < len(paws)

    def check_win(self, position: tuple[int, int]) -> bool:
        """
        Check if there's a winning condition on the given position
        Args:
            position (tuple[int, int]): The board position to check.
        Returns:
            bool: Return true if this play leads to the win.
        """
        if position in self.paws_coverage:
            if len(self.paws_coverage[position]) == 4:
                return True
        return False

    def get_valid_moves(self, color: Color) -> list[tuple[int, tuple[int, int]]]:
        """
        Retrieve all possible moves for the given color.

        Args:
            color (Color): The color to get moves from.

        Returns:
            list[tuple[int, tuple[int, int]]]: A list of (paw_index, destination).
        """
        valid_moves = []

        all_paws = [paw for paw_list in self.paws_coverage.values() for paw in paw_list]

        paws = self.get_unicolor_list(all_paws, color)

        for index, paw in enumerate(paws):
            possible_moves = self.possible_movements(paw)
            for destination in possible_moves:
                valid_moves.append((index, destination))
        return valid_moves
