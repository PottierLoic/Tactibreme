from color import Color
import random
from paw import Paw, PawType


class GameFinished(Exception):
    def __init__(self, winner_color):
        self.winner_color = winner_color


class Board:
    def __init__(self) -> None:
        random.seed(42)
        paws = [PawType.DONKEY, PawType.DOG, PawType.CAT, PawType.ROOSTER, -1]
        self.paws_coverage: dict[tuple[int, int], list[Paw]] = {}
        random.shuffle(paws)
        for i, paw_type in enumerate(paws):
            if paw_type == -1:
                continue
            pos = (0, i)
            paw = Paw(paw_type, Color.RED, pos)
            self.paws_coverage[pos] = [paw]
        random.shuffle(paws)
        for i, paw_type in enumerate(paws):
            if paw_type == -1:
                continue
            pos = (4, i)
            paw = Paw(paw_type, Color.BLUE, pos)
            self.paws_coverage[pos] = [paw]
        self.is_retreat = False

    def move_paw(self, paw: Paw, destination: tuple[int, int]) -> int:
        """
        Move a paw to a new position.
        Args:
            paw (Paw): The paw to move.
            destination (tuple[int, int]): The new position (row, col).
        Returns:
            int: 1 if this move leads a retreat, 0 otherwise.
        """
        if not self.is_valid_position(destination):
            raise ValueError(f"Invalid destination: {destination}")
        if destination not in self.possible_movements(paw):
            raise ValueError(
                f"Destination {destination} is not possible for {paw.paw_type}"
            )
        origin_pos = paw.position
        paw_at_origin = self.find_paw_at(origin_pos)
        pawns_to_move = [
            p
            for p in paw_at_origin
            if p is paw or p.paw_type.value > paw.paw_type.value
        ]
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
            return 1
        return 0

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

    def possible_movements(self, paw: Paw) -> list[tuple[int, int]]:
        """
        Return a list of all possible movements for a given paw.
        Args:
            paw (Paw): The paw to check.
        Returns:
            list[tuple[int, int]]: A list of valid destinations for this paw.
        """
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
                        if occupant[0].paw_type.value < paw.paw_type.value:
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
                    if not occupant or occupant[0].paw_type.value < paw.paw_type.value:
                        moves.append((nx, ny))
        return moves

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

    def check_win(self, position: tuple[int, int]) -> Color | int:
        """
        Check if there's a winning condition on the given position
        Args:
            position (tuple[int, int]): The board position to check.
        Returns:
            Color | int: The color of the winner if there's a winning condition,
                         otherwise -1 if no winning condition is met.
        """
        if position in self.paws_coverage:
            if len(self.paws_coverage[position]) == 4:
                print(f"situation gagnante, {self.paws_coverage[position]}")
                return self.paws_coverage[position][-1].color
        return -1

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
