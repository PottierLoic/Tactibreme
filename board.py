from paw import Paw, PawType


class Board:
    def __init__(self):
        """
        Represents the game board.
        """
        paw_order = [PawType.DONKEY, PawType.DOG, PawType.CAT, PawType.ROOSTER]

        self.paws_coverage: dict[tuple[int, int], list[Paw]] = {}
        for i, paw_type in enumerate(paw_order):
            pos = (0, i)
            paw = Paw(paw_type, "red", pos)
            self.paws_coverage[pos] = [paw]

        for i, paw_type in enumerate(paw_order):
            pos = (4, i)
            paw = Paw(paw_type, "blue", pos)
            self.paws_coverage[pos] = [paw]

    def move_paw(self, paw: Paw, destination: tuple[int, int]) -> None:
        """
        Move a paw to a new position.
        Args:
          paw (Paw): The paw to move.
          destination (Tuple[int, int]): The new position (row, col).
        """
        if not self.is_valid_position(destination):
            raise ValueError(f"Invalid destination: {destination}")
        if destination not in self.possible_movements(paw):
            raise ValueError(f"Destination {destination} is not possible for {paw.paw_type}")
        origin_pos = paw.position
        paw_at_origin = self.find_paw_at(origin_pos)
        pawns_to_move = [
            p for p in paw_at_origin if p is paw or p.paw_type.value > paw.paw_type.value
        ]
        self.paws_coverage[origin_pos] = [p for p in paw_at_origin if p not in pawns_to_move]
        if not self.paws_coverage[origin_pos]:
            del self.paws_coverage[origin_pos]
        for p in pawns_to_move:
            p.position = destination
        if destination not in self.paws_coverage:
            self.paws_coverage[destination] = []
        self.paws_coverage[destination].extend(pawns_to_move)

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        """
        Check if a position is within the board boundaries.
        Args:
          position (tuple[int, int]): The position to check
        """
        row, col = position
        return 0 <= row < 5 and 0 <= col < 5

    def find_paw_at(self, position: tuple[int, int]) -> list[Paw]:
        """
        Find a paw at the given position.
        Args:
          position (tuple[int, int]): The position to get paw
        """
        if position in self.paws_coverage:
            return self.paws_coverage[position]
        else:
            return []

    def possible_movements(self, paw: Paw) -> list[tuple[int, int]]:
        """
        Return a list of all possible movements for a Paw
        Args:
          paw (Paw): The paw to check
        """
        directions = {
            PawType.DONKEY: [(1, 0), (-1, 0), (0, 1), (0, -1)],
            PawType.DOG: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            PawType.ROOSTER: [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
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
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2),
            ]
            for dx, dy in potential_moves:
                nx, ny = paw.position[0] + dx, paw.position[1] + dy
                if self.is_valid_position((nx, ny)):
                    occupant = self.find_paw_at((nx, ny))
                    if not occupant or occupant[0].paw_type.value < paw.paw_type.value:
                        moves.append((nx, ny))

        return moves

    
    def check_win(self) -> str:
        """
        Checks if there's a winning condition and returns the color of the winner or "none" if no winner.
        """
        for paw_list in self.paws_coverage.items():
            if len(paw_list) >= 4:
                highest_paw = max(paw_list, key=lambda p: p.paw_type.value)
                return highest_paw.color
        return "none"

