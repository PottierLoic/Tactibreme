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
        raw_positions = []
        match paw.paw_type:
            case PawType.DONKEY:
                for i in range(5):
                    if i != paw.position[0]:
                        raw_positions.append((i, paw.position[1]))
                    if i != paw.position[1]:
                        raw_positions.append((paw.position[0], i))

            case PawType.DOG:
                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions:
                    step = 1
                    while True:
                        nx = paw.position[0] + step * dx
                        ny = paw.position[1] + step * dy
                        if 0 <= nx < 5 and 0 <= ny < 5:
                            raw_positions.append((nx, ny))
                            step += 1
                        else:
                            break

            case PawType.CAT:
                moves = [
                    (2, 1), (2, -1), (-2, 1), (-2, -1),
                    (1, 2), (1, -2), (-1, 2), (-1, -2),
                ]
                for dx, dy in moves:
                    nx, ny = paw.position[0] + dx, paw.position[1] + dy
                    if 0 <= nx < 5 and 0 <= ny < 5:
                        raw_positions.append((nx, ny))

            case PawType.ROOSTER:
                for i in range(5):
                    if i != paw.position[0]:
                        raw_positions.append((i, paw.position[1]))
                    if i != paw.position[1]:
                        raw_positions.append((paw.position[0], i))

                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions:
                    step = 1
                    while True:
                        nx = paw.position[0] + step * dx
                        ny = paw.position[1] + step * dy
                        if 0 <= nx < 5 and 0 <= ny < 5:
                            raw_positions.append((nx, ny))
                            step += 1
                        else:
                            break

        filtered_positions = []
        for pos in raw_positions:
            occupant = self.find_paw_at(pos)
            if not occupant:
                filtered_positions.append(pos)
            else:
                occupant.sort(key=lambda p: p.paw_type.value, reverse=True)
                if occupant[0].paw_type.value < paw.paw_type.value:
                    filtered_positions.append(pos)

        return filtered_positions

    
    def check_win(self) -> str:
        """
        Checks if there's a winning condition and returns the color of the winner or "none" if no winner.
        """
        for paw_list in self.paws_coverage.items():
            if len(paw_list) >= 4:
                highest_paw = max(paw_list, key=lambda p: p.paw_type.value)
                return highest_paw.color
        return "none"

