from paw import Paw, PawType


class Board:
    def __init__(self):
        """
        Represents the game board.
        """
        paw_order = [PawType.DONKEY, PawType.DOG, PawType.CAT, PawType.ROOSTER]

        self.paws: List[Paws] = []
        for i, paw_type in enumerate(paw_order):
            self.paws.append(Paw(paw_type, "red", (0, i)))

        for i, paw_type in enumerate(paw_order):
            self.paws.append(Paw(paw_type, "blue", (4, i)))

    def move_paw(self, paw: Paw, destination: tuple[int, int]) -> None:
        """
        Move a paw to a new position.
        Args:
          paw (Paw): The paw to move.
          destination (Tuple[int, int]): The new position (row, col).
        """
        if not self.is_valid_position(destination):
            raise ValueError(f"Invalid destination: {destination}")
        paw.position = destination

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
        # TODO: Check for all paws and return a list cause it can be a stack of paws
        paws = []
        for paw in self.paws:
            if paw.position == position:
                paws.append(paw)
        return paws

    def possible_movements(self, paw: Paw) -> list[tuple[int, int]]:
        """
        Return a list of all possible movements for a Paw
        Args:
          paw (Paw): The paw to check
        """
        possible_movements = []
        match paw.paw_type:
            case PawType.DONKEY:
                for i in range(5):
                    if i != paw.position[0]:
                        possible_movements.append((i, paw.position[1]))
                    if i != paw.position[1]:
                        possible_movements.append((paw.position[0], i))
            case PawType.DOG:
                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions:
                    step = 1
                    while True:
                        nx = paw.position[0] + step * dx
                        ny = paw.position[1] + step * dy
                        if 0 <= nx < 5 and 0 <= ny < 5:
                            possible_movements.append((nx, ny))
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
                        possible_movements.append((nx, ny))
            case PawType.ROOSTER:
                for i in range(5):
                    if i != paw.position[0]:
                        possible_movements.append((i, paw.position[1]))
                    if i != paw.position[1]:
                        possible_movements.append((paw.position[0], i))

                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for dx, dy in directions:
                    step = 1
                    while True:
                        nx = paw.position[0] + step * dx
                        ny = paw.position[1] + step * dy
                        if 0 <= nx < 5 and 0 <= ny < 5:
                            possible_movements.append((nx, ny))
                            step += 1
                        else:
                            break
        return possible_movements
