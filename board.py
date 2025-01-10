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
        # TODO: Check if the move is possible for this paw before.
        paw.position = destination

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        """
        Check if a position is within the board boundaries.
        Args:
          position (tuple[int, int]): The position to check
        """
        row, col = position
        return 0 <= row < 5 and 0 <= col < 5

    def find_paw_at(self, position: tuple[int, int]) -> Paw:
        """
        Find a paw at the given position.
        Args:
          position (tuple[int, int]): The position to get paw
        """
        # TODO: Check for all paws and return a list cause it can be a stack of paws
        for paw in self.paws:
            if paw.position == position:
                return paw
        return None
