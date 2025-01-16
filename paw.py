from enum import Enum

from color import Color


class PawType(Enum):
    DONKEY = 0
    DOG = 1
    CAT = 2
    ROOSTER = 3


class Paw:
    def __init__(
        self, paw_type: PawType, color: Color, position: tuple[int, int]
    ) -> None:
        """
        Initialize a game paw.
        Args:
            paw_type (PawType): Type of the paw.
            color: (Color): Color of the paw (0/BLUE or 1/RED).
            position (tuple[int, int]): The position of the paw on the board.
        """
        self.paw_type = paw_type
        self.color = color
        self.position = position  # TODO: find a way to remove this

    def __repr__(self) -> str:
        return f"{self.color}-{self.paw_type}"
