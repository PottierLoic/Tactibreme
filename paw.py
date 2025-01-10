from enum import Enum


class PawType(Enum):
    DONKEY = 0
    DOG = 1
    CAT = 2
    ROOSTER = 3


class Paw:
    def __init__(
        self, paw_type: PawType, color: str, position: tuple[int, int]
    ) -> None:
        """
        Represents a game paw.
        Args:
            paw_type (PawType): Type of the paw.
            color: (str): Color of the paw ("red" or "blue").
        """
        self.paw_type = paw_type
        self.color = color
        self.position = position

    def __repr__(self) -> str:
        return f"{self.color[0].upper()}-{self.paw_type}"
