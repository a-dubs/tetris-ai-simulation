"""Tetrimino (Tetris piece) class."""

import random


class Tetrimino:
    """Represents a Tetris piece (tetrimino)."""

    shapes = {
        "O": [["y", "y"], ["y", "y"]],
        "I": [
            [" ", " ", " ", " "],
            [" ", " ", " ", " "],
            ["c", "c", "c", "c"],
            [" ", " ", " ", " "],
        ],
        "T": [[" ", " ", " "], ["p", "p", "p"], [" ", "p", " "]],
        "L": [[" ", " ", " "], ["o", "o", "o"], [" ", " ", "o"]],
        "J": [[" ", " ", " "], ["b", "b", "b"], ["b", " ", " "]],
        "S": [[" ", " ", " "], ["g", "g", " "], [" ", "g", "g"]],
        "Z": [[" ", " ", " "], [" ", "r", "r"], ["r", "r", " "]],
    }

    def __init__(self, shape=None, x=None, y=None, orientation="N", ldml=15):
        """Initialize a tetrimino.

        Args:
            shape: Shape name ('O', 'I', 'T', 'L', 'J', 'S', 'Z') or None for random
            x: X coordinate of bottom left corner
            y: Y coordinate of bottom left corner
            orientation: One of ('N', 'E', 'S', 'W') representing orientation
            ldml: Lock down movements left - number of moves after landing before locking
        """
        self.minos = self.get_minos(self.rand_shape() if shape is None else shape)
        self.ldml = ldml  # lock down movements left
        self.landed = False
        self.landed_y = -1
        self.ld_timer = -1
        self.lowest_y = y
        # x coordinate of bottom left corner of tetrimino - D: [1, playfield width]
        self.x = x
        # y coordinate of bottom left corner of tetrimino - D: [1, playfield height]
        self.y = y
        # one of ('N', 'E', 'S', W') representing the current tetrimino's orientation
        self.orientation = orientation
        self.size = len(self.minos)

    @staticmethod
    def get_minos(shape):
        """Get the minos (block pattern) for a given shape.

        Args:
            shape: Shape name ('O', 'I', 'T', 'L', 'J', 'S', 'Z')

        Returns:
            2D list representing the tetrimino pattern
        """
        return Tetrimino.shapes[shape.upper()]

    @staticmethod
    def rand_shape():
        """Return a random shape name.

        Returns:
            Random shape name from available shapes
        """
        return random.choice(list(Tetrimino.shapes.keys()))

    @staticmethod
    def make_bag(repetitions):
        """Create a bag of tetrimino shapes.

        Creates a bag containing each shape 'repetitions' times, then shuffles it.

        Args:
            repetitions: Number of times each shape appears in the bag

        Returns:
            Shuffled list of shape names
        """
        bag = [
            list(Tetrimino.shapes.keys())[i % len(Tetrimino.shapes.keys())]
            for i in range(0, len(Tetrimino.shapes.keys()) * repetitions)
        ]
        random.shuffle(bag)
        return bag

    def reset_LDML(self):
        """Reset lock down movements left to default value (15)."""
        self.ldml = 15

    def log(self):
        """Print tetrimino information for debugging."""
        print("Facing " + self.orientation)
        print("Coords: " + self.coords())
        print("----")
        for y in range(-1, -len(self.minos) - 1, -1):
            print("".join(self.minos[y]))
        print("----")

    def coords(self):
        """Get coordinates as a string.

        Returns:
            String representation of coordinates like "(x,y)"
        """
        return "(" + str(self.x) + "," + str(self.y) + ")"
