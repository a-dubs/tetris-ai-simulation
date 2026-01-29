"""Random AI player - makes random valid moves."""

import random

from tetris.ai.base import AIPlayer
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH


class RandomAIPlayer(AIPlayer):
    """AI player that makes random moves.

    Generates a random sequence of moves ending with 'drop'.
    Useful for baseline testing and performance comparison.
    """

    def __init__(self, max_moves: int = 20):
        """Initialize random AI player.

        Args:
            max_moves: Maximum number of moves before dropping (default: 20)
        """
        self.max_moves = max_moves

    def get_move_sequence(self, state: GameState) -> list[str]:
        """Return a random sequence of moves.

        Args:
            state: Current game state

        Returns:
            List of random moves ending with "drop"
        """
        moves = []
        num_moves = random.randint(1, self.max_moves)

        # Generate random moves (excluding drop)
        possible_moves = ["left", "right", "cw", "ccw"]
        for _ in range(num_moves):
            moves.append(random.choice(possible_moves))

        # Always end with drop
        moves.append("drop")

        return moves
