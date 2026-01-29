"""Abstract base class for AI players."""

from abc import ABC, abstractmethod
from typing import List

from tetris.core.game_state import GameState


class AIPlayer(ABC):
    """Abstract base class for AI players.

    All AI implementations must inherit from this class and implement
    get_move_sequence() to return a list of moves to execute.
    """

    @abstractmethod
    def get_move_sequence(self, state: GameState) -> List[str]:
        """Return sequence of moves to make.

        Args:
            state: Current game state

        Returns:
            List of moves like ["left", "left", "cw", "drop"]
            Moves can be: "left", "right", "cw", "ccw", "drop"
        """
        pass

    def update_state(self, state: GameState) -> None:
        """Update AI's internal state (optional).

        Called when game state changes. Default implementation does nothing.
        Subclasses can override to maintain internal state or cache.

        Args:
            state: New game state
        """
        pass
