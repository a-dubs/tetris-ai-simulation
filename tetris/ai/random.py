"""Random AI player - randomly selects from valid placements."""

import random

from tetris.ai.base import AIPlayer
from tetris.core.game_state import GameState
from tetris.core.game_engine import GameEngine
from tetris.ai.placement import find_all_placements


class RandomAIPlayer(AIPlayer):
    """AI player that randomly selects from valid placements.

    Finds all possible valid placements for the current piece and
    randomly selects one. This is smarter than pure random moves
    as it ensures valid placements while still being random.

    Useful for baseline testing and performance comparison.
    """

    def __init__(self):
        """Initialize random AI player."""
        self.engine: GameEngine | None = None

    def get_move_sequence(self, state: GameState) -> list[str]:
        """Return a random valid placement sequence.

        Args:
            state: Current game state

        Returns:
            List of moves ending with "drop" that leads to a random valid placement
        """
        # Create or update engine
        if self.engine is None:
            self.engine = GameEngine(initial_state=state)
        else:
            self.engine.state = state

        # Find all possible placements
        placements = find_all_placements(state, self.engine)

        if not placements:
            # Fallback: if no valid placements found, just drop
            return ["drop"]

        # Randomly select one placement
        selected_placement = random.choice(placements)

        return selected_placement.path
