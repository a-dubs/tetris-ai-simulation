"""Greedy AI player - wraps existing TetrisAI with recursive search."""

import sys
from pathlib import Path

# Import existing TetrisAI from source (temporary until full refactor)
source_dir = Path(__file__).parent.parent.parent / "source"
sys.path.insert(0, str(source_dir))

from tetris_ai import TetrisAI  # noqa: E402
from tetris.ai.base import AIPlayer
from tetris.core.game_state import GameState


class GreedyAIPlayer(AIPlayer):
    """Greedy AI player using recursive search.

    Wraps the existing TetrisAI class to provide the new AIPlayer interface.
    Uses recursive tree search with heuristic evaluation.
    """

    def __init__(self, mps: int = 4, level: int = 1, method: str = "greedy", params: dict = None):
        """Initialize greedy AI player.

        Args:
            mps: Moves per second
            level: Starting level
            method: Search method ("greedy" or "yield_to_next")
            params: Optional parameter weights for heuristic evaluation
        """
        self.mps = mps
        self.level = level
        self.method = method
        self.params = params
        self._ai: TetrisAI | None = None

    def get_move_sequence(self, state: GameState) -> list[str]:
        """Get best move sequence using recursive search.

        Args:
            state: Current game state

        Returns:
            List of moves ending with "drop"
        """
        # Update internal AI state
        self._update_ai_state(state)

        # Get best moves from existing AI
        result = self._ai.get_best_moves()
        eval_score, path = result

        # Convert path to simplified move sequence
        moves = self._ai.get_simplified_path(path)

        return moves

    def _update_ai_state(self, state: GameState) -> None:
        """Update internal TetrisAI state from GameState.

        Args:
            state: Current game state
        """
        # Create or update TetrisAI instance
        if self._ai is None:
            self._ai = TetrisAI(
                playfield=state.playfield,
                tetrimino=state.active_tetrimino,
                next_tet=state.next_tetrimino,
                mps=self.mps,
                level=self.level,
                method=self.method,
                params=self.params,
            )
        else:
            # Update existing AI's state
            self._ai.pf = state.playfield
            self._ai.tet = state.active_tetrimino
            self._ai.next_tet = state.next_tetrimino
            self._ai.lvl = state.level

    def update_state(self, state: GameState) -> None:
        """Update AI's internal state.

        Args:
            state: New game state
        """
        self._update_ai_state(state)
