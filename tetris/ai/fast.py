"""Fast AI player - wraps existing FastTetrisAI with direct placement evaluation."""

import sys
from pathlib import Path

# Import existing FastTetrisAI from source (temporary until full refactor)
source_dir = Path(__file__).parent.parent.parent / "source"
sys.path.insert(0, str(source_dir))

from tetris_ai_fast import FastTetrisAI  # noqa: E402
from tetris.ai.base import AIPlayer
from tetris.core.game_state import GameState


class FastAIPlayer(AIPlayer):
    """Fast AI player using direct placement evaluation.

    Wraps the existing FastTetrisAI class to provide the new AIPlayer interface.
    Evaluates all possible placements directly without recursive search.
    Much faster than GreedyAIPlayer but potentially less optimal.
    """

    def __init__(self, mps: int = 4, level: int = 1, method: str = "greedy", params: dict = None):
        """Initialize fast AI player.

        Args:
            mps: Moves per second
            level: Starting level
            method: Evaluation method ("greedy")
            params: Optional parameter weights for heuristic evaluation
        """
        self.mps = mps
        self.level = level
        self.method = method
        self.params = params
        self._ai: FastTetrisAI | None = None

    def get_move_sequence(self, state: GameState) -> list[str]:
        """Get best move sequence using fast direct evaluation.

        Args:
            state: Current game state

        Returns:
            List of moves ending with "drop"
        """
        # Update internal AI state
        self._update_ai_state(state)

        # Get best moves from fast AI
        result = self._ai.get_best_moves_fast()
        eval_score, path = result

        # Convert path to simplified move sequence
        moves = self._ai.get_simplified_path(path)

        return moves

    def _update_ai_state(self, state: GameState) -> None:
        """Update internal FastTetrisAI state from GameState.

        Args:
            state: Current game state
        """
        # Create or update FastTetrisAI instance
        if self._ai is None:
            self._ai = FastTetrisAI(
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
