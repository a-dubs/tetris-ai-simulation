"""Tests for GreedyAIPlayer adapter."""

import pytest

from tetris.ai.greedy import GreedyAIPlayer
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestGreedyAIPlayer:
    """Test GreedyAIPlayer adapter class."""

    def test_create_greedy_ai(self):
        """Test creating a GreedyAIPlayer."""
        ai = GreedyAIPlayer()
        assert ai is not None
        assert ai.mps == 4
        assert ai.level == 1

    def test_get_move_sequence_returns_list(self):
        """Test that get_move_sequence returns a list of moves."""
        ai = GreedyAIPlayer(mps=4, level=1)
        playfield = [
            [" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)
        ]
        active_tet = Tetrimino("I", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
        )

        moves = ai.get_move_sequence(state)
        assert isinstance(moves, list)
        assert len(moves) > 0

    def test_get_move_sequence_ends_with_drop(self):
        """Test that move sequence ends with 'drop'."""
        ai = GreedyAIPlayer(mps=4, level=1)
        playfield = [
            [" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)
        ]
        active_tet = Tetrimino("I", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
        )

        moves = ai.get_move_sequence(state)
        assert moves[-1] == "drop"
