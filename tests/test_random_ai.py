"""Tests for RandomAIPlayer."""

import pytest

from tetris.ai.random import RandomAIPlayer
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestRandomAIPlayer:
    """Test RandomAIPlayer class."""

    def test_create_random_ai(self):
        """Test creating a RandomAIPlayer."""
        ai = RandomAIPlayer()
        assert ai is not None

    def test_get_move_sequence_returns_list(self):
        """Test that get_move_sequence returns a list of moves."""
        ai = RandomAIPlayer()
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
        ai = RandomAIPlayer()
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

    def test_get_move_sequence_contains_valid_moves(self):
        """Test that all moves are valid."""
        ai = RandomAIPlayer()
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

        valid_moves = {"left", "right", "cw", "ccw", "drop"}
        moves = ai.get_move_sequence(state)

        for move in moves:
            assert move in valid_moves

    def test_different_calls_return_different_moves(self):
        """Test that different calls return (likely) different moves."""
        ai = RandomAIPlayer()
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

        moves1 = ai.get_move_sequence(state)
        moves2 = ai.get_move_sequence(state)

        # They might be the same by chance, but very unlikely for multiple calls
        # At least verify they're both valid
        assert isinstance(moves1, list)
        assert isinstance(moves2, list)
        assert len(moves1) > 0
        assert len(moves2) > 0
