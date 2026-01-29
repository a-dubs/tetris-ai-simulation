"""Tests for AIPlayer base class."""

import pytest

from tetris.ai.base import AIPlayer
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestAIPlayer:
    """Test AIPlayer abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that AIPlayer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AIPlayer()  # type: ignore

    def test_get_move_sequence_is_abstract(self):
        """Test that get_move_sequence must be implemented."""

        class IncompleteAIPlayer(AIPlayer):
            pass

        with pytest.raises(TypeError):
            IncompleteAIPlayer()  # type: ignore

    def test_concrete_implementation(self):
        """Test that a concrete implementation works."""

        class ConcreteAIPlayer(AIPlayer):
            def get_move_sequence(self, state: GameState) -> list[str]:
                return ["left", "left", "drop"]

        ai = ConcreteAIPlayer()
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
        assert moves == ["left", "left", "drop"]
        assert isinstance(moves, list)
        assert all(isinstance(move, str) for move in moves)

    def test_update_state_is_optional(self):
        """Test that update_state is optional and has default implementation."""

        class ConcreteAIPlayer(AIPlayer):
            def get_move_sequence(self, state: GameState) -> list[str]:
                return ["drop"]

        ai = ConcreteAIPlayer()
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

        # Should not raise an error
        ai.update_state(state)
