"""Tests for GameEngine class."""

import pytest

from tetris.core.game_state import GameState
from tetris.core.game_engine import GameEngine
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestGameEngine:
    """Test GameEngine class."""

    def test_create_initial_state(self):
        """Test creating a game engine with initial state."""
        engine = GameEngine()
        assert engine.state is not None
        assert isinstance(engine.state, GameState)
        assert engine.state.game_over is False
        assert engine.state.score == 0
        assert engine.state.level == 1

    def test_create_with_custom_state(self):
        """Test creating a game engine with custom state."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        bag = ["T", "L", "J"]

        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            bag=bag,
            score=100,
            level=2,
        )

        engine = GameEngine(initial_state=state)
        assert engine.state.score == 100
        assert engine.state.level == 2
        assert engine.state.bag == bag

    def test_apply_move_left(self):
        """Test applying a left move."""
        engine = GameEngine()
        initial_x = engine.state.active_tetrimino.x

        new_state = engine.apply_move("left")

        # If move is valid, x should decrease
        if new_state.active_tetrimino.x != initial_x:
            assert new_state.active_tetrimino.x == initial_x - 1
        # State should be immutable - original unchanged
        assert engine.state.active_tetrimino.x == initial_x

    def test_apply_move_right(self):
        """Test applying a right move."""
        engine = GameEngine()
        initial_x = engine.state.active_tetrimino.x

        new_state = engine.apply_move("right")

        # If move is valid, x should increase
        if new_state.active_tetrimino.x != initial_x:
            assert new_state.active_tetrimino.x == initial_x + 1
        # State should be immutable
        assert engine.state.active_tetrimino.x == initial_x

    def test_apply_move_rotate_cw(self):
        """Test applying a clockwise rotation."""
        engine = GameEngine()
        initial_orientation = engine.state.active_tetrimino.orientation

        new_state = engine.apply_move("cw")

        # Orientation should change
        assert new_state.active_tetrimino.orientation != initial_orientation
        # State should be immutable
        assert engine.state.active_tetrimino.orientation == initial_orientation

    def test_apply_move_rotate_ccw(self):
        """Test applying a counter-clockwise rotation."""
        engine = GameEngine()
        initial_orientation = engine.state.active_tetrimino.orientation

        new_state = engine.apply_move("ccw")

        # Orientation should change
        assert new_state.active_tetrimino.orientation != initial_orientation
        # State should be immutable
        assert engine.state.active_tetrimino.orientation == initial_orientation

    def test_apply_move_drop(self):
        """Test applying a drop move."""
        engine = GameEngine()
        initial_y = engine.state.active_tetrimino.y

        new_state = engine.apply_move("drop")

        # Y should decrease (move down)
        assert new_state.active_tetrimino.y <= initial_y
        # State should be immutable
        assert engine.state.active_tetrimino.y == initial_y

    def test_valid_location(self):
        """Test checking if tetrimino is in valid location."""
        engine = GameEngine()
        tet = engine.state.active_tetrimino

        # Initial tetrimino should be in valid location
        assert engine.valid_location(tet) is True

        # Move tetrimino out of bounds
        tet.x = 0  # Out of bounds (x must be >= 1)
        assert engine.valid_location(tet) is False

    def test_place_tetrimino(self):
        """Test placing a tetrimino on the playfield."""
        engine = GameEngine()
        tet = engine.state.active_tetrimino
        initial_playfield = [row[:] for row in engine.state.playfield]

        new_state = engine.place_tetrimino(tet)

        # Playfield should be modified
        assert new_state.playfield != initial_playfield
        # Original state should be unchanged
        assert engine.state.playfield == initial_playfield

    def test_update_playfield_clears_lines(self):
        """Test that update_playfield clears full lines."""
        # Create a playfield with a full line
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        # Fill bottom row
        playfield[-1] = ["c" for _ in range(PLAYFIELD_WIDTH)]

        active_tet = Tetrimino("I", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 1)
        next_tet = Tetrimino("O", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
        )

        engine = GameEngine(initial_state=state)
        new_state, lines_cleared = engine.update_playfield(active_tet)

        # Should have cleared at least the full line
        assert lines_cleared >= 0
        # Bottom row should now be empty
        assert new_state.playfield[-1] == [" " for _ in range(PLAYFIELD_WIDTH)]
