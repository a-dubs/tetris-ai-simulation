"""Tests for GameState dataclass."""

import pytest
from dataclasses import replace

# We'll need to import Tetrimino once we refactor it
# For now, we'll use a mock or import from source
import sys
from pathlib import Path

# Add source directory to path temporarily
source_dir = Path(__file__).parent.parent / "source"
sys.path.insert(0, str(source_dir))

from tetrimino import Tetrimino
from tetris.core.game_state import GameState
from tetris.core.constants import TOTAL_PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH


class TestGameState:
    """Test GameState dataclass."""

    def test_create_initial_state(self):
        """Test creating an initial game state."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        bag = ["T", "L", "J", "S", "Z"]

        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            bag=bag,
        )

        assert state.playfield == playfield
        assert state.active_tetrimino == active_tet
        assert state.next_tetrimino == next_tet
        assert state.bag == bag
        assert state.score == 0
        assert state.level == 1
        assert state.lines_cleared == 0
        assert state.game_over is False
        assert state.time_elapsed == 0.0

    def test_state_is_immutable(self):
        """Test that GameState is immutable (frozen dataclass)."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
        )

        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            state.score = 100

    def test_with_playfield(self):
        """Test creating a new state with updated playfield."""
        playfield1 = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        playfield2 = [["c" for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield1,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            score=100,
        )

        state2 = state1.with_playfield(playfield2)

        assert state1.playfield == playfield1
        assert state2.playfield == playfield2
        assert state2.score == 100  # Other fields unchanged
        assert state1 is not state2  # New instance

    def test_with_active_tetrimino(self):
        """Test creating a new state with updated active tetrimino."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet1 = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        active_tet2 = Tetrimino("T", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet1,
            next_tetrimino=next_tet,
        )

        state2 = state1.with_active_tetrimino(active_tet2)

        assert state1.active_tetrimino == active_tet1
        assert state2.active_tetrimino == active_tet2
        assert state1 is not state2

    def test_with_next_tetrimino(self):
        """Test creating a new state with updated next tetrimino."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet1 = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet2 = Tetrimino("T", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet1,
        )

        state2 = state1.with_next_tetrimino(next_tet2)

        assert state1.next_tetrimino == next_tet1
        assert state2.next_tetrimino == next_tet2
        assert state1 is not state2

    def test_with_score(self):
        """Test creating a new state with updated score."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            score=100,
        )

        state2 = state1.with_score(200)

        assert state1.score == 100
        assert state2.score == 200
        assert state1 is not state2

    def test_with_level(self):
        """Test creating a new state with updated level."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            level=1,
        )

        state2 = state1.with_level(2)

        assert state1.level == 1
        assert state2.level == 2
        assert state1 is not state2

    def test_with_lines_cleared(self):
        """Test creating a new state with updated lines cleared."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            lines_cleared=5,
        )

        state2 = state1.with_lines_cleared(10)

        assert state1.lines_cleared == 5
        assert state2.lines_cleared == 10
        assert state1 is not state2

    def test_with_game_over(self):
        """Test creating a new state with updated game_over flag."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            game_over=False,
        )

        state2 = state1.with_game_over(True)

        assert state1.game_over is False
        assert state2.game_over is True
        assert state1 is not state2

    def test_with_time_elapsed(self):
        """Test creating a new state with updated time elapsed."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            time_elapsed=10.5,
        )

        state2 = state1.with_time_elapsed(20.0)

        assert state1.time_elapsed == 10.5
        assert state2.time_elapsed == 20.0
        assert state1 is not state2

    def test_with_bag(self):
        """Test creating a new state with updated bag."""
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        active_tet = Tetrimino("I", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=4, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        bag1 = ["T", "L"]
        bag2 = ["J", "S", "Z"]

        state1 = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            bag=bag1,
        )

        state2 = state1.with_bag(bag2)

        assert state1.bag == bag1
        assert state2.bag == bag2
        assert state1 is not state2
