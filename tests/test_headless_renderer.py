"""Tests for HeadlessRenderer."""

import pytest

from tetris.render.headless import HeadlessRenderer
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestHeadlessRenderer:
    """Test HeadlessRenderer class."""

    def test_create_headless_renderer(self):
        """Test creating a HeadlessRenderer."""
        renderer = HeadlessRenderer()
        assert renderer is not None

    def test_render_does_nothing(self):
        """Test that render method does nothing (no-op)."""
        renderer = HeadlessRenderer()
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

        # Should not raise an error and do nothing
        renderer.render(state)

    def test_render_with_game_over_state(self):
        """Test that render works with game over state."""
        renderer = HeadlessRenderer()
        playfield = [
            [" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)
        ]
        active_tet = Tetrimino("I", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)
        next_tet = Tetrimino("O", x=5, y=TOTAL_PLAYFIELD_HEIGHT - 3)

        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            game_over=True,
        )

        # Should not raise an error
        renderer.render(state)
