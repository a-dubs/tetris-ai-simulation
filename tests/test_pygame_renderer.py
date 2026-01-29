"""Tests for PygameRenderer."""

import pytest
import os

# Set test environment before importing
os.environ["PYTEST_CURRENT_TEST"] = "test"

from tetris.render.pygame_gui import PygameRenderer
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestPygameRenderer:
    """Test PygameRenderer class."""

    def test_create_pygame_renderer_in_test_mode(self):
        """Test that PygameRenderer doesn't initialize pygame in test mode."""
        renderer = PygameRenderer()
        assert renderer._test_mode is True
        assert renderer.screen is None
        assert renderer.running is True

    def test_render_does_nothing_in_test_mode(self):
        """Test that render does nothing in test mode."""
        renderer = PygameRenderer()
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

    def test_cleanup_in_test_mode(self):
        """Test that cleanup does nothing in test mode."""
        renderer = PygameRenderer()
        # Should not raise an error
        renderer.cleanup()
