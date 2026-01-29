"""Tests for Renderer base class."""

import pytest

from tetris.render.base import Renderer
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestRenderer:
    """Test Renderer abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Renderer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Renderer()  # type: ignore

    def test_render_is_abstract(self):
        """Test that render must be implemented."""

        class IncompleteRenderer(Renderer):
            pass

        with pytest.raises(TypeError):
            IncompleteRenderer()  # type: ignore

    def test_concrete_implementation(self):
        """Test that a concrete implementation works."""

        class ConcreteRenderer(Renderer):
            def render(self, state: GameState) -> None:
                pass

        renderer = ConcreteRenderer()
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
        renderer.render(state)
