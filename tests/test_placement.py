"""Tests for placement helper functions."""

import pytest

from tetris.ai.placement import find_all_placements, Placement
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.game_engine import GameEngine
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestPlacement:
    """Test placement helper functions."""

    def test_find_all_placements_returns_list(self):
        """Test that find_all_placements returns a list."""
        engine = GameEngine()
        state = engine.state

        placements = find_all_placements(state, engine)

        assert isinstance(placements, list)
        assert len(placements) > 0  # Should find at least some placements

    def test_placements_have_required_attributes(self):
        """Test that placements have required attributes."""
        engine = GameEngine()
        state = engine.state

        placements = find_all_placements(state, engine)

        if placements:
            placement = placements[0]
            assert hasattr(placement, "tetrimino")
            assert hasattr(placement, "rotation_count")
            assert hasattr(placement, "x_position")
            assert hasattr(placement, "path")
            assert isinstance(placement.path, list)
            assert placement.path[-1] == "drop"

    def test_placement_paths_end_with_drop(self):
        """Test that all placement paths end with drop."""
        engine = GameEngine()
        state = engine.state

        placements = find_all_placements(state, engine)

        for placement in placements:
            assert placement.path[-1] == "drop"
            assert len(placement.path) > 0

    def test_placement_paths_contain_valid_moves(self):
        """Test that placement paths contain only valid moves."""
        engine = GameEngine()
        state = engine.state

        placements = find_all_placements(state, engine)
        valid_moves = {"left", "right", "cw", "ccw", "drop"}

        for placement in placements:
            for move in placement.path:
                assert move in valid_moves

    def test_find_placements_with_custom_state(self):
        """Test finding placements with a custom game state."""
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

        engine = GameEngine(initial_state=state)
        placements = find_all_placements(state, engine)

        assert isinstance(placements, list)
        # Should find multiple placements for an I piece
        assert len(placements) > 0
