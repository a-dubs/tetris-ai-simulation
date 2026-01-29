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

    def test_piece_drops_to_bottom_in_empty_column(self):
        """Test that pieces drop to bottom in an empty column.
        
        Scenario: Create a playfield with filled rows, leaving one column empty.
        A vertical I-piece in that empty column should still drop to the bottom
        because adjacent columns do not provide support.
        """
        from tetris.core.constants import spawn_y_for_size
        
        # Create playfield with bottom 4 rows filled except column 0
        playfield = [
            [" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)
        ]
        # Fill rows 20-23 (0-indexed) with blocks in columns 1-9, leaving column 0 empty
        for row_idx in range(TOTAL_PLAYFIELD_HEIGHT - 4, TOTAL_PLAYFIELD_HEIGHT):
            playfield[row_idx] = [" "] + ["c"] * (PLAYFIELD_WIDTH - 1)
        
        state = GameState(
            playfield=playfield,
            active_tetrimino=Tetrimino("I", x=PLAYFIELD_WIDTH // 2 - 1, y=spawn_y_for_size(4)),
            next_tetrimino=Tetrimino("O", x=5, y=1),
        )
        
        engine = GameEngine(initial_state=state)
        placements = find_all_placements(state, engine)
        
        assert len(placements) > 0, "Should find at least one placement"
        
        # Find a vertical I placement that occupies playfield column 0
        column_zero_vertical = None
        for placement in placements:
            tet = placement.tetrimino
            block_cells = [
                (row, col)
                for row in range(tet.size)
                for col in range(tet.size)
                if tet.minos[row][col] != " "
            ]
            if not block_cells:
                continue
            cols = {col for _, col in block_cells}
            is_vertical = len(cols) == 1
            if not is_vertical:
                continue
            block_col = next(iter(cols))
            playfield_col = tet.x + block_col - 1  # 0-indexed
            if playfield_col == 0:
                column_zero_vertical = placement
                break
        
        assert column_zero_vertical is not None, \
            "Should find a vertical I placement in column 0"
        
        # The piece should drop to the bottom (y=1) in an empty column
        assert column_zero_vertical.tetrimino.y == 1, \
            f"Piece should drop to y=1, but got y={column_zero_vertical.tetrimino.y}"

    def test_piece_stops_at_bottom_when_no_blocks_below(self):
        """Test that pieces stop at y=1 when there are no blocks below.
        
        Scenario: Empty playfield. Piece should drop to the bottom (y=1).
        """
        from tetris.core.constants import spawn_y_for_size
        
        playfield = [
            [" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)
        ]
        
        state = GameState(
            playfield=playfield,
            active_tetrimino=Tetrimino("I", x=PLAYFIELD_WIDTH // 2 - 1, y=spawn_y_for_size(4)),
            next_tetrimino=Tetrimino("O", x=5, y=1),
        )
        
        engine = GameEngine(initial_state=state)
        placements = find_all_placements(state, engine)
        
        assert len(placements) > 0, "Should find placements"
        
        # All placements should have y >= 1 (bottom of playfield)
        for placement in placements:
            assert placement.tetrimino.y >= 1, \
                f"Placement should have y >= 1, but got y={placement.tetrimino.y}"
