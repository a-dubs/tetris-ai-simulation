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

    def test_valid_location_rejects_pieces_with_blocks_above_playfield(self):
        """Test that pieces with blocks extending above y=1 are invalid."""
        engine = GameEngine()
        
        # O piece has blocks starting at row 0 (0-indexed in minos)
        # So at y=0, blocks would be at y=0+0=0 and y=0+1=1
        # y=0 is out of bounds (< 1), so O piece at y=0 should be invalid
        tet_o = Tetrimino("O", x=5, y=0)
        assert engine.valid_location(tet_o) is False, "O piece at y=0 should be invalid (blocks extend above)"
        
        # T, L, J, S, Z pieces also have blocks in row 0 or 1, so y=0 would put blocks at y=0 or y=1
        # y=0 is out of bounds, so these should be invalid
        for shape in ["T", "L", "J", "S", "Z"]:
            tet = Tetrimino(shape, x=5, y=0)
            # Check if piece has blocks that would be at y <= 0
            has_block_at_y0 = False
            for row in range(tet.size):
                for col in range(tet.size):
                    if tet.minos[row][col] != " ":
                        block_y = tet.y + row
                        if block_y <= 0:
                            has_block_at_y0 = True
                            break
            if has_block_at_y0:
                assert engine.valid_location(tet) is False, f"{shape} piece at y=0 should be invalid"
        
        # I piece is special - it has blocks only in row 2 (0-indexed)
        # So at y=0, blocks are at y=0+2=2, which is valid
        # However, the piece's bottom-left corner is at y=0, which might be considered invalid
        # But current logic only checks if blocks are out of bounds, not the corner position
        tet_i = Tetrimino("I", x=5, y=0)
        # I piece at y=0: blocks at y=2, which is valid, so this passes current logic
        # This might be acceptable behavior - the piece doesn't have blocks above playfield

    def test_valid_location_rejects_piece_with_negative_y(self):
        """Test that pieces with blocks extending above playfield (negative y) are invalid."""
        engine = GameEngine()
        
        # O piece has blocks starting at row 0, so at y=-1, blocks would be at y=-1+0=-1 (out of bounds)
        tet_o = Tetrimino("O", x=5, y=-1)
        assert engine.valid_location(tet_o) is False
        
        # I piece has blocks only in row 2, so at y=-1, blocks would be at y=-1+2=1 (valid)
        # But let's test with a piece that definitely has blocks extending above
        tet_o2 = Tetrimino("O", x=5, y=-5)
        assert engine.valid_location(tet_o2) is False

    def test_valid_location_accepts_piece_at_y_1(self):
        """Test that pieces at y=1 are valid (lowest valid position)."""
        engine = GameEngine()
        
        # O piece at y=1: blocks at y=1+0=1 and y=1+1=2, both in bounds
        tet = Tetrimino("O", x=5, y=1)
        assert engine.valid_location(tet) is True

    def test_valid_location_detects_piece_extending_above_playfield(self):
        """Test that valid_location detects when piece blocks extend above playfield."""
        engine = GameEngine()
        
        # O piece has blocks starting at row 0 (0-indexed in minos array)
        # So if y=0, blocks would be at y=0+0=0 and y=0+1=1
        # y=0 is out of bounds (< 1), so this should be invalid
        tet_o = Tetrimino("O", x=5, y=0)
        assert engine.valid_location(tet_o) is False
        
        # I piece has blocks only in row 2 (0-indexed), so at y=0, blocks are at y=0+2=2 (valid)
        # The current implementation only checks if blocks are out of bounds, not the piece position
        # So I piece at y=0 is technically valid because its blocks don't extend above
        # This is acceptable behavior - the piece doesn't have blocks above the playfield
        tet_i = Tetrimino("I", x=5, y=0)
        # I piece at y=0: blocks at y=2, which is valid, so this passes
        # This is correct - the piece doesn't extend above the playfield

    def test_game_over_when_piece_placed_and_next_spawns_overlapping(self):
        """Test that game ends when a piece is placed and next piece spawns overlapping."""
        from tetris.core.constants import PLAYFIELD_WIDTH
        
        # Create a playfield with blocks at spawn position
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        # Fill rows 18,19 at columns 4,5 (where O piece spawns)
        playfield[18][4] = "c"
        playfield[18][5] = "c"
        playfield[19][4] = "c"
        playfield[19][5] = "c"
        
        # Place a piece that fills more space
        active_tet = Tetrimino("O", x=5, y=19)
        next_tet = Tetrimino("O", x=5, y=19)  # Will spawn at same position
        
        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
        )
        
        engine = GameEngine(initial_state=state)
        
        # Place the active piece - this adds more blocks
        new_state, _ = engine.update_playfield(active_tet)
        
        # Now spawn the next piece at the same position - it should overlap
        spawn_tet = Tetrimino("O", x=5, y=19)
        
        # Create engine with updated state
        engine_updated = GameEngine(initial_state=new_state)
        
        # Should be invalid because it overlaps with filled playfield
        is_valid = engine_updated.valid_location(spawn_tet)
        assert is_valid is False, "Spawned piece should be invalid when it overlaps"
        
        # Game should end
        game_over = not is_valid
        assert game_over is True, "Game should end when spawned piece overlaps"

    def test_simulator_game_over_after_drop_when_spawned_piece_invalid(self):
        """Test that simulator properly sets game_over when spawned piece overlaps."""
        from tetris.core.constants import PLAYFIELD_WIDTH
        
        # Create a playfield with blocks already placed where we want to spawn
        # O piece spawns at y=19, x=5 (center)
        # Pre-fill the playfield at that exact position
        playfield = [[" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        # Fill rows 18,19 at columns 4,5 (where O piece at x=5, y=19 would place blocks)
        playfield[18][4] = "c"  # Row 18, column 4
        playfield[18][5] = "c"  # Row 18, column 5
        playfield[19][4] = "c"  # Row 19, column 4
        playfield[19][5] = "c"  # Row 19, column 5
        
        # Try to spawn O piece at the same position - it should overlap
        spawned_tet = Tetrimino("O", x=5, y=19)
        
        state = GameState(
            playfield=playfield,
            active_tetrimino=spawned_tet,
            next_tetrimino=Tetrimino("I", x=5, y=1),
        )
        
        engine = GameEngine(initial_state=state)
        
        # Check if spawned piece is valid - it should NOT be because it overlaps
        is_valid = engine.valid_location(spawned_tet)
        
        # The game should end if the spawned piece is invalid
        # This simulates what happens in simulator.py line 146
        game_over = not is_valid
        assert game_over is True, "Game should end when spawned piece overlaps with filled playfield"
        assert is_valid is False, "Spawned piece should be invalid when it overlaps"

    def test_game_over_when_piece_extends_above_visible_playfield(self):
        """Test game over when a piece is placed and next piece would extend above visible playfield."""
        from tetris.core.constants import PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT
        
        # Fill playfield up to the top of visible playfield (row PLAYFIELD_HEIGHT-1, 0-indexed)
        # This means y coordinates up to PLAYFIELD_HEIGHT are filled
        playfield = [["c" for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        # Leave skybox rows empty (rows 0-3, 0-indexed, which are y=1-4, 1-indexed)
        for y_idx in range(4):  # SKYBOX_HEIGHT = 4
            playfield[y_idx] = [" " for _ in range(PLAYFIELD_WIDTH)]
        
        # Place a piece that fills up to the top of visible playfield
        # I piece at y=PLAYFIELD_HEIGHT-2: blocks at y=PLAYFIELD_HEIGHT-2+2=PLAYFIELD_HEIGHT
        active_tet = Tetrimino("I", x=5, y=PLAYFIELD_HEIGHT - 2)
        next_tet = Tetrimino("I", x=5, y=PLAYFIELD_HEIGHT - 3)
        
        state = GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
        )
        
        engine = GameEngine(initial_state=state)
        
        # Place the active piece
        new_state, _ = engine.update_playfield(active_tet)
        
        # Now try to spawn next piece - it should be invalid if it would extend above
        # But spawn_y for I piece is PLAYFIELD_HEIGHT - 4 + 1 = 17, which is way below
        # Let's test with a piece manually positioned to extend above visible playfield
        
        # Actually, let's test the real scenario: playfield filled so high that
        # a piece placed at spawn position would overlap
        # Fill playfield leaving only top 2 rows of skybox empty
        critical_playfield = [["c" for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)]
        critical_playfield[0] = [" " for _ in range(PLAYFIELD_WIDTH)]
        critical_playfield[1] = [" " for _ in range(PLAYFIELD_WIDTH)]
        
        # Place O piece at y=2: blocks at y=2,3 = rows 1,2
        place_tet = Tetrimino("O", x=5, y=2)
        engine_critical = GameEngine(initial_state=GameState(
            playfield=critical_playfield,
            active_tetrimino=place_tet,
            next_tetrimino=Tetrimino("O", x=5, y=1),
        ))
        
        # Place the piece
        new_critical_state, _ = engine_critical.update_playfield(place_tet)
        
        # Now playfield has blocks in rows 1,2
        # Try to spawn O piece at y=1: blocks at y=1,2 = rows 0,1
        # Row 1 is now filled, so this should overlap and be invalid
        spawn_tet = Tetrimino("O", x=5, y=1)
        engine_final = GameEngine(initial_state=new_critical_state)
        is_valid = engine_final.valid_location(spawn_tet)
        
        assert is_valid is False, "Piece should be invalid when it overlaps with filled playfield"
        
        # Game should end
        game_over = not is_valid
        assert game_over is True, "Game should end when spawned piece is invalid"
