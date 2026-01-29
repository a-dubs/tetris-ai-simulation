"""Game engine for Tetris - pure game logic without AI or rendering."""

import copy
from typing import Tuple

from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT, BAG_SIZE


class GameEngine:
    """Pure game logic engine - no AI, no rendering.

    Manages game state and provides methods for game mechanics.
    All methods return new GameState instances (immutable).
    """

    def __init__(self, initial_state: GameState = None):
        """Initialize game engine.

        Args:
            initial_state: Optional initial game state. If None, creates default state.
        """
        self.state = initial_state or self._create_initial_state()

    def _create_initial_state(self) -> GameState:
        """Create initial game state."""
        playfield = [
            [" " for _ in range(PLAYFIELD_WIDTH)] for _ in range(TOTAL_PLAYFIELD_HEIGHT)
        ]
        bag = Tetrimino.make_bag(BAG_SIZE)
        next_tet = Tetrimino(
            bag.pop(0), x=PLAYFIELD_WIDTH // 2 - 1, y=TOTAL_PLAYFIELD_HEIGHT - 3
        )
        active_tet = Tetrimino(
            bag.pop(0), x=PLAYFIELD_WIDTH // 2 - 1, y=TOTAL_PLAYFIELD_HEIGHT - 3
        )

        return GameState(
            playfield=playfield,
            active_tetrimino=active_tet,
            next_tetrimino=next_tet,
            bag=bag,
        )

    def apply_move(self, move: str) -> GameState:
        """Apply a move to the current state, returning a new state.

        Args:
            move: One of "left", "right", "cw", "ccw", "drop"

        Returns:
            New GameState with move applied
        """
        tet = copy.copy(self.state.active_tetrimino)
        tet.minos = [row[:] for row in tet.minos]  # Deep copy minos

        if move in ("left", "right"):
            self._move_tetrimino(tet, move)
        elif move in ("cw", "ccw"):
            self._rotate_tetrimino(tet, move)
        elif move == "drop":
            self._drop_tetrimino(tet)

        return self.state.with_active_tetrimino(tet)

    def _move_tetrimino(self, tet: Tetrimino, direction: str):
        """Move tetrimino left or right.

        Args:
            tet: Tetrimino to move (mutated)
            direction: "left" or "right"
        """
        if direction in ("left", "l", "west", "w"):
            xshift = -1
        elif direction in ("right", "r", "east", "e"):
            xshift = 1
        else:
            return

        tet.x += xshift
        if not self.valid_location(tet):
            tet.x -= xshift
        elif tet.landed:
            tet.ldml -= 1

    def _rotate_tetrimino(self, tet: Tetrimino, direction: str):
        """Rotate tetrimino clockwise or counter-clockwise.

        Args:
            tet: Tetrimino to rotate (mutated)
            direction: "cw" or "ccw"
        """
        mult = None
        if direction in ("ccw", "left", "l"):
            mult = -1
        elif direction in ("cw", "right", "r"):
            mult = 1
        else:
            return

        # Pure Python rotation
        size = len(tet.minos)
        minos = []
        if mult == 1:  # Clockwise rotation
            for i in range(size):
                minos.append([tet.minos[size - 1 - j][i] for j in range(size)])
        else:  # Counter-clockwise rotation
            for i in range(size):
                minos.append([tet.minos[j][size - 1 - i] for j in range(size)])

        tet.minos = minos
        dirs = ("N", "E", "S", "W")
        tet.orientation = dirs[(dirs.index(tet.orientation) + mult) % 4]

        # Push tetrimino if overflowing playfield
        xshift = 0
        if tet.x < 1:
            for col in range(tet.size):
                for row in range(tet.size):
                    if xshift == 0 and tet.minos[row][col] != " ":
                        xshift = 1 - tet.x + col
                        break

        elif tet.x > PLAYFIELD_WIDTH + 1 - tet.size:
            for col in range(-1, -tet.size - 1, -1):
                for row in range(tet.size):
                    if xshift == 0 and tet.minos[row][col] != " ":
                        xshift = PLAYFIELD_WIDTH - tet.x - tet.size - col
                        break

        tet.x += xshift
        while not self.valid_location(tet) and tet.y < TOTAL_PLAYFIELD_HEIGHT - 2:
            tet.y += 1

        if tet.landed:
            tet.ldml -= 1

    def _drop_tetrimino(self, tet: Tetrimino):
        """Drop tetrimino to lowest valid position.

        Args:
            tet: Tetrimino to drop (mutated)
        """
        start = tet.y
        while self.valid_location(tet):
            tet.y -= 1
        tet.y += 1
        tet.y = start if tet.y > start else tet.y
        tet.ldml = 0

    def valid_location(self, tet: Tetrimino) -> bool:
        """Check if tetrimino is in a valid location.

        Args:
            tet: Tetrimino to check

        Returns:
            True if location is valid, False otherwise
        """
        pf = self.state.playfield
        for row in range(tet.size):
            for col in range(tet.size):
                ib = self._in_bounds(tet.x + col, tet.y + row)
                if (
                    ib
                    and tet.minos[row][col] != " "
                    and pf[tet.y + row - 1][tet.x + col - 1] != " "
                ) or (not ib and tet.minos[row][col] != " "):
                    return False

        return True

    def _in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates are within playfield bounds.

        Args:
            x: X coordinate (1-indexed)
            y: Y coordinate (1-indexed)

        Returns:
            True if in bounds, False otherwise
        """
        return (
            x >= 1
            and x <= PLAYFIELD_WIDTH
            and y >= 1
            and y <= TOTAL_PLAYFIELD_HEIGHT
        )

    def place_tetrimino(self, tet: Tetrimino) -> GameState:
        """Place tetrimino on playfield without clearing lines.

        Args:
            tet: Tetrimino to place

        Returns:
            New GameState with tetrimino placed
        """
        pf = [row[:] for row in self.state.playfield]  # Copy playfield

        for row in range(tet.size):
            for col in range(tet.size):
                if tet.minos[row][col] != " ":
                    # Convert 1-indexed coordinates to 0-indexed
                    pf_y = tet.y + row - 1
                    pf_x = tet.x + col - 1
                    # Bounds check
                    if 0 <= pf_y < len(pf) and 0 <= pf_x < len(pf[0]):
                        pf[pf_y][pf_x] = tet.minos[row][col]

        return self.state.with_playfield(pf)

    def update_playfield(self, tet: Tetrimino) -> Tuple[GameState, int]:
        """Place tetrimino and clear full lines.

        Args:
            tet: Tetrimino to place

        Returns:
            Tuple of (new GameState, number of lines cleared)
        """
        # Place tetrimino first
        state = self.place_tetrimino(tet)
        pf = [row[:] for row in state.playfield]  # Copy playfield

        # Clear full lines (from bottom to top)
        lines_cleared = 0
        i = len(pf) - 1
        while i >= 0:
            if pf[i].count(" ") == 0:
                del pf[i]
                pf.append([" " for _ in range(PLAYFIELD_WIDTH)])
                lines_cleared += 1
                # Don't increment i since we deleted a row
            else:
                i -= 1

        return state.with_playfield(pf), lines_cleared

    def _make_clears(self, playfield: list) -> int:
        """Count and clear full lines (helper method).

        Args:
            playfield: Playfield to check

        Returns:
            Number of lines cleared
        """
        clears = 0
        for i in range(-1, -len(playfield) - 1, -1):
            if playfield[i].count(" ") == 0:
                clears += 1
        return clears
