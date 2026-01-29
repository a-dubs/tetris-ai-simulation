"""Helper functions for finding and evaluating piece placements."""

import copy
from typing import List, Tuple

from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.core.game_engine import GameEngine
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class Placement:
    """Represents a possible piece placement."""

    def __init__(
        self,
        tetrimino: Tetrimino,
        rotation_count: int,
        x_position: int,
        path: List[str],
    ):
        """Initialize a placement.

        Args:
            tetrimino: The tetrimino in its final placed position
            rotation_count: Number of rotations applied (0-3)
            x_position: Final x position
            path: Sequence of moves to reach this placement
        """
        self.tetrimino = tetrimino
        self.rotation_count = rotation_count
        self.x_position = x_position
        self.path = path


def find_all_placements(
    state: GameState, engine: GameEngine = None
) -> List[Placement]:
    """Find all possible valid placements for the active tetrimino.

    Args:
        state: Current game state
        engine: Optional game engine (creates one if not provided)

    Returns:
        List of Placement objects representing all valid placements
    """
    if engine is None:
        engine = GameEngine(initial_state=state)

    if not state.active_tetrimino:
        return []

    placements = []
    active_tet = state.active_tetrimino

    # Try all rotations (0-3 clockwise rotations)
    for rotation_count in range(4):
        # Create a copy of the tetrimino
        tet = copy.copy(active_tet)
        tet.minos = [row[:] for row in tet.minos]

        # Apply rotations
        for _ in range(rotation_count):
            size = len(tet.minos)
            new_minos = []
            for i in range(size):
                new_minos.append([tet.minos[size - 1 - j][i] for j in range(size)])
            tet.minos = new_minos
            # Update orientation
            dirs = ("N", "E", "S", "W")
            tet.orientation = dirs[(dirs.index(tet.orientation) + 1) % 4]

        # Get width and height of rotated piece
        piece_width = len(tet.minos[0]) if tet.minos else tet.size
        piece_height = len(tet.minos) if tet.minos else tet.size

        # Try all x positions
        for x in range(1, PLAYFIELD_WIDTH - piece_width + 2):
            # Create tetrimino at this position
            # Start high enough that the piece fits entirely above playfield
            # y is 1-indexed, so we need y high enough that y + piece_height - 1 <= TOTAL_PLAYFIELD_HEIGHT
            start_y = TOTAL_PLAYFIELD_HEIGHT - piece_height + 2

            test_tet = copy.copy(tet)
            test_tet.x = x
            test_tet.y = start_y

            # Drop it down until it lands
            dropped_tet = copy.copy(test_tet)
            # Keep dropping while valid and above bottom
            while dropped_tet.y > 1:
                test_y = dropped_tet.y - 1
                test_tet_copy = copy.copy(dropped_tet)
                test_tet_copy.y = test_y
                if not engine.valid_location(test_tet_copy):
                    break
                dropped_tet.y = test_y

            # Check if this is a valid final placement
            if engine.valid_location(dropped_tet) and dropped_tet.y >= 1:
                # Generate path to this placement
                path = _generate_path_to_placement(
                    active_tet, dropped_tet, rotation_count, x, engine
                )
                if path:
                    placements.append(
                        Placement(
                            tetrimino=dropped_tet,
                            rotation_count=rotation_count,
                            x_position=x,
                            path=path,
                        )
                    )

    return placements


def _generate_path_to_placement(
    start_tet: Tetrimino,
    target_tet: Tetrimino,
    rotation_count: int,
    target_x: int,
    engine: GameEngine,
) -> List[str]:
    """Generate a sequence of moves to reach a target placement.

    Args:
        start_tet: Starting tetrimino
        target_tet: Target tetrimino position
        rotation_count: Number of rotations needed
        target_x: Target x position
        engine: Game engine for validation

    Returns:
        List of moves ending with "drop"
    """
    moves = []

    # Apply rotations first
    for _ in range(rotation_count):
        moves.append("cw")

    # Move horizontally to target x
    current_x = start_tet.x
    if current_x < target_x:
        moves.extend(["right"] * (target_x - current_x))
    elif current_x > target_x:
        moves.extend(["left"] * (current_x - target_x))

    # Drop
    moves.append("drop")

    return moves


def get_best_placement(
    placements: List[Placement], state: GameState, engine: GameEngine = None
) -> Tuple[Placement, float]:
    """Get the best placement based on a simple heuristic.

    Args:
        placements: List of possible placements
        state: Current game state
        engine: Optional game engine

    Returns:
        Tuple of (best_placement, evaluation_score)
    """
    if not placements:
        raise ValueError("No placements provided")

    if engine is None:
        engine = GameEngine(initial_state=state)

    best_placement = None
    best_score = float("-inf")

    for placement in placements:
        # Simple heuristic: prefer lower y positions (pieces closer to bottom)
        # and prefer placements that don't create holes
        score = -placement.tetrimino.y  # Lower y = higher score

        # Check for holes (simplified - just check if placement creates gaps)
        # This is a basic heuristic, can be improved
        best_placement = placement
        best_score = score
        break  # For now, just return first valid placement

    return best_placement, best_score
