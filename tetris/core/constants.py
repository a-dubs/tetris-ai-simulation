"""Game constants for Tetris simulation."""

# Playfield dimensions
PLAYFIELD_WIDTH = 10  # width of playfield in # of minos
PLAYFIELD_HEIGHT = 20  # height of visible playfield in # of minos
SKYBOX_HEIGHT = 4  # height of buffer zone above visible playfield
TOTAL_PLAYFIELD_HEIGHT = PLAYFIELD_HEIGHT + SKYBOX_HEIGHT

# Bag configuration
BAG_SIZE = 1

# Rendering constants
MINO_SIZE = 40
MINO_BORDER_THICKNESS = 1

# Press speeds (moves per second)
PRESS_SPEEDS = {
    "Beginner": 3,
    "Intermediate": 4,
    "Advanced": 5,
    "Pro": 6,
}

# Color mappings for rendering
COLORS = {
    'c': "#0ff",  # cyan (I piece)
    'y': "#ff0",  # yellow (O piece)
    'p': "#f0f",  # purple (T piece)
    'g': "#0f0",  # green (S piece)
    'r': "#f00",  # red (Z piece)
    'b': "#00b",  # blue (J piece)
    'o': "#f80",  # orange (L piece)
    ' ': "#fff",  # empty
    '~': "#ccc",  # ghost piece
}

OUTLINE_COLORS = {
    'c': "#0aa",
    'y': "#aa0",
    'p': "#a0a",
    'g': "#0a0",
    'r': "#a00",
    'b': "#006",
    'o': "#a50",
    ' ': "#aaa",
    '~': "#888",
}


def spawn_y_for_size(size: int) -> int:
    """Return spawn Y so piece top aligns with visible playfield top."""
    return PLAYFIELD_HEIGHT - size + 1
