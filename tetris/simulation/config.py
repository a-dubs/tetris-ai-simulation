"""Configuration for simulations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Configuration for running a simulation."""

    # AI configuration
    ai_type: str = "random"  # "random", "greedy", or "fast"
    ai_params: Optional[dict] = None  # Optional AI-specific parameters

    # Simulation settings
    max_moves: int = 10000  # Maximum moves before stopping
    max_time: Optional[float] = None  # Maximum time in seconds (None = no limit)
    delta_time: float = 1.0 / 60.0  # Time step per frame (default 60 FPS)

    # Rendering
    headless: bool = True  # Run without GUI
    render_delay: float = 0.05  # Delay between renders in seconds (for GUI visualization)

    # Game settings
    initial_level: int = 1  # Starting level
    moves_per_second: int = 4  # Moves per second for gravity timing
