"""Factory functions for creating AI players and renderers."""

from tetris.ai.base import AIPlayer
from tetris.ai.random import RandomAIPlayer
from tetris.ai.greedy import GreedyAIPlayer
from tetris.ai.fast import FastAIPlayer
from tetris.render.base import Renderer
from tetris.render.headless import HeadlessRenderer
from tetris.simulation.config import SimulationConfig


def create_ai(ai_type: str, config: SimulationConfig) -> AIPlayer:
    """Create an AI player based on type.

    Args:
        ai_type: Type of AI ("random", "greedy", or "fast")
        config: Simulation configuration

    Returns:
        AIPlayer instance

    Raises:
        ValueError: If ai_type is not recognized
    """
    ai_params = config.ai_params or {}

    if ai_type == "random":
        return RandomAIPlayer(**ai_params)
    elif ai_type == "greedy":
        return GreedyAIPlayer(
            mps=config.moves_per_second,
            level=config.initial_level,
            method="greedy",
            params=ai_params.get("params"),
        )
    elif ai_type == "fast":
        return FastAIPlayer(
            mps=config.moves_per_second,
            level=config.initial_level,
            method="greedy",
            params=ai_params.get("params"),
        )
    else:
        raise ValueError(f"Unknown AI type: {ai_type}. Must be one of: random, greedy, fast")


def create_renderer(headless: bool) -> Renderer:
    """Create a renderer based on configuration.

    Args:
        headless: If True, create headless renderer, otherwise create GUI renderer

    Returns:
        Renderer instance

    Note:
        GUI renderer not yet implemented, always returns HeadlessRenderer
    """
    if headless:
        return HeadlessRenderer()
    else:
        # TODO: Implement GUI renderer
        # For now, return headless renderer
        return HeadlessRenderer()
