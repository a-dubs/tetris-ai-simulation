"""Tests for factory functions."""

import pytest

from tetris.simulation.factory import create_ai, create_renderer
from tetris.simulation.config import SimulationConfig
from tetris.ai.base import AIPlayer
from tetris.ai.random import RandomAIPlayer
from tetris.ai.greedy import GreedyAIPlayer
from tetris.ai.fast import FastAIPlayer
from tetris.render.base import Renderer
from tetris.render.headless import HeadlessRenderer


class TestFactory:
    """Test factory functions."""

    def test_create_ai_random(self):
        """Test creating random AI."""
        config = SimulationConfig(ai_type="random")
        ai = create_ai("random", config)
        assert isinstance(ai, RandomAIPlayer)
        assert isinstance(ai, AIPlayer)

    def test_create_ai_greedy(self):
        """Test creating greedy AI."""
        config = SimulationConfig(ai_type="greedy", moves_per_second=4, initial_level=1)
        ai = create_ai("greedy", config)
        assert isinstance(ai, GreedyAIPlayer)
        assert isinstance(ai, AIPlayer)

    def test_create_ai_fast(self):
        """Test creating fast AI."""
        config = SimulationConfig(ai_type="fast", moves_per_second=4, initial_level=1)
        ai = create_ai("fast", config)
        assert isinstance(ai, FastAIPlayer)
        assert isinstance(ai, AIPlayer)

    def test_create_ai_invalid_type(self):
        """Test that invalid AI type raises ValueError."""
        config = SimulationConfig(ai_type="invalid")
        with pytest.raises(ValueError, match="Unknown AI type"):
            create_ai("invalid", config)

    def test_create_renderer_headless(self):
        """Test creating headless renderer."""
        renderer = create_renderer(headless=True)
        assert isinstance(renderer, HeadlessRenderer)
        assert isinstance(renderer, Renderer)

    def test_create_renderer_gui(self):
        """Test creating GUI renderer (currently returns headless)."""
        # TODO: When GUI renderer is implemented, update this test
        renderer = create_renderer(headless=False)
        assert isinstance(renderer, Renderer)
        # Currently returns HeadlessRenderer as placeholder
        assert isinstance(renderer, HeadlessRenderer)
