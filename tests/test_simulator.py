"""Tests for Simulator class."""

import pytest

from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState
from tetris.ai.random import RandomAIPlayer
from tetris.render.headless import HeadlessRenderer
from tetris.simulation.simulator import Simulator, SimulationResult
from tetris.simulation.config import SimulationConfig
from tetris.core.tetrimino import Tetrimino
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class TestSimulator:
    """Test Simulator class."""

    def test_create_simulator(self):
        """Test creating a simulator."""
        engine = GameEngine()
        ai = RandomAIPlayer()
        renderer = HeadlessRenderer()
        config = SimulationConfig(max_moves=10)

        sim = Simulator(engine, ai, renderer, config)
        assert sim is not None
        assert sim.engine == engine
        assert sim.ai == ai
        assert sim.renderer == renderer
        assert sim.config == config

    def test_run_short_simulation(self):
        """Test running a very short simulation."""
        engine = GameEngine()
        ai = RandomAIPlayer()
        renderer = HeadlessRenderer()
        config = SimulationConfig(max_moves=5, headless=True)

        sim = Simulator(engine, ai, renderer, config)
        result = sim.run()

        assert isinstance(result, SimulationResult)
        assert result.moves_executed <= config.max_moves
        assert result.time_elapsed >= 0
        assert isinstance(result.final_state, GameState)

    def test_simulation_respects_max_moves(self):
        """Test that simulation stops at max_moves."""
        engine = GameEngine()
        ai = RandomAIPlayer()
        renderer = HeadlessRenderer()
        config = SimulationConfig(max_moves=3, headless=True)

        sim = Simulator(engine, ai, renderer, config)
        result = sim.run()

        assert result.moves_executed <= config.max_moves

    def test_simulation_result_has_final_state(self):
        """Test that simulation result contains final state."""
        engine = GameEngine()
        ai = RandomAIPlayer()
        renderer = HeadlessRenderer()
        config = SimulationConfig(max_moves=5, headless=True)

        sim = Simulator(engine, ai, renderer, config)
        result = sim.run()

        assert result.final_state is not None
        assert isinstance(result.final_state, GameState)

    def test_simulation_with_different_ai(self):
        """Test that simulator works with different AI types."""
        engine = GameEngine()
        renderer = HeadlessRenderer()
        config = SimulationConfig(max_moves=3, headless=True)

        # Test with RandomAIPlayer
        ai = RandomAIPlayer()
        sim = Simulator(engine, ai, renderer, config)
        result = sim.run()
        assert result.moves_executed <= config.max_moves
