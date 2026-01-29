"""Simulator orchestrates game engine, AI, and renderer."""

from dataclasses import dataclass
from typing import Optional

from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState
from tetris.ai.base import AIPlayer
from tetris.render.base import Renderer
from tetris.simulation.config import SimulationConfig


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    final_state: GameState
    moves_executed: int
    time_elapsed: float
    game_over: bool


class Simulator:
    """Orchestrates game engine, AI, and renderer to run simulations.

    The Simulator coordinates all components to run a complete game:
    - GameEngine manages game state and mechanics
    - AIPlayer provides move decisions
    - Renderer handles visualization (if any)
    """

    def __init__(
        self,
        engine: GameEngine,
        ai: AIPlayer,
        renderer: Renderer,
        config: SimulationConfig,
    ):
        """Initialize simulator.

        Args:
            engine: Game engine managing state
            ai: AI player making decisions
            renderer: Renderer for visualization
            config: Simulation configuration
        """
        self.engine = engine
        self.ai = ai
        self.renderer = renderer
        self.config = config

    def run(self) -> SimulationResult:
        """Run a single game simulation.

        Returns:
            SimulationResult with final state and statistics
        """
        moves_executed = 0
        current_moves: list[str] = []
        time_elapsed = 0.0

        # Initial render
        self.renderer.render(self.engine.state)

        # Main game loop
        while not self.engine.state.game_over:
            # Check termination conditions
            if moves_executed >= self.config.max_moves:
                break

            if self.config.max_time and time_elapsed >= self.config.max_time:
                break

            # Get moves from AI if we don't have any
            if not current_moves:
                current_moves = self.ai.get_move_sequence(self.engine.state)
                self.ai.update_state(self.engine.state)

            # Execute next move
            if current_moves:
                move = current_moves.pop(0)
                self.engine.state = self.engine.apply_move(move)
                moves_executed += 1
                time_elapsed += self.config.delta_time

                # Render after move
                self.renderer.render(self.engine.state)

                # If piece was dropped, update playfield and spawn new piece
                if move == "drop":
                    new_state, lines_cleared = self.engine.update_playfield(
                        self.engine.state.active_tetrimino
                    )

                    # Update score and level
                    from tetris.core.constants import BAG_SIZE, PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT
                    from tetris.core.tetrimino import Tetrimino

                    # Calculate score (simplified - should use proper scoring)
                    score = new_state.score + lines_cleared * 100 * new_state.level

                    # Check level up
                    new_level = new_state.level
                    clears_needed = new_level * ((5 + new_level * 5) / 2)
                    if new_state.lines_cleared >= clears_needed:
                        new_level += 1

                    # Spawn new tetrimino
                    bag = new_state.bag.copy()
                    if not bag:
                        bag = Tetrimino.make_bag(BAG_SIZE)

                    active_tet = new_state.next_tetrimino
                    next_tet = Tetrimino(
                        bag.pop(0),
                        x=PLAYFIELD_WIDTH // 2 - 1,
                        y=TOTAL_PLAYFIELD_HEIGHT - 3,
                    )

                    # Check game over
                    temp_engine = GameEngine(initial_state=new_state)
                    game_over = not temp_engine.valid_location(active_tet)

                    # Update state
                    self.engine.state = (
                        new_state.with_score(score)
                        .with_level(new_level)
                        .with_lines_cleared(new_state.lines_cleared + lines_cleared)
                        .with_active_tetrimino(active_tet)
                        .with_next_tetrimino(next_tet)
                        .with_bag(bag)
                        .with_game_over(game_over)
                        .with_time_elapsed(time_elapsed)
                    )

                    # Clear move queue for new piece
                    current_moves = []

                    # Render after state update
                    self.renderer.render(self.engine.state)

        return SimulationResult(
            final_state=self.engine.state,
            moves_executed=moves_executed,
            time_elapsed=time_elapsed,
            game_over=self.engine.state.game_over,
        )
