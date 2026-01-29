"""Immutable game state for Tetris simulation."""

from dataclasses import dataclass, field, replace
from typing import List, Optional

from tetris.core.tetrimino import Tetrimino


@dataclass(frozen=True)
class GameState:
    """Immutable game state for Tetris.

    All state mutations return a new GameState instance,
    preserving immutability.
    """

    playfield: List[List[str]]
    active_tetrimino: Optional[Tetrimino]
    next_tetrimino: Tetrimino
    bag: List[str] = field(default_factory=list)
    score: int = 0
    level: int = 1
    lines_cleared: int = 0
    game_over: bool = False
    time_elapsed: float = 0.0

    def with_playfield(self, playfield: List[List[str]]) -> "GameState":
        """Return new state with updated playfield."""
        return replace(self, playfield=playfield)

    def with_active_tetrimino(self, active_tetrimino: Optional[Tetrimino]) -> "GameState":
        """Return new state with updated active tetrimino."""
        return replace(self, active_tetrimino=active_tetrimino)

    def with_next_tetrimino(self, next_tetrimino: Tetrimino) -> "GameState":
        """Return new state with updated next tetrimino."""
        return replace(self, next_tetrimino=next_tetrimino)

    def with_bag(self, bag: List[str]) -> "GameState":
        """Return new state with updated bag."""
        return replace(self, bag=bag)

    def with_score(self, score: int) -> "GameState":
        """Return new state with updated score."""
        return replace(self, score=score)

    def with_level(self, level: int) -> "GameState":
        """Return new state with updated level."""
        return replace(self, level=level)

    def with_lines_cleared(self, lines_cleared: int) -> "GameState":
        """Return new state with updated lines cleared."""
        return replace(self, lines_cleared=lines_cleared)

    def with_game_over(self, game_over: bool) -> "GameState":
        """Return new state with updated game_over flag."""
        return replace(self, game_over=game_over)

    def with_time_elapsed(self, time_elapsed: float) -> "GameState":
        """Return new state with updated time elapsed."""
        return replace(self, time_elapsed=time_elapsed)
