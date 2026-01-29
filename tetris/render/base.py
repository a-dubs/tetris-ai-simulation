"""Abstract base class for renderers."""

from abc import ABC, abstractmethod

from tetris.core.game_state import GameState


class Renderer(ABC):
    """Abstract base class for game renderers.

    Renderers handle visualization of the game state.
    Can be headless (no-op), GUI (Tkinter), or other implementations.
    """

    @abstractmethod
    def render(self, state: GameState) -> None:
        """Render the current game state.

        Args:
            state: Current game state to render
        """
        pass
