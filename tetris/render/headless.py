"""Headless renderer - no-op implementation for non-visual simulations."""

from tetris.render.base import Renderer
from tetris.core.game_state import GameState


class HeadlessRenderer(Renderer):
    """Headless renderer that does nothing.

    Used for performance testing and batch simulations where
    visualization is not needed.
    """

    def render(self, state: GameState) -> None:
        """Do nothing - headless renderer.

        Args:
            state: Current game state (ignored)
        """
        pass
