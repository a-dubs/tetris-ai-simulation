"""Pygame-based GUI renderer for Tetris simulation."""

import pygame
from typing import Optional

from tetris.render.base import Renderer
from tetris.core.game_state import GameState
from tetris.core.constants import (
    PLAYFIELD_WIDTH,
    TOTAL_PLAYFIELD_HEIGHT,
    MINO_SIZE,
    COLORS,
    OUTLINE_COLORS,
)


class PygameRenderer(Renderer):
    """Pygame-based GUI renderer for visualizing Tetris games.

    Renders the game state including playfield, active tetrimino,
    next tetrimino, score, level, and lines cleared.
    """

    def __init__(self, window_title: str = "Tetris AI Simulation", fps: int = 60):
        """Initialize Pygame renderer.

        Args:
            window_title: Title for the window
            fps: Frames per second for rendering
        """
        # Check if we're in a test environment (pytest sets this)
        import os

        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
            # Don't initialize pygame in test environment
            self._test_mode = True
            self.screen = None
            self.clock = None
            self.fps = fps
            self.running = True
            return

        self._test_mode = False
        pygame.init()
        self.fps = fps
        self.clock = pygame.time.Clock()

        # Calculate window dimensions
        playfield_width_px = PLAYFIELD_WIDTH * MINO_SIZE
        playfield_height_px = TOTAL_PLAYFIELD_HEIGHT * MINO_SIZE
        sidebar_width = 200  # For score, next piece, etc.
        window_width = playfield_width_px + sidebar_width
        window_height = playfield_height_px

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(window_title)

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Colors
        self.bg_color = (20, 20, 20)
        self.sidebar_color = (30, 30, 30)
        self.text_color = (255, 255, 255)

        self.running = True

    def render(self, state: GameState) -> None:
        """Render the current game state.

        Args:
            state: Current game state to render
        """
        # Skip rendering in test mode
        if self._test_mode:
            return

        # Handle pygame events (for window close, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

        if not self.running:
            return

        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw playfield
        self._draw_playfield(state)

        # Draw active tetrimino
        if state.active_tetrimino:
            self._draw_tetrimino(state.active_tetrimino, state.playfield)

        # Draw sidebar (score, level, next piece)
        self._draw_sidebar(state)

        # Update display
        pygame.display.flip()
        # Small delay to make it easier to see (only for GUI, not headless)
        self.clock.tick(self.fps)

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color string to RGB tuple.

        Args:
            hex_color: Hex color string like "#0ff" or "#00ffff"

        Returns:
            RGB tuple (r, g, b)
        """
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            # Expand shorthand like "0ff" to "00ffff"
            hex_color = "".join(c * 2 for c in hex_color)
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def _draw_playfield(self, state: GameState) -> None:
        """Draw the playfield grid.
        
        Note: Playfield array row 0 is at the bottom logically (y=1),
        so we flip the Y coordinate when rendering.
        """
        for row in range(TOTAL_PLAYFIELD_HEIGHT):
            for col in range(PLAYFIELD_WIDTH):
                x = col * MINO_SIZE
                # Flip Y: row 0 (bottom) should be at bottom of screen
                # playfield_height - row - 1 converts array index to screen Y
                screen_y = (TOTAL_PLAYFIELD_HEIGHT - row - 1) * MINO_SIZE

                # Get cell color
                cell_char = state.playfield[row][col]
                color_hex = COLORS.get(cell_char, COLORS[" "])
                outline_hex = OUTLINE_COLORS.get(cell_char, OUTLINE_COLORS[" "])
                color = self._hex_to_rgb(color_hex)
                outline_color = self._hex_to_rgb(outline_hex)

                # Draw cell
                rect = pygame.Rect(x, screen_y, MINO_SIZE, MINO_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, outline_color, rect, 1)

    def _draw_tetrimino(self, tetrimino, playfield) -> None:
        """Draw a tetrimino on the playfield.
        
        Note: Tetrimino y coordinate is 1-indexed with y=1 at bottom.
        We need to flip the Y coordinate to match screen coordinates.
        """
        if not tetrimino:
            return

        for row in range(tetrimino.size):
            for col in range(tetrimino.size):
                if tetrimino.minos[row][col] != " ":
                    # Calculate playfield position
                    # Note: tetrimino coordinates are 1-indexed, y=1 is at bottom
                    pf_x = tetrimino.x + col - 1
                    pf_y = tetrimino.y + row - 1  # This is 1-indexed, y=1 is bottom

                    # Bounds check
                    if 0 <= pf_x < PLAYFIELD_WIDTH and 0 <= pf_y < TOTAL_PLAYFIELD_HEIGHT:
                        screen_x = pf_x * MINO_SIZE
                        # Flip Y: convert 1-indexed bottom-up to screen coordinates
                        # pf_y is 1-indexed (1 = bottom), convert to 0-indexed array row
                        array_row = pf_y - 1  # Convert to 0-indexed
                        screen_y = (TOTAL_PLAYFIELD_HEIGHT - array_row - 1) * MINO_SIZE

                        # Get color
                        cell_char = tetrimino.minos[row][col]
                        color_hex = COLORS.get(cell_char, COLORS[" "])
                        outline_hex = OUTLINE_COLORS.get(cell_char, OUTLINE_COLORS[" "])
                        color = self._hex_to_rgb(color_hex)
                        outline_color = self._hex_to_rgb(outline_hex)

                        # Draw cell
                        rect = pygame.Rect(screen_x, screen_y, MINO_SIZE, MINO_SIZE)
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, outline_color, rect, 2)

    def _draw_sidebar(self, state: GameState) -> None:
        """Draw sidebar with game information."""
        sidebar_x = PLAYFIELD_WIDTH * MINO_SIZE
        sidebar_y = 20

        # Background for sidebar
        sidebar_rect = pygame.Rect(sidebar_x, 0, 200, TOTAL_PLAYFIELD_HEIGHT * MINO_SIZE)
        pygame.draw.rect(self.screen, self.sidebar_color, sidebar_rect)

        # Score
        score_text = self.font_medium.render(f"Score: {state.score:,}", True, self.text_color)
        self.screen.blit(score_text, (sidebar_x + 10, sidebar_y))

        # Level
        level_text = self.font_medium.render(f"Level: {state.level}", True, self.text_color)
        self.screen.blit(level_text, (sidebar_x + 10, sidebar_y + 40))

        # Lines cleared
        lines_text = self.font_medium.render(
            f"Lines: {state.lines_cleared}", True, self.text_color
        )
        self.screen.blit(lines_text, (sidebar_x + 10, sidebar_y + 80))

        # Game over indicator
        if state.game_over:
            game_over_text = self.font_large.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(
                center=(sidebar_x + 100, TOTAL_PLAYFIELD_HEIGHT * MINO_SIZE // 2)
            )
            self.screen.blit(game_over_text, text_rect)

        # Next piece preview
        if state.next_tetrimino:
            next_label = self.font_small.render("Next:", True, self.text_color)
            self.screen.blit(next_label, (sidebar_x + 10, sidebar_y + 120))

            # Draw next tetrimino preview (scaled down)
            preview_size = MINO_SIZE // 2
            preview_x = sidebar_x + 20
            preview_y = sidebar_y + 150

            for row in range(state.next_tetrimino.size):
                for col in range(state.next_tetrimino.size):
                    if state.next_tetrimino.minos[row][col] != " ":
                        cell_char = state.next_tetrimino.minos[row][col]
                        color_hex = COLORS.get(cell_char, COLORS[" "])
                        outline_hex = OUTLINE_COLORS.get(cell_char, OUTLINE_COLORS[" "])
                        color = self._hex_to_rgb(color_hex)
                        outline_color = self._hex_to_rgb(outline_hex)

                        rect = pygame.Rect(
                            preview_x + col * preview_size,
                            preview_y + row * preview_size,
                            preview_size,
                            preview_size,
                        )
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, outline_color, rect, 1)

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        if not self._test_mode:
            pygame.quit()

    def is_running(self) -> bool:
        """Check if the renderer window is still running.

        Returns:
            True if window is open, False if closed
        """
        return self.running
