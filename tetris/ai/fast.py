"""Fast AI player - direct placement evaluation with improved heuristics."""

import copy
from typing import Dict, Optional

from tetris.ai.base import AIPlayer
from tetris.ai.placement import find_all_placements, Placement
from tetris.core.game_state import GameState
from tetris.core.game_engine import GameEngine
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class FastAIPlayer(AIPlayer):
    """Fast AI player using direct placement evaluation.

    Evaluates all possible placements directly without recursive search.
    Uses improved heuristics that prioritize:
    - Low stack height
    - Avoiding unreachable holes
    - Going for line clears

    Much faster than GreedyAIPlayer but potentially less optimal.
    """

    def __init__(self, mps: int = 4, level: int = 1, method: str = "greedy", params: Optional[Dict] = None):
        """Initialize fast AI player.

        Args:
            mps: Moves per second (for compatibility, not used in new structure)
            level: Starting level (for compatibility, not used in new structure)
            method: Evaluation method (unused, kept for compatibility)
            params: Optional parameter weights for heuristic evaluation
        """
        self.mps = mps
        self.level = level
        self.method = method
        self.params = params or self._get_improved_default_params()
        self._eval_cache: Dict = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _get_improved_default_params() -> Dict:
        """Get improved default parameters for Fast AI."""
        return {
            # Stack height penalties (keep stack low)
            "m_stack_w": -8.0,  # Strong penalty for max stack height
            "m_stack_e": 1.5,   # Moderate exponent
            "a_stack_w": -5.0,  # Penalty for average stack height
            "a_stack_e": 1.2,
            
            # Cliff penalties (avoid creating holes)
            "cliff_l_w": -6.0,  # Penalty for long horizontal cliffs
            "cliff_l_e": 1.3,
            "cliff_h_w": -7.0,  # Strong penalty for tall vertical cliffs (holes)
            "cliff_h_e": 1.5,
            
            # Stack danger (avoid high stacks)
            "stack_d_w": -4.0,
            "stack_d_e": 1.2,
            "stack_d_thresh": 15,  # Start penalizing above row 15
            
            # Score rewards (prioritize clears - MUCH higher for multiple clears)
            "score_w": 150.0,  # Base reward per line clear (increased significantly)
            "tetris_bonus": 1000.0,  # MASSIVE bonus for 4-line clear (Tetris) - should ALWAYS be chosen!
            
            # Game over (very strong penalty)
            "go_w": -1000.0,
        }

    def get_move_sequence(self, state: GameState) -> list[str]:
        """Get best move sequence using fast direct evaluation.

        Args:
            state: Current game state

        Returns:
            List of moves ending with "drop"
        """
        if not state.active_tetrimino:
            return ["drop"]

        engine = GameEngine(initial_state=state)
        
        # Find all possible placements
        placements = find_all_placements(state, engine)
        
        if not placements:
            return ["drop"]
        
        # Evaluate each placement and select the best
        best_placement = None
        best_score = float("-inf")
        
        for placement in placements:
            score = self._evaluate_placement(placement, state, engine)
            if score > best_score:
                best_score = score
                best_placement = placement
        
        if best_placement:
            return best_placement.path
        
        return ["drop"]

    def _evaluate_placement(self, placement: Placement, state: GameState, engine: GameEngine) -> float:
        """Evaluate a placement using improved heuristics.

        Args:
            placement: Placement to evaluate
            state: Current game state
            engine: Game engine

        Returns:
            Evaluation score (higher is better)
        """
        # Create a copy of the state and place the piece
        test_engine = GameEngine(initial_state=state)
        new_state, lines_cleared = test_engine.update_playfield(placement.tetrimino)
        
        # Special bonus: if this creates a 4-line clear (Tetris), give massive boost
        # This ensures Tetris opportunities are ALWAYS taken
        base_score = self._improved_heuristic_evaluation(new_state.playfield, lines_cleared)
        
        if lines_cleared == 4:
            # Extra boost for Tetris - make it irresistible!
            base_score += 500.0
        
        return base_score

    def _improved_heuristic_evaluation(self, playfield: list, lines_cleared: int) -> float:
        """Improved heuristic evaluation prioritizing low stacks, avoiding holes, and clears.

        Args:
            playfield: Playfield after placing piece
            lines_cleared: Number of lines cleared

        Returns:
            Evaluation score (higher is better)
        """
        # Try cache first
        cache_key = self._playfield_hash(playfield)
        if cache_key in self._eval_cache:
            self._cache_hits += 1
            cached_score = self._eval_cache[cache_key]
            # Add line clear bonus (not cached)
            clear_bonus = lines_cleared * self.params["score_w"]
            if lines_cleared == 4:
                clear_bonus += self.params.get("tetris_bonus", 200.0)
            elif lines_cleared >= 2:
                clear_bonus += lines_cleared * 20.0
            return cached_score + clear_bonus
        
        self._cache_misses += 1
        
        pfw = len(playfield[0])
        pfh = len(playfield)
        
        # Calculate stack height metrics
        row_empty_counts = [row.count(" ") for row in playfield]
        max_stack_height = sum([int(empty_count < pfw) for empty_count in row_empty_counts])
        
        # Calculate column heights for average stack height
        column_heights = []
        for col in range(pfw):
            max_height = 0
            for row in range(pfh):
                if playfield[row][col] != " ":
                    max_height = pfh - row
                    break
            column_heights.append(max_height)
        a_stacks = sum(column_heights) / float(pfw) if pfw > 0 else 0
        
        # Count holes (unreachable empty spaces)
        hole_count, hole_depth_sum = self._count_holes(playfield)
        
        # Count wells (deep columns)
        well_depth_sum = self._count_wells(playfield, column_heights)
        
        # Calculate cliff penalties
        cliff_heights_sum, cliff_lengths = self._calculate_cliff_penalties(playfield, pfw, pfh)
        
        # Stack danger (penalize high stacks)
        stack_d_thresh_int = int(self.params["stack_d_thresh"])
        stack_danger = sum([
            int(row_empty_counts[row] < pfw) * ((row - stack_d_thresh_int) ** self.params["stack_d_e"])
            for row in range(stack_d_thresh_int + 1, pfh)
        ])
        
        # Build evaluation score
        eval_score = 0
        
        # Clear rewards FIRST (highest priority - evaluate BEFORE stack penalties)
        # Multiple clears are exponentially better (Tetris = 4 clears is amazing!)
        if lines_cleared > 0:
            # Base reward for clears
            eval_score += lines_cleared * self.params["score_w"]
            # Extra bonus for Tetris (4-line clear)
            if lines_cleared == 4:
                eval_score += self.params.get("tetris_bonus", 200.0)
            # Bonus multiplier for multiple clears (2 clears > 2x single clear)
            elif lines_cleared >= 2:
                eval_score += lines_cleared * 20.0  # Extra bonus for 2+ clears
        
        # Stack height penalties (keep stack low)
        # NOTE: These are evaluated AFTER clears, so clearing lines reduces penalties
        eval_score += (max_stack_height ** self.params["m_stack_e"]) * self.params["m_stack_w"]
        eval_score += (a_stacks ** self.params["a_stack_e"]) * self.params["a_stack_w"]
        eval_score += stack_danger * self.params["stack_d_w"]
        
        # Hole penalties (avoid creating unreachable holes)
        eval_score += hole_count * -10.0  # Base penalty per hole
        eval_score += hole_depth_sum * -2.0  # Additional penalty for deep holes
        
        # Well penalties (avoid deep columns)
        eval_score += well_depth_sum * -3.0
        
        # Cliff penalties
        eval_score += cliff_lengths
        eval_score += cliff_heights_sum
        
        # Game over penalty (very strong)
        game_over = self._check_game_over(playfield)
        eval_score += int(game_over) * self.params["go_w"]
        
        # Bonus for keeping stack balanced (lower variance in column heights)
        if len(column_heights) > 1:
            height_variance = sum([(h - a_stacks) ** 2 for h in column_heights]) / len(column_heights)
            eval_score += height_variance * -1.0  # Prefer balanced stacks
        
        # Cache result (without line clear bonuses, since those vary)
        clear_bonus = lines_cleared * self.params["score_w"]
        if lines_cleared == 4:
            clear_bonus += self.params.get("tetris_bonus", 200.0)
        elif lines_cleared >= 2:
            clear_bonus += lines_cleared * 20.0
        
        if len(self._eval_cache) < 10000:
            self._eval_cache[cache_key] = eval_score - clear_bonus
        
        return eval_score

    def _count_holes(self, pf: list) -> tuple[int, int]:
        """Count unreachable holes (empty cells with blocks above them).

        Returns:
            Tuple of (hole_count, hole_depth_sum)
        """
        pfw = len(pf[0])
        pfh = len(pf)
        hole_count = 0
        hole_depth_sum = 0
        
        for col in range(pfw):
            # Scan from top to bottom to find holes
            for row in range(pfh):
                if pf[row][col] == " ":  # Empty cell
                    # Check if there are any blocks above this empty cell
                    blocks_above = False
                    depth = 0
                    for check_row in range(row - 1, -1, -1):
                        if pf[check_row][col] != " ":
                            blocks_above = True
                            depth += 1
                        else:
                            break
                    
                    if blocks_above:
                        hole_count += 1
                        hole_depth_sum += depth
        
        return hole_count, hole_depth_sum

    def _count_wells(self, pf: list, column_heights: list) -> float:
        """Count wells (deep columns significantly lower than neighbors).

        Returns:
            well_depth_sum
        """
        pfw = len(pf[0])
        well_depth_sum = 0.0
        
        # Check for wells (columns significantly lower than neighbors)
        for col in range(1, pfw - 1):
            center_height = column_heights[col]
            left_height = column_heights[col - 1]
            right_height = column_heights[col + 1]
            
            # Well if center is at least 3 blocks lower than both neighbors
            if center_height < left_height - 2 and center_height < right_height - 2:
                well_depth = min(left_height, right_height) - center_height
                well_depth_sum += well_depth ** 1.5  # Exponential penalty for deeper wells
        
        return well_depth_sum

    def _calculate_cliff_penalties(self, pf: list, pfw: int, pfh: int) -> tuple[float, float]:
        """Calculate cliff penalties (horizontal and vertical).

        Returns:
            Tuple of (cliff_heights_sum, cliff_lengths)
        """
        # Detect cliffs (horizontal overhangs)
        cliffs = []
        for row in range(1, pfh - 4):
            cliff_row = []
            prev_row_data = pf[row - 1]
            curr_row_data = pf[row]
            for col in range(pfw):
                cliff_row.append(int(curr_row_data[col] != " " and prev_row_data[col] == " "))
            cliffs.append(cliff_row)
        
        # Calculate cliff heights (vertical)
        cliff_heights = [[0 for _ in row] for row in cliffs]
        for col in range(pfw):
            prev_row = -1
            for row in range(len(cliffs) - 1, -1, -1):
                if cliffs[row][col] == 1 and prev_row == -1:
                    prev_row = row
                elif pf[row][col] != " " and prev_row != -1:
                    cliff_heights[prev_row][col] = (
                        (prev_row - row) ** self.params["cliff_h_e"]
                    ) * self.params["cliff_h_w"]
                if row == 0 and prev_row != -1 and pf[row][col] == " ":
                    cliff_heights[prev_row][col] = (
                        (prev_row - row + 1) ** self.params["cliff_h_e"]
                    ) * self.params["cliff_h_w"]
        
        cliff_heights_sum = sum([sum(row) for row in cliff_heights])
        
        # Calculate cliff lengths (horizontal)
        cliff_lengths = 0.0
        for row in range(len(cliffs)):
            prev_i = -1
            for i in range(pfw):
                if cliffs[row][i] > 0 and prev_i == -1:
                    prev_i = i
                elif cliffs[row][i] == 0 and prev_i != -1:
                    cliff_lengths += (i - prev_i) ** self.params["cliff_l_e"] * self.params["cliff_l_w"]
                if i == pfw - 1 and prev_i != -1 and cliffs[row][i] > 0:
                    cliff_lengths += (i - prev_i + 1) ** self.params["cliff_l_e"] * self.params["cliff_l_w"]
        
        return cliff_heights_sum, cliff_lengths

    def _check_game_over(self, playfield: list) -> bool:
        """Check if game is over (blocks in top 4 rows)."""
        top_rows = playfield[:4] if len(playfield) >= 4 else playfield
        return any(any(cell != " " for cell in row) for row in top_rows)

    def _playfield_hash(self, playfield: list) -> tuple:
        """Create hashable key from playfield for caching."""
        # Use top 16 rows for performance
        return tuple(tuple(row) for row in playfield[:16])

    def update_state(self, state: GameState) -> None:
        """Update AI's internal state (clears cache on significant changes).

        Args:
            state: New game state
        """
        # Clear cache if it gets too large
        if len(self._eval_cache) > 10000:
            self._eval_cache.clear()
