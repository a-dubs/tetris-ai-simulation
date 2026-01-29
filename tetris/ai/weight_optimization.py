"""Weight optimization for FastAI heuristic parameters.

Provides multiple approaches to iteratively improve FastAI weights:
1. Evolutionary Strategy (CMA-ES) - gradient-free optimization
2. Bayesian Optimization - sample-efficient black-box optimization
3. Meta-RL - RL agent that learns to set weights
4. Hybrid RL - RL agent that uses FastAI as action space

Usage:
    # Evolutionary optimization
    python -m tetris.ai.weight_optimization evolve --runs 50 --games-per-run 20
    
    # Bayesian optimization
    python -m tetris.ai.weight_optimization bayesian --iterations 30 --games-per-iter 20
    
    # Meta-RL (learns weights via RL)
    python -m tetris.ai.weight_optimization meta_rl --timesteps 100000
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import random
import time

from tetris.ai.fast import FastAIPlayer
from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState
from tetris.simulation.simulator import Simulator
from tetris.simulation.config import SimulationConfig
from tetris.simulation.factory import create_renderer


@dataclass
class WeightBounds:
    """Bounds for each weight parameter."""
    # Stack height penalties
    m_stack_w: Tuple[float, float] = (-15.0, -1.0)
    m_stack_e: Tuple[float, float] = (0.5, 3.0)
    a_stack_w: Tuple[float, float] = (-10.0, -1.0)
    a_stack_e: Tuple[float, float] = (0.5, 3.0)
    
    # Cliff penalties
    cliff_l_w: Tuple[float, float] = (-10.0, -1.0)
    cliff_l_e: Tuple[float, float] = (0.5, 3.0)
    cliff_h_w: Tuple[float, float] = (-12.0, -1.0)
    cliff_h_e: Tuple[float, float] = (0.5, 3.0)
    
    # Stack danger
    stack_d_w: Tuple[float, float] = (-8.0, -1.0)
    stack_d_e: Tuple[float, float] = (0.5, 3.0)
    stack_d_thresh: Tuple[int, int] = (10, 20)
    
    # Score rewards
    score_w: Tuple[float, float] = (50.0, 200.0)
    tetris_bonus: Tuple[float, float] = (100.0, 500.0)
    
    # Game over penalty
    go_w: Tuple[float, float] = (-2000.0, -100.0)
    
    def get_param_names(self) -> List[str]:
        """Get list of parameter names."""
        return [
            "m_stack_w", "m_stack_e", "a_stack_w", "a_stack_e",
            "cliff_l_w", "cliff_l_e", "cliff_h_w", "cliff_h_e",
            "stack_d_w", "stack_d_e", "stack_d_thresh",
            "score_w", "tetris_bonus", "go_w",
        ]
    
    def get_bounds_array(self) -> np.ndarray:
        """Get bounds as numpy array for optimization."""
        bounds = []
        for name in self.get_param_names():
            low, high = getattr(self, name)
            bounds.append([low, high])
        return np.array(bounds)
    
    def clip_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clip parameters to valid bounds."""
        clipped = {}
        for name in self.get_param_names():
            value = params[name]
            low, high = getattr(self, name)
            if isinstance(value, float):
                clipped[name] = np.clip(value, low, high)
            else:  # int (stack_d_thresh)
                clipped[name] = int(np.clip(value, low, high))
        return clipped


@dataclass
class EvaluationResult:
    """Result from evaluating a set of weights."""
    params: Dict[str, Any]
    avg_score: float
    avg_lines: float
    avg_moves: float
    std_score: float
    std_lines: float
    num_games: int
    seed: Optional[int] = None
    min_score: Optional[float] = None
    min_lines: Optional[float] = None
    min_moves: Optional[float] = None
    
    def fitness(self, robust: bool = True) -> float:
        """Calculate fitness score (higher is better).
        
        Args:
            robust: If True, use robust fitness that considers mean, minimum,
                    and consistency. If False, use simple mean-only fitness.
        
        Returns:
            Fitness score (higher is better)
        """
        if not robust:
            # Simple mean-only fitness (original)
            score_component = self.avg_score / 10000.0  # ~10k is good
            lines_component = self.avg_lines / 50.0     # ~50 lines is good
            moves_component = self.avg_moves / 1000.0  # ~1000 moves is good
            return 0.5 * score_component + 0.4 * lines_component + 0.1 * moves_component
        
        # Robust fitness: balances mean, minimum, and consistency
        # Average performance (50%)
        avg_component = 0.5 * (
            0.5 * (self.avg_score / 10000.0) +
            0.4 * (self.avg_lines / 50.0) +
            0.1 * (self.avg_moves / 1000.0)
        )
        
        # Minimum performance (30%) - prevent terrible games
        min_component = 0.0
        if self.min_score is not None and self.min_lines is not None and self.min_moves is not None:
            min_score_norm = self.min_score / 10000.0
            min_lines_norm = self.min_lines / 50.0
            min_moves_norm = self.min_moves / 1000.0
            min_component = 0.3 * (
                0.5 * min_score_norm +
                0.4 * min_lines_norm +
                0.1 * min_moves_norm
            )
        
        # Consistency penalty (20%) - penalize high variance
        consistency_penalty = 0.0
        if self.avg_score > 0 and self.std_score > 0:
            cv_score = self.std_score / self.avg_score  # Coefficient of variation
            cv_lines = self.std_lines / self.avg_lines if self.avg_lines > 0 else 0.0
            # Penalize high variance (normalize CV to [0, 1] range, assuming CV < 2.0 is reasonable)
            consistency_penalty = -0.2 * min(1.0, (cv_score + cv_lines) / 2.0)
        
        return avg_component + min_component + consistency_penalty


# Module-level constant - WeightBounds never changes, reuse for performance
_DEFAULT_BOUNDS = WeightBounds()


def evaluate_weights(
    params: Dict[str, Any],
    num_games: int = 20,
    seed: Optional[int] = None,
    headless: bool = True,
    perf_log: Optional[Dict[str, List[float]]] = None,
) -> EvaluationResult:
    """Evaluate a set of weights by running full games.
    
    Plays complete games from start to game over with fixed weights.
    No weight adjustments occur during gameplay - weights are constant
    for the entire duration of each game.
    
    Args:
        params: Weight parameters for FastAI (fixed for entire game)
        num_games: Number of full games to run
        seed: Random seed for reproducibility
        headless: Whether to run headless
        perf_log: Optional dict to log performance timings
        
    Returns:
        EvaluationResult with statistics from completed games
    """
    if perf_log is None:
        perf_log = {}
    
    eval_start = time.perf_counter()
    scores = []
    lines = []
    moves = []
    
    # Clip params to valid bounds (use module-level constant)
    clip_start = time.perf_counter()
    params = _DEFAULT_BOUNDS.clip_params(params)
    if "clip_params_time" not in perf_log:
        perf_log["clip_params_time"] = []
    perf_log["clip_params_time"].append(time.perf_counter() - clip_start)
    
    # Create reusable objects (renderer and config are stateless for headless)
    setup_start = time.perf_counter()
    renderer = create_renderer(headless)
    config = SimulationConfig(
        ai_type="fast",  # Will be overridden
        headless=headless,
        max_moves=None,  # No limit - run until game over
        max_time=None,
        initial_level=1,
        moves_per_second=4,
    )
    if "setup_time" not in perf_log:
        perf_log["setup_time"] = []
    perf_log["setup_time"].append(time.perf_counter() - setup_start)
    
    # Track per-game timings
    game_times = []
    ai_creation_times = []
    engine_creation_times = []
    sim_run_times = []
    
    for game_idx in range(num_games):
        game_start = time.perf_counter()
        
        game_seed = seed + game_idx if seed is not None else None
        if game_seed is not None:
            random.seed(game_seed)
            np.random.seed(game_seed)
        
        # Create AI with these weights
        ai_start = time.perf_counter()
        ai = FastAIPlayer(params=params)
        ai_creation_times.append(time.perf_counter() - ai_start)
        
        # Create engine and simulator (engine has state, must be new each game)
        engine_start = time.perf_counter()
        engine = GameEngine()
        sim = Simulator(engine, ai, renderer, config)
        engine_creation_times.append(time.perf_counter() - engine_start)
        
        # Run simulation
        sim_start = time.perf_counter()
        result = sim.run()
        sim_run_times.append(time.perf_counter() - sim_start)
        
        scores.append(result.final_state.score)
        lines.append(result.final_state.lines_cleared)
        moves.append(result.moves_executed)
        
        game_times.append(time.perf_counter() - game_start)
    
    total_time = time.perf_counter() - eval_start
    
    # Log performance metrics
    perf_log["total_eval_time"] = total_time
    perf_log["num_games"] = num_games
    perf_log["avg_game_time"] = statistics.mean(game_times) if game_times else 0.0
    perf_log["avg_ai_creation_time"] = statistics.mean(ai_creation_times) if ai_creation_times else 0.0
    perf_log["avg_engine_creation_time"] = statistics.mean(engine_creation_times) if engine_creation_times else 0.0
    perf_log["avg_sim_run_time"] = statistics.mean(sim_run_times) if sim_run_times else 0.0
    perf_log["total_game_times"] = game_times
    
    return EvaluationResult(
        params=params,
        avg_score=statistics.mean(scores),
        avg_lines=statistics.mean(lines),
        avg_moves=statistics.mean(moves),
        std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        std_lines=statistics.stdev(lines) if len(lines) > 1 else 0.0,
        num_games=num_games,
        seed=seed,
        min_score=min(scores) if scores else None,
        min_lines=min(lines) if lines else None,
        min_moves=min(moves) if moves else None,
    )


def params_to_vector(params: Dict[str, Any], bounds: WeightBounds) -> np.ndarray:
    """Convert params dict to normalized vector [0, 1]."""
    vector = []
    for name in bounds.get_param_names():
        value = params[name]
        low, high = getattr(bounds, name)
        # Normalize to [0, 1]
        normalized = (value - low) / (high - low)
        vector.append(normalized)
    return np.array(vector)


def vector_to_params(vector: np.ndarray, bounds: WeightBounds) -> Dict[str, Any]:
    """Convert normalized vector [0, 1] to params dict."""
    params = {}
    param_names = bounds.get_param_names()
    for i, name in enumerate(param_names):
        normalized = np.clip(vector[i], 0.0, 1.0)
        low, high = getattr(bounds, name)
        value = normalized * (high - low) + low
        if name == "stack_d_thresh":
            params[name] = int(round(value))
        else:
            params[name] = float(value)
    return params


# ============================================================================
# Evolutionary Strategy (CMA-ES)
# ============================================================================

def optimize_evolutionary(
    initial_params: Optional[Dict[str, Any]] = None,
    num_runs: int = 50,
    games_per_run: int = 20,
    population_size: int = 10,
    seed: Optional[int] = None,
    output_file: Optional[str] = None,
    robust_fitness: bool = True,
) -> EvaluationResult:
    """Optimize weights using CMA-ES evolutionary strategy.
    
    Process: For each candidate weight set:
    1. Play full game(s) with fixed weights
    2. Get final score/performance
    3. Adjust weights based on performance
    4. Repeat
    
    No weight adjustments occur during gameplay.
    
    Args:
        initial_params: Starting parameters (uses defaults if None)
        num_runs: Number of optimization iterations
        games_per_run: Full games to play per weight evaluation
        population_size: CMA-ES population size
        seed: Random seed
        output_file: Path to save best params JSON
        robust_fitness: If True, use robust fitness (mean + minimum + consistency).
                       If False, use simple mean-only fitness.
        
    Returns:
        Best EvaluationResult found
    """
    try:
        import cma
    except ImportError:
        raise ImportError(
            "CMA-ES requires 'cma' package. Install with: uv pip install cma"
        )
    
    # Start from initial params or defaults
    if initial_params is None:
        initial_params = FastAIPlayer._get_improved_default_params()
    
    # Convert to normalized vector
    x0 = params_to_vector(initial_params, _DEFAULT_BOUNDS)
    
    # CMA-ES options
    sigma0 = 0.3  # Initial standard deviation (30% of range)
    options = {
        "maxiter": num_runs,
        "popsize": population_size,
        "seed": seed,
        "verb_disp": 1,
        "verb_log": 1,
    }
    
    best_result = None
    best_fitness = float("-inf")
    history = []
    perf_log = {
        "total_eval_times": [],
        "vector_times": [],
        "eval_weights_times": [],
        "fitness_times": [],
        "all_game_times": [],
    }
    eval_count = 0
    total_start_time = time.perf_counter()
    
    def objective(x: np.ndarray) -> float:
        """Objective function: negative fitness (CMA-ES minimizes)."""
        nonlocal best_result, best_fitness, eval_count
        
        eval_start = time.perf_counter()
        eval_count += 1
        
        vector_start = time.perf_counter()
        params = vector_to_params(x, _DEFAULT_BOUNDS)
        vector_time = time.perf_counter() - vector_start
        perf_log["vector_times"].append(vector_time)
        
        eval_weights_start = time.perf_counter()
        # Create fresh perf_log for this evaluation to avoid overwriting
        eval_perf_log = {}
        # Use different seed for each evaluation (seed + eval_count)
        # This ensures each evaluation uses different piece sequences
        eval_seed = seed + eval_count
        result = evaluate_weights(params, num_games=games_per_run, seed=eval_seed, perf_log=eval_perf_log)
        eval_weights_time = time.perf_counter() - eval_weights_start
        perf_log["eval_weights_times"].append(eval_weights_time)
        
        # Accumulate game times
        if "total_game_times" in eval_perf_log:
            perf_log["all_game_times"].extend(eval_perf_log["total_game_times"])
        
        fitness_start = time.perf_counter()
        # Use robust fitness (considers mean, minimum, consistency)
        fitness = result.fitness(robust=robust_fitness)
        fitness_time = time.perf_counter() - fitness_start
        perf_log["fitness_times"].append(fitness_time)
        
        total_eval_time = time.perf_counter() - eval_start
        perf_log["total_eval_times"].append(total_eval_time)
        
        history.append({
            "iteration": len(history),
            "fitness": fitness,
            "avg_score": result.avg_score,
            "avg_lines": result.avg_lines,
            "params": params,
            "eval_time": total_eval_time,
        })
        
        # Track best
        if fitness > best_fitness:
            best_fitness = fitness
            best_result = result
        
        # Log progress every 50 evaluations (less verbose)
        if eval_count % 50 == 0:
            avg_eval_time = statistics.mean([h["eval_time"] for h in history[-50:]])
            print(f"  [Progress] Eval #{eval_count}: avg_time={avg_eval_time:.3f}s per evaluation")
        
        return -fitness  # Minimize negative fitness
    
    # Generate random seed if not provided
    if seed is None:
        import secrets
        seed = secrets.randbelow(2**31)  # Random 32-bit integer
        print(f"Starting CMA-ES optimization (using random seed: {seed})...")
    else:
        print(f"Starting CMA-ES optimization (using seed: {seed})...")
    print(f"  Runs: {num_runs}")
    print(f"  Games per run: {games_per_run}")
    print(f"  Population size: {population_size}")
    print(f"  Robust fitness: {robust_fitness} (mean{' + min + consistency' if robust_fitness else ''})")
    print(f"  Initial params: {initial_params}")
    print()
    
    # Update CMA-ES options with actual seed (was using original seed param)
    options["seed"] = seed
    
    # Run CMA-ES
    cma_start_time = time.perf_counter()
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    es.optimize(objective)
    cma_total_time = time.perf_counter() - cma_start_time
    
    print(f"\nOptimization complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    if best_result:
        print(f"Best avg score: {best_result.avg_score:.1f}")
        print(f"Best avg lines: {best_result.avg_lines:.1f}")
        print(f"Best params:")
        for key, value in best_result.params.items():
            print(f"  {key}: {value}")
    
    # Print performance summary
    if perf_log and perf_log["total_eval_times"]:
        total_eval_time = sum(perf_log["total_eval_times"])
        total_games = len(perf_log["all_game_times"])
        
        print(f"\n{'='*70}")
        print("Performance Summary")
        print(f"{'='*70}")
        print(f"Total time: {cma_total_time:.2f}s")
        print(f"Total evaluations: {eval_count}")
        print(f"Total games evaluated: {total_games}")
        if perf_log["total_eval_times"]:
            print(f"Avg time per evaluation: {statistics.mean(perf_log['total_eval_times']):.3f}s")
        if perf_log["all_game_times"]:
            print(f"Avg time per game: {statistics.mean(perf_log['all_game_times']):.4f}s")
        print(f"{'='*70}")
    
    # Save best params
    if output_file and best_result:
        with open(output_file, 'w') as f:
            json.dump({
                "params": best_result.params,
                "fitness": best_fitness,
                "avg_score": best_result.avg_score,
                "avg_lines": best_result.avg_lines,
                "avg_moves": best_result.avg_moves,
                "num_games": best_result.num_games,
            }, f, indent=2)
        print(f"\nSaved best params to {output_file}")
    
    return best_result


# ============================================================================
# Bayesian Optimization
# ============================================================================

def optimize_bayesian(
    initial_params: Optional[Dict[str, Any]] = None,
    num_iterations: int = 30,
    games_per_iter: int = 20,
    seed: Optional[int] = None,
    output_file: Optional[str] = None,
) -> EvaluationResult:
    """Optimize weights using Bayesian Optimization.
    
    Process: For each candidate weight set:
    1. Play full game(s) with fixed weights
    2. Get final score/performance
    3. Use Bayesian optimization to select next weights
    4. Repeat
    
    No weight adjustments occur during gameplay.
    
    Args:
        initial_params: Starting parameters
        num_iterations: Number of BO iterations
        games_per_iter: Full games to play per weight evaluation
        seed: Random seed
        output_file: Path to save best params JSON
        
    Returns:
        Best EvaluationResult found
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
    except ImportError:
        raise ImportError(
            "Bayesian Optimization requires 'scikit-optimize' package. "
            "Install with: uv pip install scikit-optimize"
        )
    
    # Define search space
    dimensions = []
    param_names = _DEFAULT_BOUNDS.get_param_names()
    
    for name in param_names:
        low, high = getattr(_DEFAULT_BOUNDS, name)
        if name == "stack_d_thresh":
            dimensions.append(Integer(low, high, name=name))
        else:
            dimensions.append(Real(low, high, name=name))
    
    # Initial params
    if initial_params is None:
        initial_params = FastAIPlayer._get_improved_default_params()
    
    x0 = [[initial_params[name] for name in param_names]]
    
    best_result = None
    best_fitness = float("-inf")
    history = []
    
    # Generate random seed if not provided
    if seed is None:
        import secrets
        seed = secrets.randbelow(2**31)
        print(f"Starting Bayesian Optimization (using random seed: {seed})...")
    else:
        print(f"Starting Bayesian Optimization (using seed: {seed})...")
    
    iter_count = 0
    
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        """Objective function: negative fitness."""
        nonlocal best_result, best_fitness, iter_count
        
        # Use different seed for each iteration (seed + iter_count)
        # This ensures each iteration evaluates with different piece sequences
        eval_seed = seed + iter_count
        iter_count += 1
        result = evaluate_weights(params, num_games=games_per_iter, seed=eval_seed)
        fitness = result.fitness(robust=True)
        
        history.append({
            "iteration": len(history),
            "fitness": fitness,
            "avg_score": result.avg_score,
            "avg_lines": result.avg_lines,
            "params": params,
        })
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_result = result
        
        return -fitness  # Minimize negative fitness
    
    print(f"Starting Bayesian Optimization...")
    print(f"  Iterations: {num_iterations}")
    print(f"  Games per iteration: {games_per_iter}")
    print(f"  Initial params: {initial_params}")
    print()
    
    # Run BO
    result_bo = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=num_iterations,
        x0=x0,
        random_state=seed,
        n_initial_points=5,
        acq_func="EI",  # Expected Improvement
    )
    
    print(f"\nOptimization complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    if best_result:
        print(f"Best avg score: {best_result.avg_score:.1f}")
        print(f"Best avg lines: {best_result.avg_lines:.1f}")
        print(f"Best params:")
        for key, value in best_result.params.items():
            print(f"  {key}: {value}")
    
    # Save best params
    if output_file and best_result:
        with open(output_file, 'w') as f:
            json.dump({
                "params": best_result.params,
                "fitness": best_fitness,
                "avg_score": best_result.avg_score,
                "avg_lines": best_result.avg_lines,
                "avg_moves": best_result.avg_moves,
                "num_games": best_result.num_games,
            }, f, indent=2)
        print(f"\nSaved best params to {output_file}")
    
    return best_result


# ============================================================================
# Meta-RL: RL agent that learns to set weights
# ============================================================================

def optimize_meta_rl(
    timesteps: int = 100000,
    games_per_eval: int = 5,
    seed: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Use RL to learn optimal weights (meta-learning).
    
    Episodic RL where each episode:
    1. Agent sets weights (action)
    2. Play full game(s) with those fixed weights
    3. Get final score/performance
    4. Reward based on performance
    5. Episode ends, next episode starts
    
    No weight adjustments occur during gameplay - weights are fixed
    for the entire duration of each game.
    
    Creates an RL environment where:
    - State: Previous weights + previous performance
    - Action: New weight values to try
    - Reward: Performance from playing full game(s) with those weights
    
    Args:
        timesteps: Total training timesteps
        games_per_eval: Full games to play per weight evaluation
        seed: Random seed
        output_file: Path to save best params JSON
        
    Returns:
        Best parameters found
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        RL_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "Meta-RL requires RL dependencies. "
            "Install with: uv pip install gymnasium stable-baselines3"
        )
    
    param_names = _DEFAULT_BOUNDS.get_param_names()
    n_params = len(param_names)
    
    class WeightOptimizationEnv(gym.Env):
        """RL environment for weight optimization.
        
        Episodic: Each episode is:
        1. Set weights (from action)
        2. Play full game(s) with those weights
        3. Get final score/performance
        4. Reward based on performance
        5. Episode done, reset for next episode
        """
        
        def __init__(self):
            super().__init__()
            
            # Action: weight values (normalized [0, 1])
            # Agent directly sets weights, not adjustments
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(n_params,), dtype=np.float32
            )
            
            # State: previous weights + previous performance (for context)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(n_params + 3,), dtype=np.float32
            )
            
            # Previous episode's weights and performance
            self.prev_params = FastAIPlayer._get_improved_default_params()
            self.prev_vector = params_to_vector(self.prev_params, _DEFAULT_BOUNDS)
            self.prev_score = 0.0
            self.prev_lines = 0.0
            self.prev_moves = 0.0
            self.episode_count = 0
        
        def reset(self, seed=None, options=None):
            """Reset environment for new episode."""
            super().reset(seed=seed)
            self.episode_count += 1
            # Keep previous episode's info for observation
            # (weights will be set by action in first step)
            return self._get_obs(), {}
        
        def step(self, action):
            """Take step: set weights, play full game(s), get score.
            
            Action sets the weights directly. Then we play full game(s)
            with those weights and return the final performance as reward.
            """
            # Action directly sets weights (normalized [0, 1])
            new_vector = np.clip(action, 0.0, 1.0)
            new_params = vector_to_params(new_vector, _DEFAULT_BOUNDS)
            
            # Play full game(s) with these weights
            # No weight changes during gameplay - weights are fixed for entire game(s)
            # Use different seed for each episode (seed + episode_count)
            # This ensures each episode evaluates with different piece sequences
            # Note: seed is captured from outer scope
            episode_seed = seed + self.episode_count
            result = evaluate_weights(
                new_params, num_games=games_per_eval, seed=episode_seed, headless=True
            )
            
            # Calculate reward from final game performance
            fitness = result.fitness(robust=True)
            reward = fitness * 10.0  # Scale up for RL
            
            # Update previous episode info for next observation
            self.prev_params = new_params
            self.prev_vector = new_vector
            self.prev_score = result.avg_score
            self.prev_lines = result.avg_lines
            self.prev_moves = result.avg_moves
            
            # Episode is done after each evaluation
            # Next episode will start with reset(), then new weights from action
            done = True
            
            return self._get_obs(), reward, done, False, {
                "fitness": fitness,
                "avg_score": result.avg_score,
                "avg_lines": result.avg_lines,
                "avg_moves": result.avg_moves,
                "params": new_params,
            }
        
        def _get_obs(self):
            """Get current observation: previous weights + previous performance."""
            # Normalize performance metrics
            score_norm = np.clip(self.prev_score / 20000.0, 0.0, 1.0)
            lines_norm = np.clip(self.prev_lines / 100.0, 0.0, 1.0)
            moves_norm = np.clip(self.prev_moves / 2000.0, 0.0, 1.0)
            
            obs = np.concatenate([
                self.prev_vector,
                [score_norm, lines_norm, moves_norm]
            ]).astype(np.float32)
            
            return obs
    
    # Generate random seed if not provided
    if seed is None:
        import secrets
        seed = secrets.randbelow(2**31)
        print(f"Starting Meta-RL optimization (using random seed: {seed})...")
    else:
        print(f"Starting Meta-RL optimization (using seed: {seed})...")
    print(f"  Timesteps: {timesteps}")
    print(f"  Games per eval: {games_per_eval}")
    print()
    
    # Create environment (seed is captured from outer scope in closure)
    env = WeightOptimizationEnv()
    
    # Create PPO agent
    # Note: Since episodes are very short (one evaluation each),
    # we use smaller n_steps to update more frequently
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=32,  # Smaller since episodes are single evaluations
        batch_size=32,
        n_epochs=10,
        gamma=0.99,  # Still use discounting for multi-step learning
        verbose=1,
        seed=seed,
    )
    
    # Track best during training
    best_params = None
    best_fitness = float("-inf")
    
    def track_best_callback(locals_, globals_):
        """Callback to track best weights during training."""
        nonlocal best_params, best_fitness
        
        if "infos" in locals_:
            for info in locals_["infos"]:
                if info and "params" in info:
                    fitness = info.get("fitness", 0.0)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_params = info["params"].copy()
        return True
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=track_best_callback,
        progress_bar=True,
    )
    
    # Final evaluation
    if best_params:
        # Use a high seed offset for final evaluation to ensure different sequences
        final_seed = seed + 10000 if seed is not None else None
        final_result = evaluate_weights(
            best_params, num_games=games_per_eval * 4, seed=final_seed
        )
        print(f"\nTraining complete!")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Best avg score: {final_result.avg_score:.1f}")
        print(f"Best avg lines: {final_result.avg_lines:.1f}")
        print(f"Best params:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Save
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    "params": best_params,
                    "fitness": best_fitness,
                    "avg_score": final_result.avg_score,
                    "avg_lines": final_result.avg_lines,
                    "avg_moves": final_result.avg_moves,
                }, f, indent=2)
            print(f"\nSaved best params to {output_file}")
        
        return best_params
    
    return env.current_params


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize FastAI heuristic weights"
    )
    subparsers = parser.add_subparsers(dest="method", help="Optimization method")
    
    # Evolutionary
    evolve_parser = subparsers.add_parser("evolve", help="CMA-ES evolutionary optimization")
    evolve_parser.add_argument("--runs", type=int, default=50, help="Number of iterations")
    evolve_parser.add_argument("--games-per-run", type=int, default=20, help="Games per evaluation")
    evolve_parser.add_argument("--population-size", type=int, default=10, help="CMA-ES population size")
    evolve_parser.add_argument("--seed", type=int, default=None, 
                               help="Random seed (default: random each run for better exploration)")
    evolve_parser.add_argument("--output", type=str, help="Output JSON file")
    evolve_parser.add_argument("--no-robust-fitness", action="store_true", 
                               help="Use simple mean-only fitness instead of robust (mean+min+consistency)")
    
    # Bayesian
    bayesian_parser = subparsers.add_parser("bayesian", help="Bayesian optimization")
    bayesian_parser.add_argument("--iterations", type=int, default=30, help="Number of iterations")
    bayesian_parser.add_argument("--games-per-iter", type=int, default=20, help="Games per iteration")
    bayesian_parser.add_argument("--seed", type=int, default=None,
                                 help="Random seed (default: random each run for better exploration)")
    bayesian_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Meta-RL
    meta_rl_parser = subparsers.add_parser("meta_rl", help="Meta-RL optimization")
    meta_rl_parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    meta_rl_parser.add_argument("--games-per-eval", type=int, default=5, help="Games per evaluation")
    meta_rl_parser.add_argument("--seed", type=int, default=None,
                                help="Random seed (default: random each run for better exploration)")
    meta_rl_parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    if args.method == "evolve":
        optimize_evolutionary(
            num_runs=args.runs,
            games_per_run=args.games_per_run,
            population_size=args.population_size,
            seed=args.seed,
            output_file=args.output,
            robust_fitness=not args.no_robust_fitness,
        )
    elif args.method == "bayesian":
        optimize_bayesian(
            num_iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            seed=args.seed,
            output_file=args.output,
        )
    elif args.method == "meta_rl":
        optimize_meta_rl(
            timesteps=args.timesteps,
            games_per_eval=args.games_per_eval,
            seed=args.seed,
            output_file=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
