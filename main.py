#!/usr/bin/env python3
"""Main CLI entry point for Tetris AI Simulator."""

import argparse
import sys
import json
import random
import statistics

from tetris.core.game_engine import GameEngine
from tetris.simulation.simulator import Simulator
from tetris.simulation.config import SimulationConfig
from tetris.simulation.factory import create_ai, create_renderer


def _apply_seed(seed: int | None):
    if seed is not None:
        random.seed(seed)


def run_game(args):
    """Run a single game simulation."""
    from pathlib import Path
    
    _apply_seed(args.seed)
    
    # Load params from file if provided
    ai_params = None
    if args.params_file:
        params_path = Path(args.params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Params file not found: {args.params_file}")
        with open(params_path, "r") as f:
            params_data = json.load(f)
            # Support both direct params dict and wrapped format
            if "params" in params_data:
                ai_params = {"params": params_data["params"]}
            else:
                ai_params = {"params": params_data}
        print(f"Loaded AI params from: {args.params_file}")
    
    config = SimulationConfig(
        ai_type=args.ai,
        headless=args.headless,
        max_moves=args.max_moves,
        max_time=args.max_time,
        initial_level=args.level,
        moves_per_second=args.mps,
        render_delay=args.render_delay,
        ai_params=ai_params,
    )

    engine = GameEngine()
    ai = create_ai(config.ai_type, config)
    renderer = create_renderer(config.headless)

    sim = Simulator(engine, ai, renderer, config)

    print(f"Starting simulation with {config.ai_type} AI...")
    if config.max_moves:
        print(f"Max moves: {config.max_moves}")
    if config.max_time:
        print(f"Max time: {config.max_time}s")
    print()

    result = sim.run()

    print()
    print("=" * 70)
    print("Simulation Complete")
    print("=" * 70)
    print(f"  Score:     {result.final_state.score:,}")
    print(f"  Level:     {result.final_state.level}")
    print(f"  Lines:     {result.final_state.lines_cleared}")
    print(f"  Moves:     {result.moves_executed:,}")
    print(f"  Time:      {result.time_elapsed:.2f}s")
    print(f"  Game Over: {result.game_over}")
    print("=" * 70)

    return result


def train(args):
    """Train AI (placeholder for future implementation)."""
    print("Training functionality not yet implemented.")
    print("This will be added in a future update.")
    return None


def _run_single_simulation(config: SimulationConfig) -> tuple:
    """Run a single simulation and return result plus runtime info."""
    import time as time_module
    
    start_time = time_module.perf_counter()
    engine = GameEngine()
    ai = create_ai(config.ai_type, config)
    renderer = create_renderer(config.headless)
    sim = Simulator(engine, ai, renderer, config)
    result = sim.run()
    execution_time = time_module.perf_counter() - start_time
    
    # Add execution time to result (store as attribute for now)
    result.execution_time = execution_time
    return result


def run_benchmark(args):
    """Run multiple headless simulations and summarize results."""
    import json
    from pathlib import Path
    
    ai_types = args.ai or ["random"]
    runs = args.runs
    if runs <= 0:
        raise ValueError("runs must be >= 1")

    output_mode = args.format
    if args.output_file and output_mode != "json":
        raise ValueError("--output-file requires --format json")
    if output_mode == "text":
        print(f"Benchmark runs: {runs}")
        print()

    # Load params from file if provided
    ai_params = None
    if args.params_file:
        params_path = Path(args.params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Params file not found: {args.params_file}")
        with open(params_path, "r") as f:
            params_data = json.load(f)
            # Support both direct params dict and wrapped format
            if "params" in params_data:
                ai_params = {"params": params_data["params"]}
            else:
                ai_params = {"params": params_data}
        if output_mode == "text":
            print(f"Loaded AI params from: {args.params_file}")
            print()

    summary_payload = {
        "runs": runs,
        "max_moves": args.max_moves,
        "max_time": args.max_time,
        "level": args.level,
        "mps": args.mps,
        "seed": args.seed,
        "params_file": args.params_file,
        "ais": [],
    }

    for ai_type in ai_types:
        results = []
        run_payloads = []
        for run_idx in range(1, runs + 1):
            if args.seed is not None:
                _apply_seed(args.seed + run_idx)

            config = SimulationConfig(
                ai_type=ai_type,
                headless=True,
                max_moves=args.max_moves,
                max_time=args.max_time,
                initial_level=args.level,
                moves_per_second=args.mps,
                render_delay=0.0,
                ai_params=ai_params,
            )

            result = _run_single_simulation(config)
            results.append(result)

            run_payload = {
                "run": run_idx,
                "score": result.final_state.score,
                "lines": result.final_state.lines_cleared,
                "level": result.final_state.level,
                "moves": result.moves_executed,
                "time_seconds": round(result.time_elapsed, 4),
                "game_over": result.game_over,
            }
            run_payloads.append(run_payload)

            if output_mode == "text":
                exec_time = getattr(result, 'execution_time', None)
                exec_str = f" exec={exec_time:.2f}s" if exec_time is not None else ""
                print(
                    f"{ai_type:>6} AI run {run_idx:>2}/{runs}: "
                    f"score={result.final_state.score:,} "
                    f"lines={result.final_state.lines_cleared} "
                    f"level={result.final_state.level} "
                    f"moves={result.moves_executed:,} "
                    f"sim_time={result.time_elapsed:.2f}s{exec_str} "
                    f"over={result.game_over}"
                )

        scores = [r.final_state.score for r in results]
        lines = [r.final_state.lines_cleared for r in results]
        levels = [r.final_state.level for r in results]
        moves = [r.moves_executed for r in results]
        sim_times = [r.time_elapsed for r in results]
        exec_times = [getattr(r, 'execution_time', None) for r in results]
        exec_times = [t for t in exec_times if t is not None]
        overs = sum(1 for r in results if r.game_over)

        def fmt_float(values: list[float]) -> str:
            return f"{statistics.mean(values):.2f}"

        summary_payload["ais"].append(
            {
                "ai": ai_type,
                "runs": run_payloads,
                "summary": {
                    "score": {
                        "avg": float(f"{statistics.mean(scores):.2f}"),
                        "min": min(scores),
                        "max": max(scores),
                    },
                    "lines": {
                        "avg": float(f"{statistics.mean(lines):.2f}"),
                        "min": min(lines),
                        "max": max(lines),
                    },
                    "level": {
                        "avg": float(f"{statistics.mean(levels):.2f}"),
                        "min": min(levels),
                        "max": max(levels),
                    },
                    "moves": {
                        "avg": float(f"{statistics.mean(moves):.2f}"),
                        "min": min(moves),
                        "max": max(moves),
                    },
                    "sim_time_seconds": {
                        "avg": float(f"{statistics.mean(sim_times):.2f}"),
                        "min": float(f"{min(sim_times):.2f}"),
                        "max": float(f"{max(sim_times):.2f}"),
                    },
                    "exec_time_seconds": {
                        "avg": float(f"{statistics.mean(exec_times):.2f}") if exec_times else None,
                        "min": float(f"{min(exec_times):.2f}") if exec_times else None,
                        "max": float(f"{max(exec_times):.2f}") if exec_times else None,
                    },
                    "game_over": {"count": overs, "total": runs},
                },
            }
        )

        if output_mode == "text":
            print()
            print(f"{ai_type} summary")
            print("-" * 70)
            print(
                f"  score avg={fmt_float(scores)} "
                f"min={min(scores):,} max={max(scores):,}"
            )
            print(
                f"  lines avg={fmt_float(lines)} "
                f"min={min(lines)} max={max(lines)}"
            )
            print(
                f"  level avg={fmt_float(levels)} "
                f"min={min(levels)} max={max(levels)}"
            )
            print(
                f"  moves avg={fmt_float(moves)} "
                f"min={min(moves):,} max={max(moves):,}"
            )
            print(
                f"  sim_time  avg={fmt_float(sim_times)}s "
                f"min={min(sim_times):.2f}s max={max(sim_times):.2f}s"
            )
            if exec_times:
                print(
                    f"  exec_time avg={fmt_float(exec_times)}s "
                    f"min={min(exec_times):.2f}s max={max(exec_times):.2f}s"
                )
            print(f"  game over {overs}/{runs}")
            print("=" * 70)
            print()

    if output_mode == "json":
        payload = json.dumps(summary_payload, indent=2, sort_keys=False)
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            print(payload)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tetris AI Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single game with random AI
  python main.py run --ai random

  # Run with greedy AI, limit to 1000 moves
  python main.py run --ai greedy --max-moves 1000

  # Run with fast AI, level 5, 60 moves per second
  python main.py run --ai fast --level 5 --mps 60
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single game simulation")
    run_parser.add_argument(
        "--ai",
        choices=["random", "greedy", "fast"],
        default="random",
        help="AI type to use (default: random)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible runs (default: none)",
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (default: False, shows Pygame window)",
    )
    run_parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum moves before stopping (default: no limit, runs until game over)",
    )
    run_parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Maximum time in seconds (default: no limit)",
    )
    run_parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Starting level (default: 1)",
    )
    run_parser.add_argument(
        "--mps",
        type=int,
        default=4,
        help="Moves per second (default: 4)",
    )
    run_parser.add_argument(
        "--render-delay",
        type=float,
        default=0.05,
        help="Delay between renders in seconds for GUI (default: 0.05)",
    )
    run_parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="JSON file containing AI parameters (for fast/greedy AI)",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train AI (not yet implemented)")
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for training (default: 10)",
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark AIs headlessly")
    bench_parser.add_argument(
        "--ai",
        choices=["random", "greedy", "fast"],
        action="append",
        help="AI type to benchmark (can be specified multiple times)",
    )
    bench_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    bench_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Write JSON output to a file (default: stdout)",
    )
    bench_parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per AI (default: 10)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed for reproducible runs (default: none)",
    )
    bench_parser.add_argument(
        "--max-moves",
        type=int,
        default=10000,
        help="Maximum moves before stopping (default: 10000)",
    )
    bench_parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Maximum time in seconds (default: no limit)",
    )
    bench_parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Starting level (default: 1)",
    )
    bench_parser.add_argument(
        "--mps",
        type=int,
        default=4,
        help="Moves per second (default: 4)",
    )
    bench_parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="JSON file containing AI parameters (for fast/greedy AI)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_game(args)
    elif args.command == "train":
        train(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
