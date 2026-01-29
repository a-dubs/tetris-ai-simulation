#!/usr/bin/env python3
"""Main CLI entry point for Tetris AI Simulator."""

import argparse
import sys

from tetris.core.game_engine import GameEngine
from tetris.simulation.simulator import Simulator
from tetris.simulation.config import SimulationConfig
from tetris.simulation.factory import create_ai, create_renderer


def run_game(args):
    """Run a single game simulation."""
    config = SimulationConfig(
        ai_type=args.ai,
        headless=args.headless,
        max_moves=args.max_moves,
        max_time=args.max_time,
        initial_level=args.level,
        moves_per_second=args.mps,
        render_delay=args.render_delay,
    )

    engine = GameEngine()
    ai = create_ai(config.ai_type, config)
    renderer = create_renderer(config.headless)

    sim = Simulator(engine, ai, renderer, config)

    print(f"Starting simulation with {config.ai_type} AI...")
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
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (default: False, shows Pygame window)",
    )
    run_parser.add_argument(
        "--max-moves",
        type=int,
        default=10000,
        help="Maximum moves before stopping (default: 10000)",
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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        run_game(args)
    elif args.command == "train":
        train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
