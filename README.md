# Tetris AI Simulation

A Tetris simulation framework for developing, testing, and training AI agents. Originally created in 2020, recently refactored with modern Python architecture.

## Features

- **Modular Architecture**: Clean separation of game logic, AI, and rendering
- **Multiple AI Implementations**: Random, Greedy (recursive search), and Fast (direct evaluation)
- **Headless Simulation**: Run fast batch simulations without GUI
- **Comprehensive Testing**: 64+ tests with 88% code coverage
- **Type-Safe**: Full type hints throughout
- **Modern Tooling**: Uses `uv` for dependency management, `ruff` for linting, `pytest` for testing

## Quick Start

### Installation

```bash
# Install dependencies using uv
uv sync

# Or install manually
pip install -e .
```

### Running Simulations

```bash
# Run a single game with random AI
python main.py run --ai random

# Run with greedy AI, limit to 1000 moves
python main.py run --ai greedy --max-moves 1000

# Run with fast AI, level 5, 60 moves per second
python main.py run --ai fast --level 5 --mps 60

# See all options
python main.py run --help
```

### CLI Options

```bash
python main.py run [OPTIONS]

Options:
  --ai {random,greedy,fast}  AI type to use (default: random)
  --headless                  Run without GUI (default: True)
  --max-moves INT            Maximum moves before stopping (default: 10000)
  --max-time FLOAT           Maximum time in seconds (default: no limit)
  --level INT                Starting level (default: 1)
  --mps INT                  Moves per second (default: 4)
```

## Architecture

The codebase is organized into clear modules:

```
tetris/
├── core/           # Core game logic (no AI, no rendering)
│   ├── game_state.py      # Immutable game state
│   ├── game_engine.py     # Game mechanics (moves, placement, line clearing)
│   ├── tetrimino.py       # Tetris piece representation
│   └── constants.py       # Game constants
│
├── ai/             # AI implementations
│   ├── base.py            # AIPlayer abstract base class
│   ├── random.py          # Random AI (baseline)
│   ├── greedy.py          # Recursive search AI
│   └── fast.py            # Direct evaluation AI
│
├── render/         # Rendering (visualization)
│   ├── base.py           # Renderer abstract base class
│   └── headless.py       # No-op renderer for headless runs
│
└── simulation/     # Simulation orchestration
    ├── simulator.py      # Main simulator class
    ├── config.py         # Configuration dataclass
    └── factory.py        # Factory functions for creating components
```

## Usage Examples

### Programmatic Usage

```python
from tetris.core.game_engine import GameEngine
from tetris.ai.random import RandomAIPlayer
from tetris.render.headless import HeadlessRenderer
from tetris.simulation.simulator import Simulator
from tetris.simulation.config import SimulationConfig

# Create components
config = SimulationConfig(
    ai_type="random",
    max_moves=1000,
    headless=True
)
engine = GameEngine()
ai = RandomAIPlayer()
renderer = HeadlessRenderer()

# Run simulation
sim = Simulator(engine, ai, renderer, config)
result = sim.run()

print(f"Score: {result.final_state.score}")
print(f"Level: {result.final_state.level}")
print(f"Moves: {result.moves_executed}")
```

### Creating Custom AI

```python
from tetris.ai.base import AIPlayer
from tetris.core.game_state import GameState

class MyCustomAI(AIPlayer):
    def get_move_sequence(self, state: GameState) -> list[str]:
        # Your AI logic here
        return ["left", "left", "drop"]
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=tetris --cov-report=html

# Run specific test file
uv run pytest tests/test_game_engine.py
```

### Linting

```bash
# Check code style
uv run ruff check .

# Format code
uv run ruff format .
```

### Type Checking

```bash
uv run mypy tetris/
```

## Project History

**Original Project (2020)**
- Created an accurate Tetris simulation using Python and Tkinter
- Developed two AI algorithms (greedy recursive search and fast direct evaluation)
- Both AIs showed inconsistent performance but could clear levels 9-13
- Less complex algorithm was faster with near-identical performance

**Refactoring (2025)**
- Complete architectural overhaul with modern Python patterns
- Separated game logic from AI and rendering
- Added comprehensive test suite
- Created unified CLI interface
- Improved modularity and maintainability

## Project Status

**Status**: Active Development  
**Progress**: Core refactoring complete, training functionality pending

## License

See LICENSE file for details.
