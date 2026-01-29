# Refactoring Plan: From Scripts to Architecture

## Executive Summary

Current codebase has **6+ ad-hoc scripts** duplicating game logic, **global state everywhere**, and **tight coupling** between components. This plan outlines a step-by-step refactor to a clean, modular architecture.

---

## Current Issues (Detailed)

### Issue 1: Global State Anti-Pattern
**Location**: `tetris_sim.py` lines 77-94

```python
# BAD: Global state
playfield = [[]]
active_tet = None
next_tet = Tetrimino('L', y=playfield_height - 3)
scoreboard = {}
ai = TetrisAI(playfield, active_tet, next_tet, 3, 1)
```

**Problems**:
- Can't run multiple games
- Can't test in isolation
- State mutations are hidden
- Thread-unsafe

**Solution**: Encapsulate in `GameEngine` class

---

### Issue 2: Duplicated Game Loop Logic
**Locations**: 
- `run_game.py` lines 75-131
- `run_game_random.py` lines 76-133  
- `run_game_fast_ai.py` lines 76-133
- `tetris_sim.py` `run_trial()` lines 441-484

**Problems**:
- Same game loop logic copy-pasted 4+ times
- Bug fixes need to be applied in multiple places
- Inconsistent behavior

**Solution**: Single `Simulator.run()` method

---

### Issue 3: Mixed Responsibilities
**Location**: `TetrisAI` class

**Current**: `TetrisAI` contains:
- Game mechanics (`gravity()`, `valid_location()`, `place_tetrimino()`)
- AI logic (`get_best_moves()`, `greedy_heuristic_evaluation()`)
- Static utility methods

**Problems**:
- Can't test game mechanics without AI
- Can't swap AI implementations easily
- Violates Single Responsibility Principle

**Solution**: 
- Move game mechanics → `GameEngine`
- Keep only AI logic → `AIPlayer` implementations

---

### Issue 4: No Clear Entry Point
**Current**: Multiple ways to run:
- `python run_game.py`
- `python run_game_random.py`
- `python run_game_fast_ai.py`
- `run_trial()` function
- `train()` function

**Problems**:
- Users don't know what to call
- No consistent interface
- Hard to script/automate

**Solution**: Single CLI with subcommands

---

## Proposed Structure

```
tetris_ai_simulation/
├── tetris/                          # Main package
│   ├── __init__.py
│   │
│   ├── core/                        # Core game logic (no AI)
│   │   ├── __init__.py
│   │   ├── game_engine.py          # Game state management
│   │   ├── game_state.py           # Immutable state dataclass
│   │   ├── tetrimino.py            # Tetrimino (refactored)
│   │   └── constants.py            # Game constants
│   │
│   ├── ai/                          # AI implementations
│   │   ├── __init__.py
│   │   ├── base.py                 # AIPlayer ABC
│   │   ├── greedy.py               # Current recursive AI
│   │   ├── fast.py                 # Fast direct eval AI
│   │   ├── random.py               # Random AI
│   │   └── heuristics.py           # Shared heuristics
│   │
│   ├── render/                      # Rendering (GUI/headless)
│   │   ├── __init__.py
│   │   ├── base.py                 # Renderer ABC
│   │   ├── headless.py             # No-op renderer
│   │   └── tkinter_gui.py          # GUI renderer
│   │
│   └── simulation/                 # Simulation orchestration
│       ├── __init__.py
│       ├── simulator.py            # Main simulator class
│       └── config.py               # Configuration
│
├── scripts/                         # CLI scripts
│   └── cli.py                      # Single CLI entry point
│
├── tests/                          # Unit tests
│   ├── test_game_engine.py
│   ├── test_ai.py
│   └── test_simulation.py
│
└── main.py                         # CLI entry point
```

---

## Step-by-Step Implementation

### Step 0: Project Setup & Tooling
**Files**: `pyproject.toml`, `.ruff.toml` (optional), `.python-version`

**Setup with uv**:
```bash
# Initialize uv project
uv init --no-readme

# Add dependencies
uv add pytest pytest-cov
uv add --dev ruff mypy
```

**File**: `pyproject.toml`
```toml
[project]
name = "tetris-ai-simulation"
version = "0.1.0"
description = "Tetris AI simulation and training framework"
requires-python = ">=3.10"
dependencies = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py310"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--cov=tetris",
    "--cov-report=term-missing",
    "--cov-report=html",
]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
disallow_incomplete_defs = false
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
```

**File**: `.python-version`
```
3.10
```

**File**: `.gitignore` (update)
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
.uv/
```

### Step 1: Create Core Package Structure
```bash
mkdir -p tetris/{core,ai,render,simulation}
touch tetris/__init__.py
touch tetris/core/__init__.py
touch tetris/ai/__init__.py
touch tetris/render/__init__.py
touch tetris/simulation/__init__.py
```

### Step 2: Extract Game State
**File**: `tetris/core/game_state.py`

```python
from dataclasses import dataclass, field
from typing import List
from tetris.core.tetrimino import Tetrimino

@dataclass(frozen=True)
class GameState:
    """Immutable game state."""
    playfield: List[List[str]]
    active_tetrimino: Tetrimino
    next_tetrimino: Tetrimino
    bag: List[str] = field(default_factory=list)
    score: int = 0
    level: int = 1
    lines_cleared: int = 0
    game_over: bool = False
    time_elapsed: float = 0.0
    
    def with_playfield(self, playfield: List[List[str]]) -> 'GameState':
        """Return new state with updated playfield."""
        return dataclass.replace(self, playfield=playfield)
    
    # ... other with_* methods for immutability
```

### Step 3: Create GameEngine
**File**: `tetris/core/game_engine.py`

```python
from tetris.core.game_state import GameState
from tetris.core.constants import PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT
from tetris.core.tetrimino import Tetrimino

class GameEngine:
    """Pure game logic - no AI, no rendering."""
    
    def __init__(self, initial_state: GameState = None):
        self.state = initial_state or self._create_initial_state()
    
    def apply_move(self, move: str) -> GameState:
        """Apply a move, return new state."""
        # Pure function - no side effects
        new_tet = self._apply_move_to_tetrimino(self.state.active_tetrimino, move)
        return self.state.with_active_tetrimino(new_tet)
    
    def tick(self, delta_time: float) -> GameState:
        """Advance game by delta_time."""
        # Apply gravity, check landing, spawn, etc.
        # Return new state
        pass
    
    def _create_initial_state(self) -> GameState:
        """Create initial game state."""
        pass
```

### Step 4: Create AI Base Class
**File**: `tetris/ai/base.py`

```python
from abc import ABC, abstractmethod
from tetris.core.game_state import GameState

class AIPlayer(ABC):
    """Abstract base class for AI players."""
    
    @abstractmethod
    def get_move_sequence(self, state: GameState) -> List[str]:
        """
        Return sequence of moves to make.
        
        Returns:
            List of moves like ["left", "left", "cw", "drop"]
        """
        pass
    
    def update_state(self, state: GameState):
        """Optional: Update AI's internal state."""
        pass
```

### Step 5: Refactor Existing AI
**File**: `tetris/ai/greedy.py`

```python
from tetris.ai.base import AIPlayer
from tetris.core.game_state import GameState

class GreedyAIPlayer(AIPlayer):
    """Current recursive search AI."""
    
    def __init__(self, params=None):
        self.params = params or self._default_params()
        self._eval_cache = {}
    
    def get_move_sequence(self, state: GameState) -> List[str]:
        """Use existing get_best_moves() logic."""
        # Adapt existing code to work with GameState
        pass
```

### Step 6: Create Simulator
**File**: `tetris/simulation/simulator.py`

```python
from tetris.core.game_engine import GameEngine
from tetris.ai.base import AIPlayer
from tetris.render.base import Renderer
from tetris.simulation.config import SimulationConfig

class Simulator:
    """Orchestrates game engine, AI, and renderer."""
    
    def __init__(
        self,
        engine: GameEngine,
        ai: AIPlayer,
        renderer: Renderer,
        config: SimulationConfig
    ):
        self.engine = engine
        self.ai = ai
        self.renderer = renderer
        self.config = config
    
    def run(self) -> GameResult:
        """Run a single game simulation."""
        state = self.engine.state
        
        while not state.game_over:
            # Get moves from AI
            moves = self.ai.get_move_sequence(state)
            
            # Apply moves
            for move in moves:
                state = self.engine.apply_move(move)
                self.renderer.render(state)
            
            # Tick game forward
            state = self.engine.tick(self.config.delta_time)
            self.renderer.render(state)
        
        return GameResult.from_state(state)
```

### Step 7: Create CLI
**File**: `main.py`

```python
#!/usr/bin/env python3
"""Main CLI entry point for Tetris AI Simulator."""

import argparse
from tetris.simulation import Simulator, SimulationConfig
from tetris.core import GameEngine
from tetris.ai import create_ai
from tetris.render import create_renderer

def run_game(args):
    """Run a single game."""
    config = SimulationConfig(
        ai_type=args.ai,
        render=not args.headless,
        max_moves=args.max_moves
    )
    
    engine = GameEngine()
    ai = create_ai(args.ai)
    renderer = create_renderer(not args.headless)
    
    sim = Simulator(engine, ai, renderer, config)
    result = sim.run()
    
    print(f"Score: {result.score}, Level: {result.level}")

def train(args):
    """Train AI."""
    # Training logic
    pass

def main():
    parser = argparse.ArgumentParser(description="Tetris AI Simulator")
    subparsers = parser.add_subparsers(dest='command')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single game')
    run_parser.add_argument('--ai', choices=['greedy', 'fast', 'random'], default='fast')
    run_parser.add_argument('--headless', action='store_true')
    run_parser.add_argument('--max-moves', type=int, default=10000)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train AI')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_game(args)
    elif args.command == 'train':
        train(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
```

---

## Migration Checklist

### Phase 0: Project Setup & Tooling
- [ ] Initialize `pyproject.toml` with uv
- [ ] Configure ruff for linting/formatting
- [ ] Set up pytest configuration
- [ ] Add mypy configuration (optional)
- [ ] Update `.gitignore`
- [ ] Create `.python-version` file
- [ ] Verify tooling works: `uv run ruff check .`, `uv run pytest`

### Phase 1: Foundation (Week 1)
- [ ] Create package structure
- [ ] Extract `GameState` dataclass
- [ ] Create `GameEngine` class
- [ ] Move game mechanics from `TetrisAI` to `GameEngine`
- [ ] Write unit tests for `GameEngine`

### Phase 2: AI Refactoring (Week 1-2)
- [ ] Create `AIPlayer` ABC
- [ ] Refactor `TetrisAI` → `GreedyAIPlayer`
- [ ] Refactor `FastTetrisAI` → `FastAIPlayer`
- [ ] Create `RandomAIPlayer`
- [ ] Write unit tests for AIs

### Phase 3: Simulation Layer (Week 2)
- [ ] Create `Simulator` class
- [ ] Create `Renderer` ABC and implementations
- [ ] Create configuration classes
- [ ] Write integration tests

### Phase 4: CLI & Cleanup (Week 2-3)
- [ ] Create `main.py` CLI
- [ ] Migrate training logic
- [ ] Update documentation
- [ ] Remove old scripts
- [ ] Update README

---

## Benefits After Refactoring

1. **Single Source of Truth**: One game loop implementation
2. **Testable**: Each component can be unit tested
3. **Extensible**: Easy to add new AI types
4. **Parallelizable**: Can run multiple games (no globals)
5. **Maintainable**: Clear separation of concerns
6. **User-Friendly**: Single CLI interface

---

## Example: Before vs After

### Before (Current)
```python
# run_game.py - 190 lines of duplicated logic
def main():
    playfield = generate_pf()  # Global state
    active_tet = next_tet       # Global state
    ai = TetrisAI(...)          # Tightly coupled
    
    while not scoreboard["game_over"]:  # Global state
        # 100+ lines of game loop logic
        pass
```

### After (Proposed)
```python
# main.py - Clean CLI
def run_game(args):
    engine = GameEngine()           # Encapsulated state
    ai = create_ai(args.ai)        # Dependency injection
    renderer = create_renderer()    # Swappable
    
    sim = Simulator(engine, ai, renderer, config)
    result = sim.run()              # Single method call
    print_result(result)
```

---

## Next Steps

1. **Review this plan** - Does this structure make sense?
2. **Start with Phase 1** - Extract game state and engine
3. **Keep old code working** - Migrate gradually
4. **Add tests as we go** - Don't break existing functionality

Would you like me to start implementing Phase 1?
