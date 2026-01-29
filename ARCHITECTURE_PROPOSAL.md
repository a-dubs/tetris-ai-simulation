# Tetris AI Simulator Architecture Proposal

## Current Problems

### 1. **Global State Everywhere**
- `playfield`, `active_tet`, `scoreboard`, `bag`, `ai` are all global variables
- Makes testing impossible
- Can't run multiple games in parallel
- Hard to reason about state changes

### 2. **No Separation of Concerns**
- `tetris_sim.py` mixes: game logic, GUI rendering, training, CSV I/O
- `TetrisAI` contains both game mechanics (gravity, collision) AND AI logic
- Hard to swap components or test in isolation

### 3. **Multiple Ad-Hoc Scripts**
- `run_game.py`, `run_game_random.py`, `run_game_fast_ai.py` all duplicate game loop logic
- Each script reimplements initialization, game loop, scoring
- No single source of truth

### 4. **Tight Coupling**
- AI directly mutates game state (`ai.pf = playfield`)
- Game logic scattered across multiple files
- Hard to add new AI types or game modes

### 5. **No Clear Entry Point**
- Multiple ways to run: `run_trial()`, `run_trials()`, `train()`, scripts
- No CLI interface
- Hard to know what to call

---

## Proposed Architecture

### Core Principles
1. **Single Responsibility**: Each class has one job
2. **Dependency Injection**: Pass dependencies, don't use globals
3. **Interfaces/ABCs**: Define contracts, allow swapping implementations
4. **Immutable Where Possible**: Reduce state mutation bugs
5. **Testable**: Easy to unit test each component

---

## Proposed Structure

```
tetris_ai_simulation/
├── tetris/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── game_engine.py      # Core game logic (no AI, no rendering)
│   │   ├── game_state.py       # Immutable game state
│   │   ├── tetrimino.py        # Tetrimino class (keep as-is, but clean up)
│   │   └── constants.py        # Game constants (width, height, etc.)
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class for AIs
│   │   ├── greedy.py           # Current recursive AI
│   │   ├── fast.py             # Fast direct evaluation AI
│   │   ├── random.py           # Random AI (for testing)
│   │   └── heuristics.py       # Heuristic evaluation functions
│   │
│   ├── render/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract renderer interface
│   │   ├── headless.py         # No-op renderer (for training)
│   │   └── tkinter_gui.py      # GUI renderer (optional)
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── simulator.py        # Orchestrates game + AI + renderer
│   │   └── config.py           # Configuration dataclasses
│   │
│   └── training/
│       ├── __init__.py
│       ├── trainer.py          # Training logic
│       └── metrics.py          # Metrics collection
│
├── scripts/
│   ├── run_game.py             # Single CLI entry point
│   └── benchmark.py            # Benchmarking script
│
├── tests/
│   ├── test_game_engine.py
│   ├── test_ai.py
│   └── test_simulation.py
│
└── main.py                      # CLI entry point
```

---

## Core Classes

### 1. `GameState` (Immutable)
```python
@dataclass(frozen=True)
class GameState:
    """Immutable game state."""
    playfield: List[List[str]]
    active_tetrimino: Tetrimino
    next_tetrimino: Tetrimino
    bag: List[str]
    score: int
    level: int
    lines_cleared: int
    game_over: bool
    time_elapsed: float
```

### 2. `GameEngine` (Pure Game Logic)
```python
class GameEngine:
    """Core game logic - no AI, no rendering."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.state = self._initial_state()
    
    def apply_move(self, move: str) -> GameState:
        """Apply a move, return new state."""
        # Pure function - returns new state
        
    def apply_gravity(self, delta_time: float) -> GameState:
        """Apply gravity, return new state."""
        
    def spawn_next_piece(self) -> GameState:
        """Spawn next piece, return new state."""
        
    def check_game_over(self) -> bool:
        """Check if game is over."""
```

### 3. `AIPlayer` (Abstract Base Class)
```python
class AIPlayer(ABC):
    """Abstract base class for AI players."""
    
    @abstractmethod
    def get_move(self, state: GameState) -> List[str]:
        """Return sequence of moves to make."""
        pass
    
    @abstractmethod
    def update_state(self, state: GameState):
        """Update AI's view of game state."""
        pass
```

### 4. `Simulator` (Orchestrator)
```python
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
            moves = self.ai.get_move(state)
            
            # Apply moves
            for move in moves:
                state = self.engine.apply_move(move)
                self.renderer.render(state)
            
            # Apply gravity, spawn, etc.
            state = self.engine.tick(delta_time)
            
        return GameResult.from_state(state)
```

### 5. `Renderer` (Abstract Base Class)
```python
class Renderer(ABC):
    """Abstract renderer interface."""
    
    @abstractmethod
    def render(self, state: GameState):
        """Render current game state."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass
```

---

## Implementation Plan

### Phase 1: Extract Core Game Logic
1. Create `GameState` dataclass
2. Extract game logic from `tetris_sim.py` into `GameEngine`
3. Remove all global state
4. Make game logic pure functions where possible

### Phase 2: Refactor AI
1. Create `AIPlayer` ABC
2. Refactor `TetrisAI` to implement `AIPlayer`
3. Move game mechanics (gravity, collision) to `GameEngine`
4. Keep only AI logic in AI classes

### Phase 3: Create Simulator
1. Create `Simulator` class
2. Create `Renderer` ABC and implementations
3. Create configuration classes

### Phase 4: Create CLI
1. Single `main.py` entry point
2. Use `argparse` or `click` for CLI
3. Support: `run`, `train`, `benchmark`, `play` modes

### Phase 5: Testing & Cleanup
1. Write unit tests for each component
2. Remove old scripts
3. Update documentation

---

## Example Usage

### Running a Single Game
```python
from tetris.simulation import Simulator, SimulationConfig
from tetris.core import GameEngine, GameConfig
from tetris.ai import FastAIPlayer
from tetris.render import HeadlessRenderer

config = SimulationConfig(
    ai_type="fast",
    render=False,
    max_moves=10000
)

engine = GameEngine(GameConfig())
ai = FastAIPlayer()
renderer = HeadlessRenderer()

sim = Simulator(engine, ai, renderer, config)
result = sim.run()

print(f"Score: {result.score}, Level: {result.level}")
```

### CLI Usage
```bash
# Run single game with fast AI
python main.py run --ai fast --headless

# Run with GUI
python main.py run --ai greedy --gui

# Train AI
python main.py train --epochs 100 --batch-size 10

# Benchmark
python main.py benchmark --ai fast --trials 100
```

---

## Benefits

1. **Testable**: Each component can be tested in isolation
2. **Extensible**: Easy to add new AI types, renderers, game modes
3. **Maintainable**: Clear separation of concerns
4. **Reusable**: Components can be used independently
5. **No Globals**: Thread-safe, can run multiple games in parallel
6. **Single Entry Point**: Clear CLI interface

---

## Migration Strategy

1. **Keep old code working** during refactor
2. **Create new structure alongside** old code
3. **Gradually migrate** functionality
4. **Add tests** as we go
5. **Remove old code** once new structure is complete

---

## Questions to Consider

1. **Should GameState be truly immutable?** (using `@dataclass(frozen=True)`)
   - Pro: Prevents bugs, easier to reason about
   - Con: More allocations (but we're already doing this)

2. **How to handle performance-critical paths?**
   - Keep fast paths in Cython/Numba?
   - Or optimize Python code first?

3. **Configuration format?**
   - Python dataclasses? (simple, type-safe)
   - YAML/JSON? (user-friendly)
   - Both? (YAML -> dataclass)

4. **AI interface design?**
   - Return moves? (current)
   - Return action + state? (more flexible)
   - Callback-based? (for async)
