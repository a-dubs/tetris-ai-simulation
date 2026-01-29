"""
Run a single Tetris game simulation and measure execution time.
No GUI required - runs headlessly.
"""
import time
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import before tetris_sim to avoid tkinter issues
from tetrimino import Tetrimino
from tetris_ai import TetrisAI

# Now import tetris_sim functions we need
# We'll need to manually set up what run_trial needs without GUI
def generate_pf():
    playfield_width = 10
    playfield_height = 20
    skybox_height = 4
    playfield_height += skybox_height
    return [[' ' for x in range(playfield_width)] for y in range(playfield_height)]

def generate_scoreboard():
    return {
        "time_elapsed": 0,
        "score": 0, 
        "level": 1,
        "clears": 0,
        "game_over": False
    }

def clears_needed(level):
    return level * ((5 + level * 5) / 2)

def main():
    print("=" * 70)
    print("Running Single Tetris Game Simulation")
    print("=" * 70)
    print()
    
    # Set up game state
    playfield = generate_pf()
    playfield_width = 10
    playfield_height = 24  # 20 + 4 skybox
    bag_size = 1
    
    import random
    bag = Tetrimino.make_bag(bag_size)
    next_tet = Tetrimino(bag.pop(), x=4, y=21)
    
    # Spawn first tetrimino
    active_tet = next_tet
    next_tet = Tetrimino(bag.pop(0), x=4, y=playfield_height - 3)
    
    scoreboard = generate_scoreboard()
    prev_time = -1
    
    # Initialize AI
    mps = 4  # Intermediate speed
    ai = TetrisAI(playfield, active_tet, next_tet, mps, 1, method="greedy")
    
    print(f"AI Method: {ai.method}")
    print(f"Moves per second: {mps}")
    print(f"Starting simulation...")
    print()
    
    # Get initial moves
    tet_moves = ai.get_simplified_path(ai.get_best_moves()[1])
    
    start_time = time.perf_counter()
    move_count = 0
    
    # Game loop
    while not scoreboard["game_over"]:
        m = ""
        if tet_moves:
            TetrisAI.execute_move(playfield, active_tet, (m:= tet_moves[0]))
            tet_moves.pop(0)
            move_count += 1
            
            # Apply gravity
            if prev_time != (p:= ai.gravity(playfield, active_tet, prev_time, scoreboard["time_elapsed"], scoreboard["level"])):
                prev_time = p

        # Check if tetrimino has landed
        if m == "drop":
            # Update playfield
            clears = TetrisAI.update_playfield(playfield, active_tet)
            scoreboard["clears"] += clears
            scoreboard["score"] += TetrisAI.score_move(clears, scoreboard["level"])
            
            if scoreboard["clears"] >= clears_needed(scoreboard["level"]):
                bag = Tetrimino.make_bag(bag_size)
                scoreboard["level"] += 1
            
            # Spawn new tetrimino
            active_tet = next_tet
            next_tet = Tetrimino(bag.pop(0), x=4, y=playfield_height - 3)
            
            if not bag:
                bag = Tetrimino.make_bag(bag_size)
            
            # Check game over
            if not TetrisAI.valid_location(playfield, active_tet):
                scoreboard["game_over"] = True
                break
            
            if not TetrisAI.tetrimino_has_landed(playfield, active_tet):
                active_tet.y -= 1
            
            # Update AI state
            ai.pf = playfield
            ai.tet = active_tet
            ai.next_tet = next_tet
            
            if not scoreboard["game_over"]:
                active_tet.ld_timer = -1
                try:
                    ai_results = ai.get_best_moves()
                    tet_moves = ai.get_simplified_path(ai_results[1])
                except Exception as e:
                    print(f"\nError during AI move calculation: {e}")
                    scoreboard["game_over"] = True
                    break
        
        scoreboard["time_elapsed"] += 1.0 / ai.mps
        
        # Progress indicator
        if move_count % 50 == 0:
            print(f"\rMoves: {move_count}, Score: {scoreboard['score']:,}, Level: {scoreboard['level']}, Clears: {scoreboard['clears']}", end="", flush=True)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    print()
    print()
    print("=" * 70)
    print("Simulation Complete")
    print("=" * 70)
    print(f"Execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print()
    print("Final Results:")
    print(f"  Score:     {scoreboard['score']:,}")
    print(f"  Level:     {scoreboard['level']}")
    print(f"  Clears:    {scoreboard['clears']}")
    print(f"  Moves:     {move_count}")
    print(f"  Game Time: {scoreboard['time_elapsed']:.2f} seconds")
    print()
    
    if elapsed > 0:
        moves_per_sec = move_count / elapsed
        print(f"Performance Metrics:")
        print(f"  Moves per second: {moves_per_sec:.2f}")
        if scoreboard['time_elapsed'] > 0:
            print(f"  Real-time factor: {scoreboard['time_elapsed'] / elapsed:.2f}x")
    print("=" * 70)

if __name__ == "__main__":
    main()
