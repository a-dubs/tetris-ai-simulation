"""
Performance benchmarks for Tetris AI optimizations.
Tests individual functions and provides before/after comparisons.
"""
import time
import sys

# Try to import dependencies, handle gracefully if tkinter is missing
try:
    from tetrimino import Tetrimino
    from tetris_ai import TetrisAI
    # Import playfield functions without tkinter
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    # Define these locally to avoid tkinter import
    playfield_width = 10
    playfield_height = 20
    skybox_height = 4
    playfield_height += skybox_height
    
    def generate_pf():
        return [[' ' for x in range(playfield_width)] for y in range(playfield_height)]
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Some benchmarks may not run.")
    sys.exit(1)


def benchmark_heuristic_evaluation(iterations=100):
    """Benchmark greedy_heuristic_evaluation function."""
    pf = generate_pf()
    # Add some blocks to make it realistic
    for row in range(15, 20):
        for col in range(playfield_width):
            if col != 5:  # Leave a gap
                pf[row][col] = 'r'
    
    tet = Tetrimino('I', x=5, y=18)
    # Initialize AI with default params
    ai = TetrisAI(pf, tet, Tetrimino('T', x=4, y=21), 3, 1, method="greedy")
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = ai.greedy_heuristic_evaluation(tet, pf)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'avg': avg_time * 1000,  # Convert to ms
        'min': min_time * 1000,
        'max': max_time * 1000,
        'total': sum(times) * 1000,
        'result': result
    }


def benchmark_get_best_moves(iterations=10):
    """Benchmark get_best_moves function (full search)."""
    pf = generate_pf()
    # Add some blocks
    for row in range(15, 20):
        for col in range(playfield_width):
            if col not in [4, 5, 6]:
                pf[row][col] = 'r'
    
    tet = Tetrimino('I', x=5, y=18)
    next_tet = Tetrimino('T', x=4, y=21)
    ai = TetrisAI(pf, tet, next_tet, 3, 1, method="greedy")
    
    times = []
    results = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = ai.get_best_moves()
        times.append(time.perf_counter() - start)
        results.append(result[0])  # Store evaluation score
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'avg': avg_time * 1000,
        'min': min_time * 1000,
        'max': max_time * 1000,
        'total': sum(times) * 1000,
        'results': results
    }


def benchmark_deepcopy_vs_shallowcopy(iterations=1000):
    """Compare deepcopy vs shallow copy performance."""
    from copy import deepcopy, copy
    
    pf = generate_pf()
    for row in range(10, 20):
        for col in range(playfield_width):
            pf[row][col] = 'r'
    
    # Test deepcopy
    deepcopy_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pf_copy = deepcopy(pf)
        deepcopy_times.append(time.perf_counter() - start)
    
    # Test shallow copy (list comprehension)
    shallowcopy_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pf_copy = [row[:] for row in pf]
        shallowcopy_times.append(time.perf_counter() - start)
    
    return {
        'deepcopy_avg': sum(deepcopy_times) / len(deepcopy_times) * 1000,
        'shallowcopy_avg': sum(shallowcopy_times) / len(shallowcopy_times) * 1000,
        'speedup': sum(deepcopy_times) / sum(shallowcopy_times)
    }


def run_all_benchmarks(baseline_avg=None, baseline_total=None):
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    print("\n1. Heuristic Evaluation Benchmark (100 iterations)")
    print("-" * 70)
    result = benchmark_heuristic_evaluation(100)
    print(f"  Average: {result['avg']:.4f} ms", end="")
    if baseline_avg:
        speedup = baseline_avg / result['avg']
        improvement = (1 - result['avg'] / baseline_avg) * 100
        print(f" (was {baseline_avg:.4f} ms, {improvement:+.1f}% faster, {speedup:.2f}x speedup)")
    else:
        print()
    print(f"  Min:     {result['min']:.4f} ms")
    print(f"  Max:     {result['max']:.4f} ms")
    print(f"  Total:   {result['total']:.4f} ms", end="")
    if baseline_total:
        speedup = baseline_total / result['total']
        print(f" (was {baseline_total:.4f} ms, {speedup:.2f}x speedup)")
    else:
        print()
    print(f"  Result:  {result['result']:.2f}")
    
    print("\n2. Get Best Moves Benchmark (10 iterations)")
    print("-" * 70)
    result = benchmark_get_best_moves(10)
    print(f"  Average: {result['avg']:.4f} ms")
    print(f"  Min:     {result['min']:.4f} ms")
    print(f"  Max:     {result['max']:.4f} ms")
    print(f"  Total:   {result['total']:.4f} ms")
    print(f"  Results: {result['results']}")
    
    print("\n3. Deepcopy vs Shallow Copy Comparison (1000 iterations)")
    print("-" * 70)
    result = benchmark_deepcopy_vs_shallowcopy(1000)
    print(f"  Deepcopy avg:    {result['deepcopy_avg']:.4f} ms")
    print(f"  Shallow copy avg: {result['shallowcopy_avg']:.4f} ms")
    print(f"  Speedup:          {result['speedup']:.2f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
