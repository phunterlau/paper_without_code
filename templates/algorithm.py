"""
Template for algorithm-related papers.
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Callable
import random

def measure_execution_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure the execution time of a function.
    
    Args:
        func (Callable): The function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        tuple: The function result and the execution time in seconds
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def generate_random_array(size: int, min_val: int = 0, max_val: int = 1000) -> List[int]:
    """
    Generate a random array of integers.
    
    Args:
        size (int): The size of the array
        min_val (int): The minimum value
        max_val (int): The maximum value
        
    Returns:
        list: The random array
    """
    return [random.randint(min_val, max_val) for _ in range(size)]

def benchmark_algorithm(algorithm: Callable, input_sizes: List[int], 
                        generate_input: Callable[[int], Any], 
                        num_runs: int = 5) -> Dict[int, List[float]]:
    """
    Benchmark an algorithm with different input sizes.
    
    Args:
        algorithm (Callable): The algorithm to benchmark
        input_sizes (list): List of input sizes to test
        generate_input (Callable): Function to generate input of a given size
        num_runs (int): Number of runs for each input size
        
    Returns:
        dict: Dictionary mapping input sizes to lists of execution times
    """
    results = {}
    
    for size in input_sizes:
        print(f"Benchmarking input size: {size}")
        times = []
        
        for _ in range(num_runs):
            input_data = generate_input(size)
            _, execution_time = measure_execution_time(algorithm, input_data)
            times.append(execution_time)
        
        results[size] = times
    
    return results

def plot_benchmark_results(results: Dict[int, List[float]], algorithm_name: str):
    """
    Plot benchmark results.
    
    Args:
        results (dict): Dictionary mapping input sizes to lists of execution times
        algorithm_name (str): The name of the algorithm
    """
    input_sizes = list(results.keys())
    avg_times = [sum(times) / len(times) for times in results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, avg_times, 'o-', label=f"{algorithm_name} (Average)")
    
    # Plot min and max times
    min_times = [min(times) for times in results.values()]
    max_times = [max(times) for times in results.values()]
    plt.fill_between(input_sizes, min_times, max_times, alpha=0.2, label="Min-Max Range")
    
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"Performance of {algorithm_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_algorithms(algorithms: Dict[str, Callable], input_sizes: List[int], 
                       generate_input: Callable[[int], Any], 
                       num_runs: int = 3) -> Dict[str, Dict[int, List[float]]]:
    """
    Compare multiple algorithms with different input sizes.
    
    Args:
        algorithms (dict): Dictionary mapping algorithm names to functions
        input_sizes (list): List of input sizes to test
        generate_input (Callable): Function to generate input of a given size
        num_runs (int): Number of runs for each input size
        
    Returns:
        dict: Dictionary mapping algorithm names to benchmark results
    """
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\nBenchmarking algorithm: {name}")
        results[name] = benchmark_algorithm(algorithm, input_sizes, generate_input, num_runs)
    
    return results

def plot_comparison_results(results: Dict[str, Dict[int, List[float]]]):
    """
    Plot comparison results for multiple algorithms.
    
    Args:
        results (dict): Dictionary mapping algorithm names to benchmark results
    """
    plt.figure(figsize=(12, 8))
    
    for name, algorithm_results in results.items():
        input_sizes = list(algorithm_results.keys())
        avg_times = [sum(times) / len(times) for times in algorithm_results.values()]
        plt.plot(input_sizes, avg_times, 'o-', label=name)
    
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Algorithm Performance Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot with logarithmic scale
    plt.figure(figsize=(12, 8))
    
    for name, algorithm_results in results.items():
        input_sizes = list(algorithm_results.keys())
        avg_times = [sum(times) / len(times) for times in algorithm_results.values()]
        plt.loglog(input_sizes, avg_times, 'o-', label=name)
    
    plt.xlabel("Input Size (log scale)")
    plt.ylabel("Execution Time (seconds, log scale)")
    plt.title("Algorithm Performance Comparison (Log-Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example algorithm implementations
def bubble_sort(arr: List[int]) -> List[int]:
    """
    Bubble sort algorithm.
    
    Args:
        arr (list): The array to sort
        
    Returns:
        list: The sorted array
    """
    n = len(arr)
    arr_copy = arr.copy()  # Create a copy to avoid modifying the original
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
    
    return arr_copy

def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick sort algorithm.
    
    Args:
        arr (list): The array to sort
        
    Returns:
        list: The sorted array
    """
    if len(arr) <= 1:
        return arr
    
    arr_copy = arr.copy()  # Create a copy to avoid modifying the original
    
    def _quick_sort(arr, low, high):
        if low < high:
            pivot_idx = partition(arr, low, high)
            _quick_sort(arr, low, pivot_idx - 1)
            _quick_sort(arr, pivot_idx + 1, high)
    
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    _quick_sort(arr_copy, 0, len(arr_copy) - 1)
    return arr_copy

def main():
    """
    Main function to demonstrate the implementation.
    """
    parser = argparse.ArgumentParser(description="Algorithm benchmarking template")
    parser.add_argument("--min-size", type=int, default=1000, help="Minimum input size")
    parser.add_argument("--max-size", type=int, default=10000, help="Maximum input size")
    parser.add_argument("--step-size", type=int, default=1000, help="Step size for input sizes")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for each input size")
    args = parser.parse_args()
    
    # Define input sizes
    input_sizes = list(range(args.min_size, args.max_size + 1, args.step_size))
    
    # Define algorithms to compare
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Quick Sort": quick_sort,
        "Python Sort": sorted
    }
    
    # Compare algorithms
    results = compare_algorithms(
        algorithms, 
        input_sizes, 
        lambda size: generate_random_array(size),
        args.num_runs
    )
    
    # Plot results
    plot_comparison_results(results)
    
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()
