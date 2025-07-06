import torch
import time
from typing import List, Tuple
import platform

def get_available_devices() -> List[str]:
    """Return list of available devices for PyTorch computation.
    
    Returns:
        List[str]: List containing available devices ('cpu', 'cuda', 'mps').
        CUDA is available on NVIDIA GPUs, MPS on Apple Silicon.
    """
    devices = ['cpu']  # CPU is always available
    
    # Check for NVIDIA GPU support
    if torch.cuda.is_available():
        devices.append('cuda')
    
    # Check for Apple Silicon (M1/M2/M3) GPU support
    if torch.backends.mps.is_available():
        devices.append('mps')
        
    return devices

def create_tensors(size: Tuple[int, int], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create random tensors of specified size on given device.
    
    Args:
        size (Tuple[int, int]): Dimensions of the tensors to create (rows, cols)
        device (str): Device to place tensors on ('cpu', 'cuda', or 'mps')
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two random tensors of the specified size
    """
    return (
        torch.randn(size, device=device),
        torch.randn(size, device=device)
    )

def benchmark_matmul(size: Tuple[int, int], device: str, num_iterations: int = 10) -> float:
    """Benchmark matrix multiplication for given size and device.
    
    Args:
        size (Tuple[int, int]): Size of matrices to multiply
        device (str): Device to run benchmark on
        num_iterations (int): Number of iterations to average over
    
    Returns:
        float: Average time per operation in milliseconds
    """
    # Create input tensors on the specified device
    a, b = create_tensors(size, device)
    
    # Perform warmup iterations to ensure GPU is at full speed
    # and avoid including compilation/optimization time
    for _ in range(10):
        _ = torch.matmul(a, b)
    
    # Ensure all operations are completed before timing
    # Different devices require different synchronization methods
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    # Time the actual benchmark iterations
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        _ = torch.matmul(a, b)
    
    # Ensure all operations are completed before stopping timer
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate average time per operation
    return (end_time - start_time) / num_iterations * 1000

def run_benchmarks():
    """Run benchmarks across all available devices.
    
    This function:
    1. Detects available devices
    2. Tests multiple matrix sizes
    3. Reports timing results in a formatted table
    4. Handles errors gracefully
    """
    # Get list of available compute devices
    devices = get_available_devices()
    
    # Define matrix sizes to test
    # Larger sizes will show more pronounced differences between devices
    sizes = [
        (1000, 1000),  # 1K x 1K elements
        (2000, 2000),  # 2K x 2K elements
        (4000, 4000),   # 4K x 4K elements
        (5000, 5000)   # 5K x 5K elements
    ]
    
    # Print system information and available devices
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.machine()}\n")
    print("Available devices:", devices)
    print("\nMatrix Multiplication Benchmark Results (milliseconds per operation):\n")
    
    # Create and print table header
    header = "Size".rjust(15)
    for device in devices:
        header += f"{device.upper().rjust(15)}"
    header += f"Speed up".rjust(15)
    print(header)
    print("-" * (15 + 15 * (len(devices) +1)))
    
    # Run benchmarks for each matrix size
    for size in sizes:
        size_str = f"{size[0]}x{size[1]}".rjust(15)
        last_time_taken = 0
        time_taken = 0
        # Test each available device
        for device in devices:
            try:
                last_time_taken = time_taken
                time_taken = benchmark_matmul(size, device)
                size_str += f"{time_taken:.2f}".rjust(15)
            except Exception as e:
                # Handle any errors (out of memory, device not supported, etc.)
                size_str += f"Error: {str(e)[:10]}".rjust(15)
        # Check the speed-up ratio
        ratio = last_time_taken / time_taken
        size_str += f"{ratio:.2f}x".rjust(15)


        print(size_str)

if __name__ == "__main__":
    run_benchmarks()
