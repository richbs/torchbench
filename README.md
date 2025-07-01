# TorchBench

PyTorch benchmarking suite for testing performance across different devices and models.

## Installation

```bash
# Install system dependencies
sudo apt install ffmpeg  # Linux
# brew install ffmpeg    # macOS

# Install Python dependencies
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu transformers accelerate safetensors ffmpeg-python 'urllib3<2.0'
```

## Benchmarks

### Token Generation Benchmark (`tokens_benchmark.py`)

Benchmarks text generation speed across available devices (CPU, CUDA, MPS).

```bash
python3 tokens_benchmark.py
```

**Features:**
- Multi-device testing in single run
- Automatic device detection (CPU, CUDA for NVIDIA GPUs, MPS for Apple Silicon)
- Performance comparison with speedup calculations
- Proper GPU synchronization for accurate timing

**Example Output:**
```
Running benchmark on devices: ['cpu', 'cuda']
==================================================

Benchmarking on CPU...
Model: gpt2
Device: cpu
Generated 100 tokens in 4.08s
Tokens per second: 24.50

Benchmarking on CUDA...
Model: gpt2
Device: cuda  
Generated 100 tokens in 1.51s
Tokens per second: 66.10

==================================================
SUMMARY:
CPU: 24.50 tokens/sec
CUDA: 66.10 tokens/sec

Speedup: 2.70x (CUDA vs others)
```

### Other Benchmarks

- `pytorch_benchmark.py` - General PyTorch operations
- `pytorch_whisper.py` - Whisper model benchmarking

## Performance Expectations

**Token Generation Speedups (GPU vs CPU):**
- Small models (GPT-2, 117M params): 2-5x speedup
- Medium models (GPT-2 XL, 1.5B params): 5-15x speedup  
- Large models (7B+ params): 10-50x+ speedup
- With batching: Additional 2-10x improvement

**Factors affecting speedup:**
- **Model size**: Larger models show better GPU utilization
- **Batch size**: GPUs excel with multiple sequences (8, 16, 32+)
- **Sequence length**: Longer sequences (512, 1024+ tokens) improve GPU efficiency
- **Memory overhead**: Small models spend more time on CPU-GPU transfers

**Device Support:**
- **CPU**: Always available baseline
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon (M1/M2/M3) Metal Performance Shaders
