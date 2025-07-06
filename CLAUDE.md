# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Dependencies

Install required dependencies:
```bash
sudo apt install ffmpeg
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu transformers accelerate safetensors ffmpeg-python 'urllib3<2.0'
```

## Running Benchmarks and Scripts

Run PyTorch matrix multiplication benchmarks:
```bash
python pytorch_benchmark.py
```

Run Whisper audio transcription:
```bash
python pytorch_whisper.py
```

Run Whisper tokens per second benchmark:
```bash
python whisper_benchmark.py
```

## Code Architecture

This is a PyTorch benchmarking and ML inference repository with three main components:

### pytorch_benchmark.py
- Benchmarks matrix multiplication performance across available devices (CPU, CUDA, MPS)
- Tests multiple matrix sizes (1K to 5K elements)
- Includes proper device synchronization for accurate timing
- Reports results in milliseconds with speedup ratios

### pytorch_whisper.py  
- Audio transcription using OpenAI's Whisper models
- Handles audio loading via ffmpeg with proper error handling
- Implements chunked processing for long audio files
- Supports multiple devices (CPU/CUDA/MPS) with automatic detection
- Uses transformers library with attention masks for proper inference

### whisper_benchmark.py
- Benchmarks Whisper model inference speed in tokens per second
- Tests multiple model sizes (tiny, base, small)
- Includes warmup iterations and proper device synchronization
- Reports comprehensive metrics: tokens/sec, processing time, token counts
- Uses deterministic generation for consistent benchmarking

Key architectural patterns:
- Device detection and selection logic is used across all scripts
- Proper synchronization patterns for different accelerators (cuda.synchronize(), mps.synchronize())
- Error handling with informative messages for missing dependencies
- Chunked processing pattern for handling large inputs
- Warmup iterations before benchmarking for accurate measurements