import os
import warnings
import torch
import numpy as np
import ffmpeg
import time
import platform
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Filter out specific warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*set_audio_backend.*")

def load_audio(file_path, sample_rate=16000):
    """
    Load an audio file using ffmpeg-python.
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Get file info
        probe = ffmpeg.probe(file_path)
        audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        
        # Load audio data
        out, _ = (
            ffmpeg
            .input(file_path)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Convert to tensor
        audio = np.frombuffer(out, np.float32).copy()
        audio = torch.from_numpy(audio)
        
        # Reshape to match expected format [1, samples]
        audio = audio.unsqueeze(0)
        
        return audio
        
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {str(e.stderr.decode())}")
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {str(e)}")

def chunk_audio(waveform, chunk_length_s=30, sample_rate=16000):
    """
    Split audio into chunks of chunk_length_s seconds.
    """
    samples_per_chunk = int(chunk_length_s * sample_rate)
    waveform = waveform.squeeze(0)  # Remove channel dimension
    
    # Calculate number of chunks needed
    num_chunks = int(np.ceil(len(waveform) / samples_per_chunk))
    
    chunks = []
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = min(start + samples_per_chunk, len(waveform))
        chunk = waveform[start:end]
        chunks.append(chunk)
    
    return chunks

def benchmark_tokens_per_second(model_size="base", chunk_length_s=30, num_iterations=3):
    """
    Benchmark Whisper model inference speed in tokens per second.
    
    Args:
        model_size (str): Whisper model size to benchmark
        chunk_length_s (int): Length of audio chunks in seconds
        num_iterations (int): Number of iterations to average over
        
    Returns:
        dict: Benchmark results containing tokens/sec, processing time, etc.
    """
    try:
        print(f"Benchmarking Whisper {model_size} model...")
        
        # Use the existing audio file for benchmarking
        audio_path = "./holysonnet_01_donne.mp3"
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Benchmark audio file not found: {audio_path}")
            
        print("Loading audio file...")
        waveform = load_audio(audio_path)
        
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and platform.system() == "Darwin" and platform.processor() == "arm":
            device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
            
        model_name = f"openai/whisper-{model_size}"
        print(f"Using device: {device}")
        
        processor = WhisperProcessor.from_pretrained(model_name)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        
        # Prepare test chunk (use first chunk for consistent benchmarking)
        chunks = chunk_audio(waveform, chunk_length_s=chunk_length_s)
        test_chunk = chunks[0]  # Use first chunk for benchmarking
        
        # Prepare input features
        features = processor(
            test_chunk.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        features = {
            k: v.to(device).to(torch.float16) if torch.is_tensor(v) else v
            for k, v in features.items()
        }
        
        # Warmup runs
        print("Running warmup iterations...")
        for _ in range(2):
            _ = model.generate(
                features['input_features'],
                attention_mask=features['attention_mask'],
                max_new_tokens=200,
                forced_decoder_ids=forced_decoder_ids
            )
        
        # Synchronize before benchmark
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
        
        # Benchmark iterations
        print(f"Running {num_iterations} benchmark iterations...")
        total_tokens = 0
        total_time = 0
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            predicted_ids = model.generate(
                features['input_features'],
                attention_mask=features['attention_mask'],
                max_new_tokens=200,
                forced_decoder_ids=forced_decoder_ids,
                do_sample=False,  # Deterministic for benchmarking
                return_timestamps=False
            )
            
            # Synchronize after generation
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
                
            end_time = time.perf_counter()
            
            iteration_time = end_time - start_time
            num_tokens = predicted_ids.shape[1] - features['input_features'].shape[2] // 320  # Subtract input length
            
            total_tokens += num_tokens
            total_time += iteration_time
            
            print(f"  Iteration {i+1}: {num_tokens} tokens in {iteration_time:.3f}s ({num_tokens/iteration_time:.1f} tokens/sec)")
        
        # Calculate averages
        avg_tokens_per_sec = total_tokens / total_time
        avg_time_per_iteration = total_time / num_iterations
        avg_tokens_per_iteration = total_tokens / num_iterations
        
        results = {
            "model_size": model_size,
            "device": str(device),
            "tokens_per_second": avg_tokens_per_sec,
            "avg_processing_time": avg_time_per_iteration,
            "avg_tokens_generated": avg_tokens_per_iteration,
            "total_iterations": num_iterations,
            "chunk_length_seconds": chunk_length_s
        }
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Benchmark failed: {str(e)}")

def run_benchmark_suite():
    """
    Run comprehensive tokens per second benchmarks across different model sizes.
    """
    model_sizes = ["tiny", "base", "small"]
    
    print("Whisper Model Benchmark Suite")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.machine()}")
    
    # Check available devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and platform.system() == "Darwin" and platform.processor() == "arm":
        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    results = []
    
    for model_size in model_sizes:
        try:
            print(f"\n{'='*20} {model_size.upper()} MODEL {'='*20}")
            result = benchmark_tokens_per_second(model_size=model_size, num_iterations=3)
            results.append(result)
            
            print(f"\nResults for {model_size} model:")
            print(f"  Tokens per second: {result['tokens_per_second']:.1f}")
            print(f"  Avg processing time: {result['avg_processing_time']:.3f}s")
            print(f"  Avg tokens generated: {result['avg_tokens_generated']:.0f}")
            
        except Exception as e:
            print(f"Benchmark failed for {model_size} model: {str(e)}")
            continue
    
    # Print summary table
    if results:
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<10} {'Tokens/sec':<12} {'Proc Time (s)':<14} {'Avg Tokens':<12}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['model_size']:<10} "
                  f"{result['tokens_per_second']:<12.1f} "
                  f"{result['avg_processing_time']:<14.3f} "
                  f"{result['avg_tokens_generated']:<12.0f}")

def main():
    """
    Run the benchmark suite.
    """
    try:
        run_benchmark_suite()
    except Exception as e:
        print(f"Error during benchmarking: {str(e)}")
        if "FFmpeg" in str(e):
            print("\nPlease ensure FFmpeg is installed:")
            if platform.system() == "Darwin":  # macOS
                print("brew install ffmpeg")
            elif platform.system() == "Linux":
                print("sudo apt-get install ffmpeg")
            else:  # Windows
                print("Download FFmpeg from https://www.ffmpeg.org/download.html")

if __name__ == "__main__":
    main()