import torch
import time
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark_tokens_per_second_device(model_name: str, prompt: str, max_tokens: int, device: str) -> float:
    """Benchmark token generation speed on a specific device.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input text prompt
        max_tokens: Number of tokens to generate
        device: Device to run on ("cpu" or "cuda")
        
    Returns:
        float: Tokens per second
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    
    end_time = time.perf_counter()
    
    tokens_per_second = max_tokens / (end_time - start_time)
    
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Generated {max_tokens} tokens in {end_time - start_time:.2f}s")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print("-" * 50)
    
    return tokens_per_second

def benchmark_tokens_per_second(model_name: str = "gpt2-medium", prompt: str = "The future of AI is", max_tokens: int = 300) -> Dict[str, float]:
    """Benchmark token generation speed on available devices.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input text prompt
        max_tokens: Number of tokens to generate
        
    Returns:
        Dict[str, float]: Device name to tokens per second mapping
    """
    results = {}
    devices = ["cpu"]
    
    if torch.cuda.is_available():
        devices.append("cuda")
    
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    print(f"Running benchmark on devices: {devices}")
    print("=" * 50)
    
    for device in devices:
        print(f"\nBenchmarking on {device.upper()}...")
        try:
            tps = benchmark_tokens_per_second_device(model_name, prompt, max_tokens, device)
            results[device] = tps
        except Exception as e:
            print(f"Error benchmarking on {device}: {e}")
            results[device] = 0.0
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    for device, tps in results.items():
        if tps > 0:
            print(f"{device.upper()}: {tps:.2f} tokens/sec")
        else:
            print(f"{device.upper()}: Failed")
    
    if len(results) > 1 and all(tps > 0 for tps in results.values()):
        speedup = max(results.values()) / min(results.values())
        faster_device = max(results, key=results.get)
        print(f"\nSpeedup: {speedup:.2f}x ({faster_device.upper()} vs others)")
    
    return results

if __name__ == "__main__":
    benchmark_tokens_per_second()
