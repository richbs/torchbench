import os
import warnings
import torch
import numpy as np
import ffmpeg
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import platform

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

def transcribe_audio(file_path, model_size="base", chunk_length_s=30):
    """
    Transcribe audio file to text using OpenAI's Whisper model.
    """
    try:
        print("Loading audio file...")
        waveform = load_audio(file_path)
        
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and platform.system() == "Darwin" and platform.processor() == "arm":
            device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        #device = "cpu"
        model_name = f"openai/whisper-{model_size}"
        print(f"Using Whisper model: {model_name} with device {device}")

        processor = WhisperProcessor.from_pretrained(model_name)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(device)
        
        print("Processing audio in chunks...")
        chunks = chunk_audio(waveform, chunk_length_s=chunk_length_s)
        
        transcription = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i} of {len(chunks)}...")
            
            # Create explicit attention mask for the input features
            features = processor(
                chunk.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True  # Explicitly request attention mask
            )
            
            # Move all tensors to device and convert to float16
            features = {
                k: v.to(device).to(torch.float16) if torch.is_tensor(v) else v
                for k, v in features.items()
            }
            
            # Generate tokens with attention mask
            predicted_ids = model.generate(
                features['input_features'],
                attention_mask=features['attention_mask'],
                do_sample=True,
                max_new_tokens=400,
                no_repeat_ngram_size=3,
                return_timestamps=False,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                forced_decoder_ids=forced_decoder_ids
            )
            
            chunk_text = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            transcription.append(chunk_text)
        
        return " ".join(transcription)
    
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")

def main():
    """
    Example usage of the transcription function.
    """
    audio_path = "./holysonnet_01_donne.mp3"
    try:
        print("\nStarting transcription process...")
        transcription = transcribe_audio(audio_path, model_size="base", chunk_length_s=25)  # Reduced chunk size
        print("\nTranscription completed successfully!")
        print("\nTranscription:")
        print(transcription)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        if "FFmpeg" in str(e):
            print("\nPlease ensure FFmpeg is installed:")
            if platform.system() == "Darwin":  # macOS
                print("brew install ffmpeg")
            elif platform.system() == "Linux":
                print("sudo apt-get install ffmpeg")
            else:  # Windows
                print("Download FFmpeg from https://www.ffmpeg.org/download.html")
        print("\nIf FFmpeg is installed and you're still seeing this error,")
        print("try running 'which ffmpeg' to verify it's in your PATH")

if __name__ == "__main__":
    main()