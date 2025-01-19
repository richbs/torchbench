import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

def load_audio(file_path, sample_rate=16000):
    """
    Load an audio file and resample it to 16kHz for Whisper.
    """
    waveform, original_sample_rate = torchaudio.load(file_path)
    
    # Convert stereo to mono by averaging channels if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if original_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(
            original_sample_rate, sample_rate
        )
        waveform = resampler(waveform)
    
    return waveform

def chunk_audio(waveform, chunk_length_s=30, sample_rate=16000):
    """
    Split audio into chunks of chunk_length_s seconds.
    Returns a list of chunks.
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
    Processes audio in chunks to handle longer files.
    """
    # Load the audio file
    waveform = load_audio(file_path)
    
    # Load Whisper model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"openai/whisper-{model_size}"
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    
    # Split audio into chunks
    chunks = chunk_audio(waveform, chunk_length_s=chunk_length_s)
    
    # Process each chunk
    transcription = []
    for chunk in chunks:
        # Convert to numpy and process
        input_features = processor(
            chunk.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device).to(torch.float16)
        
        # Generate tokens
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=448
        )
        
        # Decode the tokens to text
        chunk_text = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        transcription.append(chunk_text)
    
    # Join all chunks with proper spacing
    full_transcription = " ".join(transcription)
    
    return full_transcription

def main():
    """
    Example usage of the transcription function.
    """
    # Example usage
    audio_path = "./holysonnet_01_donne.mp3"
    try:
        # Process in 30-second chunks (you can adjust this value)
        transcription = transcribe_audio(audio_path, model_size="base", chunk_length_s=30)
        print(f"Transcription: {transcription}")
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()