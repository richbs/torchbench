import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

def load_audio(file_path, sample_rate=16000):
    """
    Load an audio file and resample it to 16kHz for Whisper.
    
    Args:
        file_path (str): Path to the audio file
        sample_rate (int): Target sample rate (Whisper expects 16kHz)
    
    Returns:
        torch.Tensor: Audio waveform
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

def transcribe_audio(file_path, model_size="base"):
    """
    Transcribe audio file to text using OpenAI's Whisper model.
    
    Args:
        file_path (str): Path to the audio file
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
        str: Transcribed text
    """
    # Load the audio file
    waveform = load_audio(file_path)
    
    # Load Whisper model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"openai/whisper-{model_size}"
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Process the audio
    input_features = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate tokens
    predicted_ids = model.generate(input_features)
    
    # Decode the tokens to text
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    return transcription

def main():
    """
    Example usage of the transcription function.
    """
    # Example usage
    audio_path = "./holysonnet_01_donne.mp3"
    try:
        transcription = transcribe_audio(audio_path, model_size="base")
        print(f"Transcription: {transcription}")
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()