import numpy as np
import torch
import torchaudio as ta


def load_audio(wave_file: str):
    """
    :param wave_file: .wav file containing the audio input
    :return: 1 x T tensor containing input audio resampled to 16kHz
    """
    audio, sr = ta.load(wave_file)
    if not sr == 16000:
        audio = ta.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # normalize such that energy matches average energy of audio used in training
    audio = 0.01 * audio / torch.mean(torch.abs(audio))
    return audio

def audio_chunking(audio: torch.Tensor, frame_rate: int = 30, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = 16000 // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = torch.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size//2, audio.shape[-1]-chunk_size//2 + 1, samples_per_frame))
    audio = torch.cat([audio[:, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], dim=0)
    return audio

def spec_chunking(spec: np.ndarray, frame_rate: int = 30, chunk_size: int = 101, stride: int = 1):
    """
    :param spec: (n_mel, n_frame) ndarray containing normalized mel-spectrogram
    :return (num_chunks, n_mel, chunk_size)
    """
    melframe_per_frame = int(16000 / 160 / frame_rate)
    padding = chunk_size // 2
    spec = np.pad(spec, ((0, 0), (padding, padding)), constant_values=0)
    spec = spec[np.newaxis, :]
    if chunk_size % 2 == 0:
        anchor_points = list(range(chunk_size//2, spec.shape[-1]-chunk_size//2 + 1, melframe_per_frame))
        spec = np.concatenate([spec[:, :, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], axis=0)
    else:
        anchor_points = list(range(chunk_size//2, spec.shape[-1]-chunk_size//2, melframe_per_frame * stride))
        spec = np.concatenate([spec[:, :, i-chunk_size//2:i+chunk_size//2+1] for i in anchor_points], axis=0)
    return spec
