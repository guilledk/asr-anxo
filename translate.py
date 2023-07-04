import torch
from transformers import pipeline
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

import os
from tqdm import tqdm


def split_audio(file_path, target_folder, chunk_length_ms=60000):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Load and convert the audio to mono
    audio = AudioSegment.from_mp3(file_path).set_channels(1)

    # Get the length of the audio
    audio_length_ms = len(audio)

    # Calculate the number of chunks to be created
    num_chunks = audio_length_ms // chunk_length_ms + int(audio_length_ms % chunk_length_ms != 0)

    # Split the audio with a progress bar
    for i in tqdm(range(0, audio_length_ms, chunk_length_ms), total=num_chunks):
        # The start and end time for each chunk
        start_time = i
        end_time = i + chunk_length_ms

        # Get a chunk of the audio
        chunk = audio[start_time:end_time]

        # Export the chunk to a WAV file
        chunk_name = f"{i // chunk_length_ms}.wav"
        chunk_path = os.path.join(target_folder, chunk_name)
        chunk.export(chunk_path, format="wav")

    print(f"\nSaved chunks to {target_folder}")


def process_audio_files(input_folder, output_folder):
    device = "cpu"
    torch.set_num_threads(10)
    asr = pipeline(
      "automatic-speech-recognition",
      model="openai/whisper-large-v2",
      chunk_length_s=30,
      device=device,
    )

    # Get all wav files in the input directory
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

    for wav_file in tqdm(wav_files, desc='Processing audio files'):
        file_path = os.path.join(input_folder, wav_file)

        # Load the wav file
        waveform, rate = librosa.load(file_path, sr=None, mono=True)

        # Resample to 16 kHz
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=16000)

        # Convert to int16 array
        waveform = (32767*waveform).astype(np.int16)

        # Prepare the input dictionary
        sample = {
            "raw": waveform,
            "sampling_rate": 16000
        }

        # Process the audio file and print the output
        prediction = asr(sample, batch_size=8, return_timestamps=True)["chunks"]

        # Generate the output file path
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(wav_file)[0]}.txt')

        # Write the output to a text file
        with open(output_file_path, 'w') as f:
            f.write(str(prediction))


split_audio('input.mp3', 'splited-input')
process_audio_files('splited-input', 'predictions')
