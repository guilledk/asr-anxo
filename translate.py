import torch
from transformers import pipeline
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import pysrt
from pysrt import SubRipTime
import json

import os
import sys
from tqdm import tqdm


import subprocess


def download_and_extract_audio(url):
    # Prepare command
    command = [
        'yt-dlp',
        '-x',  # Extract audio
        '-k',  # Keep video
        '--audio-format', 'mp3',  # Convert to mp3
        '-o', 'input.mp3',  # Output filename
        url  # URL to download
    ]

    # Execute command
    print(f'downloading {url}...')
    subprocess.run(command, check=True)
    print('done.')


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

    # Sort files numerically
    wav_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    for wav_file in tqdm(wav_files, desc='Processing audio files'):
        file_path = os.path.join(input_folder, wav_file)

        # Load the wav file
        waveform, rate = librosa.load(file_path, sr=None, mono=True)

        # Resample to 16 kHz
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=16000)

        # Convert to int16 array
        # waveform = (32767*waveform).astype(np.int16)

        # Prepare the input dictionary
        sample = {
            "raw": waveform,
            "sampling_rate": 16000
        }

        # Process the audio file and print the output
        prediction = asr(sample, batch_size=8, return_timestamps=True)["chunks"]

        # Generate the output file path
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(wav_file)[0]}.json')

        # Write the output to a text file
        with open(output_file_path, 'w') as f:
            f.write(json.dumps(prediction, indent=4))


def load_and_adjust_transcriptions(directory):
    transcriptions = []
    for filename in sorted(os.listdir(directory), key=lambda x: int(x.split(".")[0])):  # Sorting files by their integer name
        try:
            with open(f'{directory}/{filename}', 'r') as file:
                minute_transcriptions = json.load(file)
            # Adjust the timestamps
            for transcription in minute_transcriptions:
                start_time, end_time = transcription["timestamp"]
                start_time += int(filename.split(".")[0]) * 60  # Add the minutes offset
                end_time += int(filename.split(".")[0]) * 60
                transcription["timestamp"] = (start_time, end_time)
            transcriptions.extend(minute_transcriptions)
        except BaseException as e:
            print(e)

    return transcriptions



def generate_srt(transcriptions, output_file):
    subs = pysrt.SubRipFile()
    for i, transcription in enumerate(transcriptions):
        start_time = SubRipTime(seconds=transcription["timestamp"][0])
        end_time = SubRipTime(seconds=transcription["timestamp"][1])
        text = transcription["text"]
        sub = pysrt.SubRipItem(i, start=start_time, end=end_time, text=text)
        subs.append(sub)
    subs.save(output_file, encoding='utf-8')


if __name__ == '__main__':
    download_and_extract_audio(sys.argv[1])
    split_audio('input.mp3', 'splited-input')
    process_audio_files('splited-input', 'predictions')
    transcriptions = load_and_adjust_transcriptions('predictions')
    generate_srt(transcriptions, 'test.srt')
