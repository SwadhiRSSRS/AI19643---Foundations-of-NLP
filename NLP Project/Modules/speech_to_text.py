import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# === Settings
DURATION = 20  # Seconds to record
MIC_FILENAME = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\mic_recording.wav"
OUTPUT_TEXT = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\raw_text.txt"

# === Load Whisper model
model = whisper.load_model("small")  # Options: "tiny", "base", "small", "medium", "large"


# === Option 1: Record from Microphone
def record_audio(duration=DURATION, filename=MIC_FILENAME):
    print(f" Recording for {duration} seconds...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(f" Mic audio saved to {filename}")
    return filename

# === Option 2: Transcribe from any file (mp3, wav, etc.)
def transcribe_file(audio_path,output_path=OUTPUT_TEXT):
    print(f" Transcribing audio from file: {audio_path}")
    result = model.transcribe(audio_path)
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(result["text"])
    print(f" Transcription saved to {output_path}")
    return result["text"]


# === Run this
if __name__ == "__main__":
    use_mic=int(input("Use microphone? (1/0): "))==1
    # print(use_mic)
    audio_file=record_audio() if use_mic else input("Enter audio file path: ").strip()
    transcribe_file(audio_file)
