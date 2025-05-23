import asyncio
import os
import sounddevice as sd
from scipy.io.wavfile import write

from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.openai.stt import OpenAISTTService

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Install python-dotenv or set environment variables manually.")


# Function to record audio and save as WAV
def record_voice(filename="recording.wav", duration=5, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16"
    )
    sd.wait()
    write(filename, samplerate, recording)
    print(f"Recording saved to {filename}")


# Output handler to print recognized text
class PrintTranscript:
    async def process(self, frame):
        if hasattr(frame, "text"):
            print(f"\nTranscription: {frame.text}")

    async def finalize(self):
        print("Done.")


# Function to transcribe audio using OpenAI's Whisper
async def transcribe_audio(filename="recording.wav"):
    # Read the audio file
    with open(filename, "rb") as audio_file:
        audio_data = audio_file.read()

    # Initialize the STT service
    whisper = OpenAISTTService(model="whisper-1")
    output = PrintTranscript()

    # Create the pipeline
    pipeline = Pipeline([whisper, output])

    # Run the pipeline with the audio data
    try:
        await pipeline.run(input_frames=[audio_data])
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        await output.finalize()


# Main function
async def main():
    record_voice()
    await transcribe_audio()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
