import asyncio
import os
import wave

from pipecat.frames.frames import TextFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.openai.tts import OpenAITTSService

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Install python-dotenv or set environment variables manually.")


async def speak_text_from_file(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text_to_speak = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not text_to_speak.strip():
        print("The file is empty.")
        return

    print(f'Reading: "{text_to_speak[:50]}..."')

    async def text_frame_generator(text: str):
        yield TextFrame(text)
        yield EndFrame()

    # OpenAI TTS setup
    tts_service = OpenAITTSService(voice="alloy", model="tts-1")

    # Define a custom output processor to save audio to a WAV file
    class AudioFileOutput:
        def __init__(self, filename="output.wav", sample_rate=24000):
            self.filename = filename
            self.sample_rate = sample_rate
            self.audio_data = b""

        async def process(self, frame):
            if hasattr(frame, "audio"):
                self.audio_data += frame.audio

        async def finalize(self):
            with wave.open(self.filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(self.audio_data)
            print(f"Audio saved to {self.filename}")

    audio_output = AudioFileOutput()

    pipeline = Pipeline([tts_service, audio_output])

    try:
        await pipeline.run(input_frames=text_frame_generator(text_to_speak))
        await audio_output.finalize()
        print("Playback finished.")
    except Exception as e:
        print(f"Pipeline error: {e}")


async def main():
    await speak_text_from_file("response.txt")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
