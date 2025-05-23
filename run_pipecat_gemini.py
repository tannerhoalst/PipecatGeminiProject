import os
import asyncio
import logging
from dotenv import load_dotenv

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import (
    TextFrame,
    ErrorFrame,
)
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# Logging setup
logging.basicConfig(level=logging.INFO)
logging.getLogger("pipecat").setLevel(logging.DEBUG)

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("Error: GOOGLE_API_KEY not found.")
    logging.error("GOOGLE_API_KEY not found.")
    exit()


# Frame processor to log/print LLM output
class TextPrinter(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.received_text = []
        self.processed_frames = 0

    async def process_frame(self, frame: any, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.processed_frames += 1
        logging.info(
            f"TextPrinter received frame #{self.processed_frames}: {type(frame)} - {frame}"
        )

        if isinstance(frame, TextFrame):
            print(f"\nLLM Response: {frame.text}\n")
            logging.info(f"LLM Response (from TextPrinter): {frame.text}")
            self.received_text.append(frame.text)
        elif isinstance(frame, ErrorFrame):
            print(f"Error from pipeline: {frame.error}")
            logging.error(f"ErrorFrame: {frame.error}")

        await self.push_frame(frame, direction)


async def main():
    logging.info("Starting Pipecat Gemini voice pipeline...")

    try:
        # 1. Configure Audio Input with VAD
        vad_analyzer = SileroVADAnalyzer()

        # Create transport params instead of passing play_audio directly
        transport_params = LocalAudioTransportParams(
            audio_in_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            # audio_out_enabled=False by default; add audio_out_enabled=True if you need playback
        )
        audio_transport = LocalAudioTransport(params=transport_params)
        logging.info("LocalAudioTransport with VAD initialized via params.")

        # 2. Configure Speech-to-Text Service
        stt_service = GoogleSTTService()  # Uses default Google Cloud credentials
        logging.info("GoogleSTTService initialized.")

        # 3. Create the LLM service
        llm_service = GoogleLLMService(
            model="gemini-1.5-pro-latest",
            api_key=google_api_key,
        )
        logging.info("LLM service initialized.")

        # 4. Setup conversation context
        messages = []  # You can seed with a system prompt here
        context = OpenAILLMContext(messages)
        context_aggregator = llm_service.create_context_aggregator(context)

        # 5. Create output handler
        text_printer = TextPrinter()

        # 6. Assemble the pipeline
        pipeline = Pipeline(
            [
                audio_transport.input(),  # Audio from microphone
                stt_service,  # Speech to Text -> TranscriptionFrame
                context_aggregator.user(),  # Add user transcription to context
                llm_service,  # LLM processes context
                text_printer,  # Prints LLM’s TextFrame response
                context_aggregator.assistant(),  # Feed LLM response back into context
            ]
        )
        logging.info("Pipeline assembled.")

        task = PipelineTask(pipeline)
        runner = PipelineRunner()

        logging.info("Running pipeline... Speak into your microphone.")
        await runner.run(task)  # Runs until Ctrl+C or an EndFrame propagates

        logging.info("Pipeline completed or stopped.")

        if not text_printer.received_text:
            logging.warning("No LLM response was captured.")
        else:
            logging.info(f"Captured responses: {text_printer.received_text}")

    except KeyboardInterrupt:
        logging.info("Pipeline stopped by user (Ctrl+C).")
    except Exception as e:
        logging.error(f"Pipeline execution error: {e}", exc_info=True)
    finally:
        logging.info("Main function completed.")


if __name__ == "__main__":
    asyncio.run(main())
