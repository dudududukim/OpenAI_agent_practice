import asyncio
import random

import numpy as np
import sounddevice as sd
import time, json

from dotenv import load_dotenv

load_dotenv()

from agents import (
    Agent,
    function_tool,
    set_tracing_disabled,
)
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."

korean_agent = Agent(
    name="Korean",
    handoff_description="A Korean speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Korean.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Korean, handoff to the korean agent.",
    ),
    model="gpt-4o-mini",
    handoffs=[korean_agent],
    tools=[get_weather],
)

CSI = "\033["
COL = {
    "audio": "38;5;39m",
    "lc": "38;5;214m",
    "err": "38;5;196m",
    "txt": "38;5;46m",
    "dim": "2m",
    "reset": "0m",
}

def c(s, key): return f"{CSI}{COL[key]}{s}{CSI}{COL['reset']}"
def now(): return time.strftime("%H:%M:%S")
def pretty_payload(e):
    d = {k: v for k, v in getattr(e, "__dict__", {}).items() if k not in ("data",)}
    return json.dumps(d, ensure_ascii=False, default=lambda o: getattr(o, "__dict__", str(o)))

async def main():
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    buffer = np.zeros(24000 * 3, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)

    result = await pipeline.run(audio_input)

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    # Play the audio stream as it comes in
    async for event in result.stream():     # An event from the VoicePipeline, streamed via StreamedAudioResult.stream()
        t = event.type
        if t == "voice_stream_event_audio":
            player.write(event.data)
        elif t == "voice_stream_event_lifecycle":
            print(f"{now()} {c('[LIFECYCLE]', 'lc')} {pretty_payload(event)}")
            pass
        elif t == "voice_stream_event_error":
            print("error:", getattr(event, "message", event))

if __name__ == "__main__":
    asyncio.run(main())