import asyncio
import random
import numpy as np
import sounddevice as sd
import time, json
from dotenv import load_dotenv
import time as _time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from cumpa_workflow import MyWorkflow

load_dotenv()

from agents import (
    Agent, function_tool, Runner, ItemHelpers,
    enable_verbose_stdout_logging, set_tracing_export_api_key,
)
from agents.voice import (
    StreamedAudioInput, VoicePipeline, VoicePipelineConfig, SingleAgentVoiceWorkflow, VoiceWorkflowBase
)

from agents.voice.model import STTModelSettings, TTSModelSettings

# --- begin: STT event handler hotfix ---
from agents.voice.models import openai_stt as _stt
from agents.voice.models.openai_stt import OpenAISTTTranscriptionSession

def patch_stt_event_handler():
    async def _patched_handle_events(self):
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=_stt.EVENT_INACTIVITY_TIMEOUT)
                if isinstance(event, _stt.WebsocketDoneSentinel):
                    break
                et = event.get("type", "unknown")
                if et == "input_audio_buffer.speech_stopped":
                   payload = {
                       "event": "user_speech_end",
                       "type": "voice_stream_event_lifecycle",
                       "epoch_ms": int(_time.time() * 1000),
                       "source": "speech_stopped"
                   }
                   print(f"{now()} {c('[LIFECYCLE]', 'lifecycle')}" + json.dumps(payload, ensure_ascii=False))

                if et in ("input_audio_transcription_completed",
                          "conversation.item.input_audio_transcription.completed"):
                    tx = (event.get("transcript") or "").strip()
                    if tx:
                        self._end_turn(tx)
                        self._start_turn()
                        await self._output_queue.put(tx)
                await asyncio.sleep(0)
            except asyncio.TimeoutError:
                break
            except Exception as e:
                await self._output_queue.put(_stt.ErrorSentinel(e))
                raise
        await self._output_queue.put(_stt.SessionCompleteSentinel())
    OpenAISTTTranscriptionSession._handle_events = _patched_handle_events
# --- end: STT event handler hotfix ---

from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# enable_verbose_stdout_logging()

sv_vad = {
    "type": "server_vad",
    "threshold": 0.5,           # 0.3~0.6에서 시작해 튠
    "silence_duration_ms": 600, # 450~800ms 권장 범위
    "prefix_padding_ms": 300
}

sm_vad = {
    "type": "semantic_vad"
}

cfg = VoicePipelineConfig(
    stt_settings=STTModelSettings(
        language="ko",
        turn_detection=sm_vad
    ),
    tts_settings=TTSModelSettings(
        # OpenAI default enum: 'alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer')
        voice="alloy",
        # tone
        instructions=(
            "Speak in Korean with a consistent, neutral tone and stable pitch. "
            "Do not change style, accent, or expressiveness mid-utterance."
        ),
        speed=2.0,
        buffer_size=120,
    )
)

CHUNK_SEC = 0.05  # 50ms
IN_SR = 24000     # prefer 24k to minimize resampling
OUT_SR = 24000
DTYPE = np.int16
CHANNELS = 1

def on_transcription_start(text: str):
    payload = {
        "event": "transcription_start",
        "transcript": text,
    }
    print(f"{now()} {c('[ON_START]', 'info')} {json.dumps(payload, ensure_ascii=False)}")

workflow = MyWorkflow(secret_word="쿰파", on_start=on_transcription_start)

CSI = "\033["
COL = {
    "mic": "38;5;39m",       
    "audio": "38;5;46m",     
    "lifecycle": "38;5;214m",
    "error": "38;5;196m",    
    "debug": "38;5;244m",    
    "info": "38;5;33m",      

    # reset
    "reset": "0m",
}


def c(s, key): return f"{CSI}{COL[key]}{s}{CSI}{COL['reset']}"
def now(): return datetime.now().strftime("%H:%M:%S.%f")[:-3]
def pretty_payload(e):
    d = {k: v for k, v in getattr(e, "__dict__", {}).items() if k not in ("data",)}
    return json.dumps(d, ensure_ascii=False, default=lambda o: getattr(o, "__dict__", str(o)))

async def main():
    pipeline = VoicePipeline(
        workflow=workflow,
        config=cfg,
    )
    
    """
    StreamedAudioInput is used when you might need to detect when a user is done speaking. 
    It allows you to push audio chunks as they are detected, 
    and the voice pipeline will automatically run the agent workflow at the right time, 
    via a process called "activity detection".
    """
    audio_input = StreamedAudioInput()

    # create mic stream to attach to StreamedAudioInput()
    mic_stream = sd.InputStream(samplerate=IN_SR, channels=CHANNELS, dtype=DTYPE)
    mic_stream.start()

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=OUT_SR, channels=CHANNELS, dtype=DTYPE)
    player.start()

    play_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-play")
    loop = asyncio.get_running_loop()

    async def stream_mic():
        chunk_size = int(IN_SR * CHUNK_SEC)
        while True:
            if mic_stream.read_available < chunk_size:
                await asyncio.sleep(0.01)
                continue
            data, _ = mic_stream.read(chunk_size)
            await audio_input.add_audio(data)
            await asyncio.sleep(0)

    mic_task = asyncio.create_task(stream_mic())

    # Play the audio stream as it comes in
    try:
        print(f"{now()} {c('[MIC START]', 'mic')} Conversation started. Listening for input...")
        result = await pipeline.run(audio_input)

        async for event in result.stream():     # An event from the VoicePipeline, streamed via StreamedAudioResult.stream()
            t = event.type
            if t == "voice_stream_event_audio":
                # player.write(event.data)
                print(f"{now()} {c('[AUDIO PLAYING]', 'audio')} {pretty_payload(event)}")
                await loop.run_in_executor(play_exec, player.write, event.data)
            elif t == "voice_stream_event_lifecycle":
                print(f"{now()} {c('[LIFECYCLE]', 'lifecycle')} {pretty_payload(event)}")
                pass
            elif t == "voice_stream_event_error":
                payload = {
                    "event": "error",
                    "message": getattr(event, "message", str(event)),
                }
                print(f"{now()} {c('[ERROR]', 'error')} {json.dumps(payload, ensure_ascii=False)}")
    finally:
        mic_task.cancel()

        mic_stream.stop()
        mic_stream.close()
        player.stop()
        player.close()

if __name__ == "__main__":
    patch_stt_event_handler()
    asyncio.run(main())