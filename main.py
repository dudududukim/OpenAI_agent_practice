import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

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
from text_splitter import cumpa_splitter
from event_hub import EventHub, EventType, UnifiedEvent
from event_hub import EventType, UnifiedEvent, event_hub

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
                
                # 🔥 STT 이벤트를 EventHub로 전달
                if et == "input_audio_buffer.speech_started":
                    hub_event = UnifiedEvent(
                        type=EventType.USER_SPEECH_START,
                        timestamp=_time.time() * 1000,
                        source="stt"
                    )
                    await event_hub.publish(hub_event)
                    
                elif et == "input_audio_buffer.speech_stopped":
                    payload = {
                        "event": "user_speech_end",
                        "type": "voice_stream_event_lifecycle", 
                        "epoch_ms": int(_time.time() * 1000),
                        "source": "speech_stopped"
                    }
                    # print(f"{now()} {c('[LIFECYCLE]', 'lifecycle')}" + json.dumps(payload, ensure_ascii=False))
                    
                    # EventHub로 전달
                    hub_event = UnifiedEvent(
                        type=EventType.USER_SPEECH_END,
                        timestamp=_time.time() * 1000,
                        source="stt"
                    )
                    await event_hub.publish(hub_event)
                    
                if et in ("input_audio_transcription_completed",
                         "conversation.item.input_audio_transcription.completed"):
                    tx = (event.get("transcript") or "").strip()
                    if tx:
                        # EventHub로 전사 완료 이벤트 전달
                        hub_event = UnifiedEvent(
                            type=EventType.TRANSCRIPTION_DONE,
                            timestamp=_time.time() * 1000,
                            data={"transcript": tx},
                            source="stt"
                        )
                        await event_hub.publish(hub_event)
                        
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
        language="ko",      # not working
        turn_detection=sm_vad,
        prompt="The following audio contains a conversation in the Korean language. Please transcribe the speech accurately."
    ),
    tts_settings=TTSModelSettings(
        # OpenAI default enum: 'alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer')
        voice="alloy",
        # tone
        instructions="You will receive partial sentences in Korean; do not complete the sentence, just read out the text you are given.",
        speed=1.3,
        buffer_size=120,
        # text_splitter=cumpa_splitter()
    )
)

pipeline = VoicePipeline(
    workflow=MyWorkflow(secret_word="쿰파"),
    stt_model='gpt-4o-mini-transcribe',     # "whisper-1" | "gpt-4o-transcribe" | "gpt-4o-mini-transcribe"
    tts_model='gpt-4o-mini-tts',
    config=cfg,
)

CHUNK_SEC = 0.05  # 50ms
IN_SR = 24000     # prefer 24k to minimize resampling
OUT_SR = 24000
DTYPE = np.int16
CHANNELS = 1


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

    async def handle_unified_events():
        """EventHub에서 오는 통합 이벤트를 처리"""
        async for event in event_hub.subscribe():
            if event.type == EventType.BARGE_IN_DETECTED:
                print(f"{now()} {c('[BARGE-IN DETECTED]', 'error')} Interrupted at {event.data['interrupted_at']}")
                # Barge-in logic
                
            elif event.type == EventType.TRANSCRIPTION_DONE:
                print(f"{now()} {c('[TRANSCRIPTION]', 'info')} {event.data['transcript']}")
                
            elif event.type == EventType.USER_SPEECH_START:
                print(f"{now()} {c('[USER SPEECH START]', 'mic')} User started speaking")
                
            elif event.type == EventType.USER_SPEECH_END:
                print(f"{now()} {c('[USER SPEECH END]', 'mic')} User stopped speaking")
                
            elif event.type == EventType.AGENT_SPEECH_START:
                print(f"{now()} {c('[AGENT SPEECH START]', 'audio')} Agent started speaking")

            elif event.type == EventType.AGENT_SPEECH_END:
                print(f"{now()} {c('[AGENT SPEECH END]', 'audio')} Agent stopped speaking")
                
            elif event.type == EventType.AGENT_CHANGED:
                payload = {
                    "event": "agent_changed", 
                    "from": event.data["from"], 
                    "to": event.data["to"]
                }
                print(f"{now()} {c('[Agent Changed]', 'agent')}")
                # print(f"{now()} {c('[Agent Changed]', 'agent')} {json.dumps(payload, ensure_ascii=False)}")

    event_task = asyncio.create_task(handle_unified_events())

    # Play the audio stream as it comes in
    try:
        print(f"{now()} {c('[MIC START]', 'mic')} Conversation started. Listening for input...")
        result = await pipeline.run(audio_input)
        _current_pipeline_result = result

        last_segment_id = None
        current_turn = None

        async for event in result.stream():     # An event from the VoicePipeline, streamed via StreamedAudioResult.stream()
            t = event.type

            if t == "voice_stream_event_audio":
                if event.segment_id != last_segment_id:
                    last_segment_id = event.segment_id
                if event.turn_num != current_turn:
                    current_turn = event.turn_num
                    print(f"{now()} {c(f'[TURN {event.turn_num}]', 'audio')} Segment {event.segment_id}")
                await loop.run_in_executor(play_exec, player.write, event.data)

            elif t == "voice_stream_event_lifecycle":
                if event.event == "turn_started":
                    hub_event = UnifiedEvent(
                        type=EventType.AGENT_SPEECH_START,
                        timestamp=time.time() * 1000,
                        source="tts"
                    )
                    await event_hub.publish(hub_event)

                elif event.event == "turn_ended":
                    await event_hub.publish(UnifiedEvent(
                        type=EventType.AGENT_SPEECH_END,
                        timestamp=time.time() * 1000,
                        source="tts"
                    ))
                    
                pass

            elif t == "voice_stream_event_error":
                payload = {
                    "event": "error",
                    "message": getattr(event, "message", str(event)),
                }
                print(f"{now()} {c('[ERROR]', 'error')} {json.dumps(payload, ensure_ascii=False)}")
    finally:
        mic_task.cancel()
        event_task.cancel()

        mic_stream.stop()
        mic_stream.close()
        player.stop()
        player.close()

if __name__ == "__main__":
    patch_stt_event_handler()         # -> un-comment this : if you are using without modifying '.venv/lib/python3.12/site-packages/agents/voice/models/openai_stt.py' _handle_events
    asyncio.run(main())