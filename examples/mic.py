# --- begin: STT event handler hotfix ---
import asyncio
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



import os, asyncio, time
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

from agents import enable_verbose_stdout_logging, set_tracing_export_api_key, trace
from agents.voice import StreamedAudioInput
from agents.voice.model import STTModelSettings
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider

SR = 24000
CHANNELS = 1
DTYPE = np.int16
CHUNK_SEC = 0.05
LANG = "ko"
# TURN_DETECTION = {"type": "semantic_vad"}
TURN_DETECTION = {
  "type": "server_vad",
  "threshold": 0.4,
  "prefix_padding_ms": 300,    # 120~300 권장 시작 구간
  "silence_duration_ms": 450
}

def now(): return time.strftime("%H:%M:%S")

async def main():
    # enable_verbose_stdout_logging()
    set_tracing_export_api_key(os.environ.get("OPENAI_API_KEY", ""))  # upload spans to Traces

    audio_in = StreamedAudioInput()
    stt = OpenAIVoiceModelProvider().get_stt_model("gpt-4o-transcribe")

    # open a trace to avoid "No active trace..."
    with trace("Mic STT smoke test", group_id="local-dev", metadata={"sr": f"{SR}", "lang": LANG}):
        session = await stt.create_session(
            input=audio_in,
            settings=STTModelSettings(language=LANG, turn_detection=TURN_DETECTION),
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )

        async def pump_mic():
            block = int(SR * CHUNK_SEC)
            with sd.InputStream(samplerate=SR, channels=CHANNELS, dtype=DTYPE, blocksize=block) as mic:
                print("🎤 Speak now (Ctrl+C to stop)…")
                silence = 0
                while True:
                    buf, _ = mic.read(block)              # buf: (frames, channels) int16
                    if CHANNELS == 1:
                        arr = buf.reshape(-1)
                    else:
                        arr = buf.mean(axis=1).astype(np.int16)
                    await audio_in.add_audio(arr)         # expects np.int16 / np.float32 NDArray
                    peak = int(np.max(np.abs(arr))) if arr.size else 0
                    if peak < 200:
                        silence += 1
                        if silence % 40 == 0:
                            print(f"[{now()}] ~silence {silence*CHUNK_SEC:.1f}s")
                    else:
                        if silence >= 40:
                            print(f"[{now()}] sound after {silence*CHUNK_SEC:.1f}s silence (peak={peak})")
                        silence = 0
                    await asyncio.sleep(0)

        async def consume_transcripts():
            # https://openai.github.io/openai-agents-python/ref/voice/model/#agents.voice.model.StreamedTranscriptionSession
            async for text in session.transcribe_turns():  # yields per-turn finals; returns after close()
                # print("Help!")
                if text.strip():
                    print(f"[{now()}] ❯ {text}")

        mic_task = asyncio.create_task(pump_mic())
        rx_task = asyncio.create_task(consume_transcripts())
        try:
            await asyncio.gather(mic_task, rx_task)
        except KeyboardInterrupt:
            pass
        finally:
            await session.close()
            for t in (mic_task, rx_task):
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

if __name__ == "__main__":
    import contextlib
    patch_stt_event_handler()
    asyncio.run(main())
