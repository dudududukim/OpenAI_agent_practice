from enum import Enum
from dataclasses import dataclass
from typing import Optional, AsyncIterator
import asyncio
import weakref

class EventType(Enum):
    USER_SPEECH_START = "user_speech_start"
    USER_SPEECH_END = "user_speech_end"
    TRANSCRIPTION_DONE = "transcription_done"
    AGENT_SPEECH_START = "agent_speech_start"
    AGENT_SPEECH_END = "agent_speech_end"
    BARGE_IN_DETECTED = "barge_in_detected"
    BARGE_IN_RESOLVED = "barge_in_resolved"
    AGENT_CHANGED = "agent_changed"

@dataclass
class UnifiedEvent:
    type: EventType
    timestamp: float
    data: Optional[dict] = None
    source: str = "unknown"

class EventHub:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._queue = asyncio.Queue(maxsize=1000)  # 큐 크기 제한
        self._subscribers = weakref.WeakSet()
        self._agent_speaking = False
        self._user_speaking = False
        self._running = True
        self._initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def publish(self, event: UnifiedEvent):
        if not self._running:
            return
        
        try:
            self._queue.put_nowait(event)  # non-blocking put
            await self._detect_barge_in(event)
        except asyncio.QueueFull:
            # 큐가 가득 차면 가장 오래된 이벤트 제거
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass

    async def subscribe(self) -> AsyncIterator[UnifiedEvent]:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    async def _detect_barge_in(self, event: UnifiedEvent):
        if event.type == EventType.USER_SPEECH_START:
            self._user_speaking = True
            if self._agent_speaking:
                barge_in_event = UnifiedEvent(
                    type=EventType.BARGE_IN_DETECTED,
                    timestamp=event.timestamp,
                    data={"interrupted_at": event.timestamp}
                )
                try:
                    self._queue.put_nowait(barge_in_event)
                except asyncio.QueueFull:
                    pass
        elif event.type == EventType.USER_SPEECH_END:
            self._user_speaking = False
        elif event.type == EventType.AGENT_SPEECH_START:
            self._agent_speaking = True
        elif event.type == EventType.AGENT_SPEECH_END:
            self._agent_speaking = False

    async def shutdown(self):
        self._running = False

event_hub = EventHub.get_instance()
