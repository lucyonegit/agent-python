from dataclasses import dataclass
from typing import Callable, List, Optional
import time

@dataclass
class StreamEvent:
    sessionId: str
    conversationId: str
    event: dict
    timestamp: int

class StreamManager:
    def __init__(self, max_buffer_size: int = 100):
        self.is_streaming = False
        self.event_buffer: List[StreamEvent] = []
        self.max_buffer_size = max_buffer_size
        self._handlers: List[Callable[[StreamEvent], None]] = []

    def start_stream(self) -> None:
        self.is_streaming = True
        self.event_buffer.clear()
        for h in self._handlers:
            h(StreamEvent(sessionId='', conversationId='', event={'type': 'stream_start'}, timestamp=int(time.time()*1000)))

    def end_stream(self) -> None:
        self.is_streaming = False
        for h in self._handlers:
            h(StreamEvent(sessionId='', conversationId='', event={'type': 'stream_end'}, timestamp=int(time.time()*1000)))

    def emit_stream_event(self, event: StreamEvent) -> None:
        self.event_buffer.append(event)
        for h in self._handlers:
            h(event)
        if len(self.event_buffer) > self.max_buffer_size:
            self.event_buffer.pop(0)

    @property
    def buffer(self) -> List[StreamEvent]:
        return list(self.event_buffer)

    @property
    def streaming(self) -> bool:
        return self.is_streaming

    def clear_buffer(self) -> None:
        self.event_buffer.clear()

    def add_handler(self, handler: Callable[[StreamEvent], None]) -> None:
        self._handlers.append(handler)

def create_console_stream_handler(prefix: str = '[STREAM]') -> Callable[[StreamEvent], None]:
    type_colors = {
        'normal_event': '\x1b[36m',
        'task_plan_event': '\x1b[33m',
        'tool_call_event': '\x1b[32m',
        'error': '\x1b[31m'
    }
    def handle(event: StreamEvent) -> None:
        ts = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(event.timestamp/1000))
        color = type_colors.get(event.event.get('type', ''), '\x1b[37m')
        print(f"{prefix} {ts} {color}[{event.event.get('type','').upper()}]\x1b[0m", event.event)
    return handle

