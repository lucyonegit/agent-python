from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

TaskStatus = Literal['pending', 'doing', 'done']

@dataclass
class TaskStep:
    id: str
    title: str
    status: TaskStatus
    note: Optional[str] = None

@dataclass
class ReActStep:
    type: Literal['thought', 'action', 'observation']
    content: str
    toolName: Optional[str] = None
    toolInput: Any = None
    toolOutput: Any = None

@dataclass
class AgentConfig:
    model: str = 'gpt-4'
    temperature: float = 0.7
    maxTokens: int = 2000
    maxIterations: int = 10
    streamOutput: bool = True
    language: Literal['auto', 'chinese', 'english'] = 'auto'
    pauseAfterEachStep: bool = False
    autoPlanOnStart: bool = True
    autoGenerateFinalAnswer: bool = True
    strictActionUntilDone: bool = True

@dataclass
class ConversationEvent:
    id: str
    role: str
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    content: str = ''
    stream: Optional[bool] = None
    done: Optional[bool] = None

@dataclass
class StreamEvent:
    sessionId: str
    conversationId: str
    event: ConversationEvent
    timestamp: int

@dataclass
class AgentContext:
    input: str
    steps: List[ReActStep]
    tools: Dict[str, Any]
    config: AgentConfig
