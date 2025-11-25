from typing import Any, Dict, List, Optional, AsyncGenerator

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_community.chat_models.tongyi import ChatTongyi as Tongyi
except Exception:
    Tongyi = None
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
except Exception:
    SystemMessage = None
    HumanMessage = None
    AIMessage = None

class BaseChatModel:
    async def invoke(self, messages: List[Any]) -> Dict[str, Any]:
        raise NotImplementedError
    async def stream(self, messages: List[Any]) -> AsyncGenerator[Dict[str, Any], None]:
        raise NotImplementedError

import os

class LangChainLLM(BaseChatModel):
    def __init__(self, model: str, temperature: float, max_tokens: int, streaming: bool):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self._lc = None
        print(f'------------------init langchain llm----------------------- model: {model}, temperature: {temperature}, max_tokens: {max_tokens}, streaming: {streaming}')

        if ChatOpenAI is None and Tongyi is None:
            raise RuntimeError('langchain-openai or tongyi not installed')
        name = (model or '').lower()
        if 'qwen' in name or 'tongyi' in name:
            if Tongyi is None:
                raise RuntimeError('langchain-community Tongyi not installed')
            self._lc = Tongyi(model_name=model, temperature=temperature, dashscope_api_key=os.environ.get('DASHSCOPE_API_KEY'))
            print('------------------use tongyi-----------------------')
        else:
            self._lc = ChatOpenAI(model=model, temperature=temperature)

    def _to_lc_messages(self, messages: List[Dict[str, Any]]):
        out = []
        for m in messages:
            role = m.get('role')
            content = m.get('content', '')
            if role == 'system':
                out.append(SystemMessage(content=content))
            elif role == 'user':
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
        return out

    async def invoke(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        lc_messages = self._to_lc_messages(messages)
        resp = await self._lc.ainvoke(lc_messages)
        return {'content': getattr(resp, 'content', '')}

    async def stream(self, messages: List[Dict[str, Any]]):
        lc_messages = self._to_lc_messages(messages)
        async for chunk in self._lc.astream(lc_messages):
            yield {'content': getattr(chunk, 'content', '')}
