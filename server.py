import os
import asyncio
import json
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from core.react_agent import ReActAgent

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get('FRONTEND_ORIGIN', 'http://localhost:5173'), '*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ReActAgent({
    'model': os.environ.get('MODEL', 'qwen-plus'),
    'temperature': float(os.environ.get('TEMPERATURE', '0.7')),
    'streamOutput': True,
})

class RunRequest(BaseModel):
    input: str
    sessionId: Optional[str] = None
    conversationId: Optional[str] = None

@app.get('/health')
def health():
    return {'ok': True}

@app.post('/run')
async def run(req: RunRequest):
    events: list[Dict[str, Any]] = []
    def on_stream(e):
        events.append({
            'sessionId': e.sessionId,
            'conversationId': e.conversationId,
            'event': e.event,
            'timestamp': e.timestamp,
        })
    result = await agent.run_with_session(req.input, {
        'sessionId': req.sessionId,
        'conversationId': req.conversationId,
        'onStream': on_stream,
    })
    return {'result': result, 'events': events}

@app.get('/api/agent/stream')
async def agent_stream(request: Request):
    prompt = (request.query_params.get('prompt') or '')
    language = (request.query_params.get('language') or 'chinese')
    model = (request.query_params.get('model') or os.environ.get('MODEL') or 'qwen-plus')
    temperature = float(request.query_params.get('temperature') or os.environ.get('TEMPERATURE') or '0.7')
    session_id = request.query_params.get('sessionId') or None
    conversation_id = request.query_params.get('conversationId') or None
    pause_after_each = (request.query_params.get('pauseAfterEachStep') == 'true')

    if not prompt:
        async def err_gen():
            yield 'event: stream_event\n'
            yield 'data: {"error":"prompt is required"}\n\n'
            yield 'event: done\n'
            yield 'data: {"ok":false}\n\n'
        return StreamingResponse(err_gen(), media_type='text/event-stream', headers={
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        })

    local_agent = ReActAgent({
        'model': model,
        'temperature': temperature,
        'streamOutput': True,
        'language': language,
        'pauseAfterEachStep': pause_after_each,
    })

    queue: asyncio.Queue = asyncio.Queue()

    def on_stream(e):
        payload = {
            'sessionId': e.sessionId,
            'conversationId': e.conversationId,
            'event': e.event,
            'timestamp': e.timestamp,
        }
        asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, ('stream_event', payload))

    async def run_agent():
        try:
            result = await local_agent.run_with_session(prompt, {
                'sessionId': session_id,
                'conversationId': conversation_id,
                'onStream': on_stream,
            })
            payload = {
                'ok': True,
                'sessionId': result['sessionId'],
                'conversationId': result['conversationId'],
                'isPaused': result['isPaused'],
                'message': '等待用户输入...' if result['isPaused'] else '对话完成'
            }
            await queue.put(('done', payload))
        except Exception as err:
            await queue.put(('stream_event', {
                'sessionId': session_id or 'error',
                'conversationId': 'error',
                'event': { 'id': f'error_{int(asyncio.get_event_loop().time()*1000)}', 'role': 'assistant', 'type': 'normal_event', 'content': str(err) },
                'timestamp': int(asyncio.get_event_loop().time()*1000)
            }))
            await queue.put(('done', { 'ok': False }))

    task = asyncio.create_task(run_agent())

    async def event_generator():
        try:
            while True:
                event, data = await queue.get()
                payload = data if isinstance(data, str) else json.dumps(data)
                yield f'event: {event}\n'
                yield f'data: {payload}\n\n'
                if event == 'done': break
        finally:
            task.cancel()

    return StreamingResponse(event_generator(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    })

@app.get('/api/coding-agent/stream')
async def coding_agent_stream(request: Request):
    prompt = (request.query_params.get('prompt') or '')
    model = (request.query_params.get('model') or os.environ.get('MODEL') or 'qwen-plus')
    temperature = float(request.query_params.get('temperature') or os.environ.get('TEMPERATURE') or '0')
    session_id = request.query_params.get('sessionId') or None
    conversation_id = request.query_params.get('conversationId') or None

    if not prompt:
        async def err_gen():
            yield 'event: stream_event\n'
            yield 'data: {"error":"prompt is required"}\n\n'
            yield 'event: done\n'
            yield 'data: {"ok":false}\n\n'
        return StreamingResponse(err_gen(), media_type='text/event-stream', headers={
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        })

    from coder_agent.core.coding_agent import CodingAgent
    agent = CodingAgent({
        'model': model,
        'temperature': temperature,
        'streamOutput': True,
        'language': 'chinese',
        'maxTokens': 4000,
        'maxIterations': 10,
        'pauseAfterEachStep': False,
        'autoPlanOnStart': False,
        'strictActionUntilDone': True
    })

    queue: asyncio.Queue = asyncio.Queue()

    def on_stream(e):
        payload = {
            'sessionId': e.sessionId,
            'conversationId': e.conversationId,
            'event': e.event,
            'timestamp': e.timestamp,
        }
        asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, ('stream_event', payload))

    async def run_agent():
        try:
            result = await agent.run(prompt, { 'sessionId': session_id, 'conversationId': conversation_id, 'onStream': on_stream })
            payload = { 'ok': True, 'result': result['finalAnswer'] }
            await queue.put(('done', payload))
        except Exception as err:
            await queue.put(('stream_event', {
                'sessionId': session_id or 'error',
                'conversationId': 'error',
                'event': { 'id': f'error_{int(asyncio.get_event_loop().time()*1000)}', 'role': 'assistant', 'type': 'normal_event', 'content': str(err) },
                'timestamp': int(asyncio.get_event_loop().time()*1000)
            }))
            await queue.put(('done', { 'ok': False }))

    task = asyncio.create_task(run_agent())

    async def event_generator():
        try:
            while True:
                event, data = await queue.get()
                payload = data if isinstance(data, str) else json.dumps(data)
                yield f'event: {event}\n'
                yield f'data: {payload}\n\n'
                if event == 'done': break
        finally:
            task.cancel()

    return StreamingResponse(event_generator(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '3333')))
