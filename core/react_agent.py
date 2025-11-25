import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aitypes import AgentConfig, AgentContext, ReActStep, TaskStatus, TaskStep
from tools.tool_registry import ToolRegistry
from core.stream_manager import StreamManager, StreamEvent
from core.prompt import (
    create_language_prompt,
    create_system_prompt,
    create_pre_action_prompt,
    create_planner_prompt,
)
from core.llm import LangChainLLM, BaseChatModel

class SimpleLLM(BaseChatModel):
    async def invoke(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        content = 'Final Answer: 已完成'
        return {'content': content}
    async def stream(self, messages: List[Dict[str, Any]]):
        text = '正在处理...然后给出最终答案。'
        for ch in text:
            await asyncio.sleep(0.005)
            yield {'content': ch}

@dataclass
class SessionState:
    context: AgentContext
    currentIteration: int
    sessionId: str
    conversationId: str
    isPaused: bool
    waitingReason: Optional[str] = None

class ReActAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[BaseChatModel] = None):
        self.config = AgentConfig(**(config or {}))
        print(self.config)
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = LangChainLLM(model=self.config.model, temperature=self.config.temperature, max_tokens=self.config.maxTokens, streaming=self.config.streamOutput)
            except Exception:
                self.llm = SimpleLLM()
        self.tool_registry = ToolRegistry()
        self.stream_manager = StreamManager()
        self.plan_list: List[TaskStep] = []
        self.last_emitted_plan_snapshot: str = ''
        self.current_session_id: Optional[str] = None
        self.session_states: Dict[str, SessionState] = {}

    def gen_id(self, prefix: str) -> str:
        return f"{prefix}_{int(time.time()*1000)}_{str(time.time()).split('.')[1][:6]}"

    def mark_next_pending_doing(self, note: Optional[str] = None) -> bool:
        for p in self.plan_list:
            if p.status == 'pending':
                p.status = 'doing'
                if note:
                    p.note = note
                return True
        return False

    def mark_current_step_done(self, note: Optional[str] = None) -> bool:
        for p in self.plan_list:
            if p.status == 'doing':
                p.status = 'done'
                if note:
                    p.note = note
                return True
        return False

    def get_plan_snapshot(self) -> str:
        return json.dumps([{ 'id': p.id, 'title': p.title, 'status': p.status, 'note': p.note } for p in self.plan_list], ensure_ascii=False)

    def emit_plan_update(self, session_id: str, conversation_id: str, on_stream=None, force: bool = False) -> None:
        current_snapshot = self.get_plan_snapshot()
        if not force and current_snapshot == self.last_emitted_plan_snapshot:
            return
        event_id = self.gen_id('plan_update')
        self.emit('task_plan', { 'step': [p.__dict__ for p in self.plan_list] }, session_id, conversation_id, event_id, on_stream)
        self.last_emitted_plan_snapshot = current_snapshot

    async def run_with_session(self, input: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        session_id = options.get('sessionId') if options else (self.current_session_id or self.gen_id('sess'))
        self.current_session_id = session_id
        existing = self.session_states.get(session_id)
        if existing and existing.isPaused and options and options.get('conversationId'):
            conversation_id = options['conversationId']
            context = existing.context
            start_iteration = existing.currentIteration
            context.steps.append(ReActStep(type='observation', content=f"User provided additional input: {input}"))
            self.emit('normal', { 'content': f"💬 用户输入：{input}" }, session_id, conversation_id, self.gen_id('user_input'), options.get('onStream') if options else None)
            existing.isPaused = False
        else:
            conversation_id = self.gen_id('conv')
            context = AgentContext(input=input, steps=[], tools=self.tool_registry.get_all_tools(), config=self.config)
            start_iteration = 0
            self.plan_list = []
            self.last_emitted_plan_snapshot = ''
            await self.generate_pre_action_tip(input, conversation_id, session_id, options.get('onStream') if options else None)
            if self.config.autoPlanOnStart:
                await self.generate_plan(context, options.get('onStream') if options else None, conversation_id, session_id)
        result = await self.run_internal(context, session_id, conversation_id, options.get('onStream') if options else None, start_iteration)
        return { 'sessionId': session_id, 'conversationId': conversation_id, 'finalAnswer': result['finalAnswer'], 'isPaused': result['isPaused'] }

    async def run_internal(self, context: AgentContext, session_id: str, conversation_id: str, on_stream=None, start_iteration: int = 0) -> Dict[str, Any]:
        for iteration in range(start_iteration, self.config.maxIterations):
            try:
                react_result = await self.reason_and_act(context, on_stream, conversation_id, session_id, iteration + 1)
                context.steps.append(ReActStep(type='thought', content=react_result.get('thought','')))
                if react_result['type'] == 'final_answer':
                    changed = self.mark_current_step_done('✅ 已完成')
                    has_pending = any(p.status == 'pending' for p in self.plan_list)
                    if has_pending:
                        advanced = self.mark_next_pending_doing('📝 正在生成最终答案')
                        done_now = self.mark_current_step_done('✅ 已生成最终答案')
                        changed = changed or advanced or done_now
                    if changed:
                        self.emit_plan_update(session_id, conversation_id, on_stream, True)
                    self.emit('normal', { 'content': '**准备答案** - 已收集足够信息，正在生成最终答案...' }, session_id, conversation_id, f"prepare_answer_{iteration}", on_stream)
                    final_answer = await self.generate_final_answer(context, on_stream, conversation_id, session_id)
                    return { 'finalAnswer': final_answer, 'isPaused': False }
                if react_result['type'] == 'action':
                    if react_result.get('toolName') == 'wait_for_user_input':
                        self.session_states[session_id] = SessionState(context=context, currentIteration=iteration+1, sessionId=session_id, conversationId=conversation_id, isPaused=True, waitingReason=(react_result.get('toolInput') or {}).get('reason') or '需要更多信息')
                        self.emit('waiting_input', { 'message': (react_result.get('toolInput') or {}).get('message') or '请输入更多信息以继续...', 'reason': (react_result.get('toolInput') or {}).get('reason') }, session_id, conversation_id, self.gen_id('waiting'), on_stream)
                        return { 'finalAnswer': '', 'isPaused': True }
                    action_step = ReActStep(type='action', content=f"Using tool: {react_result.get('toolName')}", toolName=react_result.get('toolName'), toolInput=react_result.get('toolInput'))
                    context.steps.append(action_step)
                    tool_event_id = f"tool_{iteration}_{conversation_id}"
                    tool_started_at = int(time.time()*1000)
                    self.emit('tool_call', { 'id': tool_event_id, 'status': 'start', 'tool_name': react_result.get('toolName'), 'args': react_result.get('toolInput'), 'iteration': iteration, 'startedAt': tool_started_at }, session_id, conversation_id, tool_event_id, on_stream)
                    tool_result = await self.tool_registry.execute_tool(react_result.get('toolName'), react_result.get('toolInput'))
                    observation = f"Tool executed successfully. Result: {json.dumps(tool_result.get('result'))}" if tool_result.get('success') else f"Tool execution failed. Error: {tool_result.get('error')}"
                    context.steps.append(ReActStep(type='observation', content=observation, toolName=react_result.get('toolName'), toolOutput=tool_result))
                    tool_finished_at = int(time.time()*1000)
                    self.emit('tool_call', { 'id': tool_event_id, 'status': 'end', 'tool_name': react_result.get('toolName'), 'args': react_result.get('toolInput'), 'result': tool_result, 'success': tool_result.get('success'), 'startedAt': tool_started_at, 'finishedAt': tool_finished_at, 'durationMs': tool_finished_at - tool_started_at, 'iteration': iteration }, session_id, conversation_id, tool_event_id, on_stream)
                    await self.generate_observation(tool_result, react_result.get('toolName'), on_stream, conversation_id, session_id, iteration)
                    if tool_result.get('success'):
                        has_change = self.mark_current_step_done(f"✅ 已使用 {react_result.get('toolName')}")
                        tasks = (tool_result.get('result') or {}).get('tasks') or ((tool_result.get('result') or {}).get('plan') or {}).get('steps')
                        if isinstance(tasks, list) and tasks:
                            try:
                                steps = [TaskStep(id=s.get('id') or f"plan_{i+1}", title=s.get('title'), status='pending') for i, s in enumerate(tasks)]
                                self.plan_list = steps
                                has_change = True
                            except Exception:
                                pass
                        plan_update = (tool_result.get('result') or {}).get('planUpdate')
                        if plan_update:
                            before = json.dumps([p.__dict__ for p in self.plan_list], ensure_ascii=False)
                            complete_ids = plan_update.get('completeIds') or []
                            complete_titles = plan_update.get('completeTitles') or []
                            complete_all = bool(plan_update.get('completeAll'))
                            if complete_ids:
                                self.plan_list = [TaskStep(id=p.id, title=p.title, status=('done' if p.id in complete_ids else p.status), note=p.note) for p in self.plan_list]
                            if complete_titles:
                                self.plan_list = [TaskStep(id=p.id, title=p.title, status=('done' if any([json.dumps(t) and (t.lower() in p.title.lower()) for t in complete_titles]) else p.status), note=p.note) for p in self.plan_list]
                            if complete_all:
                                self.plan_list = [TaskStep(id=p.id, title=p.title, status='done', note=p.note) for p in self.plan_list]
                            has_change = has_change or before != json.dumps([p.__dict__ for p in self.plan_list], ensure_ascii=False)
                        if has_change:
                            self.emit_plan_update(session_id, conversation_id, on_stream, True)
                    if self.config.pauseAfterEachStep:
                        self.session_states[session_id] = SessionState(context=context, currentIteration=iteration+1, sessionId=session_id, conversationId=conversation_id, isPaused=True, waitingReason='等待用户确认是否继续')
                        self.emit('waiting_input', { 'message': '当前步骤已完成，请输入继续执行或提供新的指令...', 'reason': '人机协作模式 - 每步后等待确认' }, session_id, conversation_id, self.gen_id('waiting'), on_stream)
                        return { 'finalAnswer': '', 'isPaused': True }
            except Exception as e:
                self.emit('normal', { 'content': f"❌ 错误：{str(e)}" }, session_id, conversation_id, f"error_{iteration}", on_stream)
                raise
        final_answer = await self.generate_final_answer(context, on_stream, conversation_id, session_id)
        return { 'finalAnswer': final_answer, 'isPaused': False }

    def mark_all_pending_done(self, note: Optional[str] = None) -> None:
        self.plan_list = [TaskStep(id=p.id, title=p.title, status=('done' if p.status == 'pending' else p.status), note=(note or p.note)) for p in self.plan_list]

    async def generate_plan(self, context: AgentContext, on_stream=None, conversation_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
        language_prompt = create_language_prompt(self.config.language)
        try:
            response = await self.llm.invoke([{ 'role': 'system', 'content': create_planner_prompt(context.input) }])
            self.plan_list = [TaskStep(id=f"plan_{i+1}", title=t['title'], status='pending') for i, t in enumerate(json.loads(response['content']))]
            return
        except Exception:
            pass
        self.plan_list = [
            TaskStep(id='plan_1', title='分析问题与制定计划', status='pending'),
            TaskStep(id='plan_2', title='执行必要的工具动作获取信息', status='pending'),
            TaskStep(id='plan_3', title='整理观察并撰写答案', status='pending')
        ]

    async def generate_pre_action_tip(self, input: str, conversation_id: str, session_id: str, on_stream=None) -> str:
        pre_prompt = create_pre_action_prompt(input)
        stream = self.llm.stream([{ 'role': 'system', 'content': pre_prompt }, { 'role': 'user', 'content': input }])
        pre_action_event_id = self.gen_id('pre_action')
        tip = ''
        async for chunk in stream:
            tip += chunk.get('content') or ''
            self.emit('normal', { 'content': chunk.get('content') or '', 'stream': True }, session_id, conversation_id, pre_action_event_id, on_stream)
        return tip

    async def reason_and_act(self, context: AgentContext, on_stream=None, conversation_id: Optional[str] = None, session_id: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        has_doing = any(p.status == 'doing' for p in self.plan_list)
        if not has_doing:
            changed = self.mark_next_pending_doing('🤔 正在推理')
            if changed:
                self.emit_plan_update(session_id or 'default', conversation_id or 'default', on_stream)
        current_step = next((p for p in self.plan_list if p.status == 'doing'), None) or next((p for p in self.plan_list if p.status == 'pending'), None)
        tools_description = self.tool_registry.get_tools_description()
        system_prompt = self.build_react_prompt(current_step, tools_description)
        conversation_history = self.build_conversation_history(context)
        messages = [{ 'role': 'system', 'content': system_prompt }] + conversation_history
        response = await self.llm.invoke(messages)
        content = response.get('content') or ''
        parsed = self.parse_react_output(content)
        has_incomplete = any(p.status != 'done' for p in self.plan_list)
        if self.config.strictActionUntilDone and has_incomplete and parsed['type'] == 'final_answer':
            pending_titles = [p.title for p in self.plan_list if p.status != 'done']
            self.emit('normal', { 'content': f"⚠️ 检测到存在未完成的计划步骤，已阻止提前输出最终答案。待完成步骤：{'，'.join(pending_titles)}" }, session_id or 'default', conversation_id or 'default', self.gen_id('block_final'), on_stream)
            return { 'type': 'action', 'thought': parsed.get('thought',''), 'toolName': 'continue_thinking', 'toolInput': { 'reason': 'incomplete_plan', 'pending': pending_titles } }
        if parsed.get('thought') and on_stream:
            self.emit('normal', { 'content': f"💭[thought] 第{iteration or 1}次迭代 {parsed.get('thought')}" }, session_id or 'default', conversation_id or 'default', self.gen_id('thought'), on_stream)
        if parsed['type'] == 'action' and parsed.get('toolName') and on_stream:
            friendly = self.format_friendly_tool_message(parsed.get('toolName'), parsed.get('toolInput'))
            if friendly:
                self.emit('normal', { 'content': f"[toolcall：{parsed.get('toolName')}] ｜ {friendly}" }, session_id or 'default', conversation_id or 'default', self.gen_id('action'), on_stream)
        return parsed

    def parse_react_output(self, content: str) -> Dict[str, Any]:
        thought = ''
        try:
            import re
            t = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|$)", content, re.S)
            thought = t.group(1).strip() if t else ''
            if 'Final Answer:' in content:
                m = re.search(r"Final Answer:\s*(.+)", content, re.S)
                final_answer = m.group(1).strip() if m else ''
                return { 'type': 'final_answer', 'thought': thought, 'content': final_answer }
            if 'Action:' in content:
                am = re.search(r"Action:\s*([^\n]+)", content)
                im = re.search(r"Input:\s*(.+)", content, re.S)
                if am:
                    raw_tool = am.group(1).strip()
                    tool_name = raw_tool
                    tool_input = {}
                    raw_input_str = None
                    if im:
                        raw_input_str = im.group(1).strip()
                        try:
                            jm = re.search(r"\{[\s\S]*\}", raw_input_str)
                            if jm:
                                tool_input = json.loads(jm.group(0))
                            else:
                                tool_input = json.loads(raw_input_str)
                        except Exception:
                            tool_input = { 'input': raw_input_str }
                    if tool_name.lower().strip() == 'final answer':
                        answer_text = tool_input if isinstance(tool_input, str) else (tool_input.get('input') if isinstance(tool_input, dict) else (raw_input_str or ''))
                        return { 'type': 'final_answer', 'thought': thought, 'content': str(answer_text).strip() }
                    return { 'type': 'action', 'thought': thought, 'toolName': tool_name, 'toolInput': tool_input }
        except Exception:
            pass
        return { 'type': 'action', 'thought': content, 'toolName': 'continue_thinking', 'toolInput': { 'thought': content } }

    def build_react_prompt(self, current_step: Optional[TaskStep] = None, tools_description: Optional[str] = None) -> str:
        language_instructions = create_language_prompt(self.config.language)
        base_prompt = create_system_prompt(language_instructions, tools_description)
        if current_step:
            remaining = '\n'.join([f"- {p.title}" for p in self.plan_list if p.status != 'done']) or '- 无'
            return (
                f"{base_prompt}\n\n**当前任务步骤**: {current_step.title}\n请专注完成当前步骤，并优先使用工具执行所需操作。\n在所有计划步骤完成之前，请勿输出 Final Answer；完成当前步骤后再推进到下一步。\n\n剩余步骤:\n{remaining}"
            )
        return base_prompt

    def format_friendly_tool_message(self, tool_name: str, tool_input: Any) -> str:
        msgs = {
            'search': lambda i: f"🔍 正在搜索：{i.get('query') or i.get('input') or '相关信息'}...",
            'web_search': lambda i: f"🌐 正在联网搜索：{i.get('query') or i.get('input') or ''}...",
            'read_file': lambda i: f"📖 正在读取文件：{i.get('file_path') or i.get('path') or ''}...",
            'write_file': lambda i: f"✍️ 正在写入文件：{i.get('file_path') or i.get('path') or ''}...",
            'execute_code': lambda i: "⚙️ 正在执行代码...",
            'calculate': lambda i: f"🧮 正在计算：{i.get('expression') or ''}...",
            'rag_search': lambda i: "📚 正在知识库中查找相关信息...",
            'wait_for_user_input': lambda i: '',
            'create_coding_plan': lambda i: "✍️ 正在分析需求并输出高层实现计划...",
            'get_component_list': lambda i: "🔍 正在获取可用组件列表...",
            'search_component_docs': lambda i: "🔍 正在获取组件文档...",
        }
        fn = msgs.get(tool_name)
        if fn:
            return fn(tool_input or {})
        return "🔧 正在执行操作..."

    async def generate_observation(self, tool_result: Any, tool_name: str, on_stream=None, conversation_id: Optional[str] = None, session_id: Optional[str] = None, iteration: Optional[int] = None) -> None:
        observation_event_id = f"observation_{int(time.time()*1000)}"
        if tool_result.get('success'):
            if tool_name == 'generate_code_project' and (tool_result.get('result') or {}).get('project'):
                project = (tool_result.get('result') or {}).get('project')
                files = project.get('files') if isinstance(project, dict) else []
                files_count = len(files) if isinstance(files, list) else 0
                summary = project.get('summary') if isinstance(project, dict) else ''
                content = f"✅ 代码生成完成\n文件数: {files_count}\n摘要: {summary or '(无)'}\n已完成当前“代码生成”阶段，准备推进后续步骤（若有）。"
            else:
                preview = self.format_result_preview(tool_result.get('result'))
                content = f"✅ 工具执行成功\n结果: {preview}"
        else:
            content = f"❌ 工具执行失败\n错误: {tool_result.get('error')}"
        self.emit('normal', { 'content': content }, session_id or 'default', conversation_id or 'default', observation_event_id, on_stream)

    def format_result_preview(self, result: Any) -> str:
        if result is None:
            return '(空)'
        result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return (result_str[:100] + '...') if len(result_str) > 100 else result_str

    async def generate_final_answer(self, context: AgentContext, on_stream=None, conversation_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        language_instructions = create_language_prompt(self.config.language)
        system_prompt = create_system_prompt(language_instructions)
        history = self.build_conversation_history(context)
        messages = [{ 'role': 'system', 'content': system_prompt }] + history + [{ 'role': 'user', 'content': f"Based on the above reasoning and observations, please provide a final answer to: {context.input}\n\nPlease be concise and direct in your response." }]
        if self.config.streamOutput and on_stream:
            stream = self.llm.stream(messages)
            full = ''
            stream_event_id = f"final_answer_{conversation_id or int(time.time()*1000)}"
            async for chunk in stream:
                c = chunk.get('content') or ''
                if c:
                    full += c
                    self.emit('normal', { 'content': c, 'stream': True }, session_id or 'default', conversation_id or 'default', stream_event_id, on_stream)
            self.emit('normal', { 'content': '', 'stream': True, 'done': True }, session_id or 'default', conversation_id or 'default', stream_event_id, on_stream)
            return full
        else:
            response = await self.llm.invoke(messages)
            content = response.get('content') or ''
            self.emit('normal', { 'content': content }, session_id or 'default', conversation_id or 'default', f"final_full_{int(time.time()*1000)}", on_stream)
            return content

    def build_conversation_history(self, context: AgentContext) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{ 'role': 'user', 'content': f"User Question: {context.input}" }]
        plan_summary = '\n'.join([f"{i+1}. {p.title} [{p.status}]" for i, p in enumerate(self.plan_list)])
        if plan_summary:
            messages.append({ 'role': 'assistant', 'content': f"Plan Status:\n{plan_summary}" })
        recent_steps = context.steps[-6:]
        for step in recent_steps:
            if step.type == 'thought':
                messages.append({ 'role': 'assistant', 'content': f"Thought: {step.content}" })
            elif step.type == 'action':
                messages.append({ 'role': 'assistant', 'content': f"Action: {step.toolName or 'unknown'}\nInput: {json.dumps(step.toolInput, ensure_ascii=False)}" })
            elif step.type == 'observation':
                messages.append({ 'role': 'user', 'content': f"Observation: {self.truncate_observation(step.content)}" })
        return messages

    def truncate_observation(self, content: str, max_length: int = 500) -> str:
        return content if len(content) <= max_length else (content[:max_length] + '... (truncated)')

    def emit(self, type: str, payload: Any, session_id: str, conversation_id: str, event_id: str, on_stream=None) -> None:
        if not on_stream:
            return
        if type == 'normal':
            event = { 'id': event_id, 'role': 'assistant', 'type': 'normal_event', **payload }
        elif type == 'task_plan':
            event = { 'id': event_id, 'role': 'assistant', 'type': 'task_plan_event', 'data': payload }
        elif type == 'tool_call':
            event = { 'id': event_id, 'role': 'assistant', 'type': 'tool_call_event', 'data': payload }
        elif type == 'waiting_input':
            event = { 'id': event_id, 'role': 'assistant', 'type': 'waiting_input_event', 'data': payload }
        else:
            event = { 'id': event_id, 'role': 'assistant', 'type': type, 'data': payload }
        stream_event = StreamEvent(sessionId=session_id, conversationId=conversation_id, event=event, timestamp=int(time.time()*1000))
        self.stream_manager.emit_stream_event(stream_event)
        on_stream(stream_event)

    def get_stream_manager(self) -> StreamManager:
        return self.stream_manager

    def get_tool_registry(self) -> ToolRegistry:
        return self.tool_registry

    def getToolRegistry(self) -> ToolRegistry:
        return self.tool_registry

    def update_config(self, new_config: Dict[str, Any]) -> None:
        data = self.config.__dict__.copy()
        data.update(new_config or {})
        self.config = AgentConfig(**data)
