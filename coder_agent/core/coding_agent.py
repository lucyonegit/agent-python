import json
from typing import Any, Dict
from core.llm import BaseChatModel, LangChainLLM
from aitypes import AgentConfig, TaskStep, TaskStatus
from core.stream_manager import StreamEvent
from coder_agent.planner.coding_planner import CodingPlanner
from coder_agent.bdd.bdd_decomposer import BDDDecomposer
from coder_agent.generator.code_generator import CodeGenerator
from core.react_agent import ReActAgent

class CodingAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = AgentConfig(**config)
        try:
            self.llm: BaseChatModel = LangChainLLM(model=self.config.model, temperature=self.config.temperature, max_tokens=self.config.maxTokens, streaming=self.config.streamOutput)
        except Exception:
            from core.react_agent import SimpleLLM
            self.llm = SimpleLLM()
        self.planner = CodingPlanner(self.llm)
        self.bdd = BDDDecomposer(self.llm)
        self.generator = CodeGenerator(self.llm)

    def gen_id(self, prefix: str) -> str:
        import time, random
        return f"{prefix}_{int(time.time()*1000)}_{format(random.randint(0, 36**6-1), 'x')}"

    async def run(self, input_text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        options = options or {}
        on_stream = options.get('onStream')
        react = ReActAgent({ 'model': self.config.model, 'temperature': self.config.temperature, 'streamOutput': True, 'language': self.config.language, 'maxTokens': self.config.maxTokens, 'maxIterations': self.config.maxIterations, 'pauseAfterEachStep': False, 'autoPlanOnStart': False })
        final_project = None
        async def create_plan_tool_exec(tool_input):
            plan = await self.planner.create_plan(tool_input.get('input') or input_text)
            steps = [TaskStep(id=s['id'], title=s['title'], status='pending', note=s.get('description')) for s in plan['steps']]
            if on_stream:
                on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('task_plan'), 'role': 'assistant', 'type': 'task_plan_event', 'data': { 'step': [p.__dict__ for p in steps] } }, timestamp=self._now()))
            return { 'plan': plan }
        async def bdd_tool_exec(tool_input):
            features = await self.bdd.decompose(tool_input.get('requirement') or input_text)
            if on_stream:
                on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('bdd_event'), 'role': 'assistant', 'type': 'bdd_event', 'data': { 'features': features } }, timestamp=self._now()))
            return { 'features': features }
        async def generate_tool_exec(tool_input):
            nonlocal final_project
            project = await self.generator.generate(self.config, tool_input.get('bdd'), {
                'onThought': lambda content: on_stream and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('react_piece'), 'role': 'assistant', 'type': 'normal_event', 'content': content }, timestamp=self._now())),
                'onToolCall': lambda payload: on_stream and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': payload.get('id') or self.gen_id('tool_call'), 'role': 'assistant', 'type': 'tool_call_event', 'data': payload }, timestamp=self._now())),
                'onRagUsed': lambda data: on_stream and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('rag_used'), 'role': 'assistant', 'type': 'rag_used_event', 'data': data }, timestamp=self._now())),
                'onRagSources': lambda sources: on_stream and sources and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('rag_event'), 'role': 'assistant', 'type': 'rag_event', 'data': { 'sources': sources } }, timestamp=self._now())),
                'onRagDoc': lambda payload: on_stream and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('rag_doc'), 'role': 'assistant', 'type': 'rag_doc_event', 'data': payload }, timestamp=self._now())),
                'onScenarioMatches': lambda matches: on_stream and matches and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('scenario_match'), 'role': 'assistant', 'type': 'scenario_match_event', 'data': { 'matches': matches } }, timestamp=self._now())),
                'onArchitectLog': lambda message: on_stream and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('architect_log'), 'role': 'assistant', 'type': 'architect_event', 'data': { 'message': message } }, timestamp=self._now())),
                'onArchitecture': lambda architecture: on_stream and on_stream(StreamEvent(sessionId=options.get('sessionId') or 'default', conversationId=options.get('conversationId') or 'default', event={ 'id': self.gen_id('architecture'), 'role': 'assistant', 'type': 'architecture_event', 'data': { 'architecture': architecture } }, timestamp=self._now())),
                'onArchitectStream': lambda evt: on_stream and on_stream(evt),
            })
            final_project = project
            return { 'project': project, 'planUpdate': { 'completeIds': ['step_3'], 'completeTitles': ['代码生成','code gen'] } }
        react.getToolRegistry().registerTools([
            { 'name': 'create_coding_plan', 'description': '高优先级：在开始任何实现之前，首先调用此工具以创建简洁的步骤计划（3-5步）。当用户需求是页面/组件/交互开发时，务必先执行本工具。关键词: planner, 计划, planning。', 'parameters': [ { 'name': 'input', 'type': 'string', 'description': '用户需求', 'required': True } ], 'execute': create_plan_tool_exec },
            { 'name': 'decompose_bdd', 'description': '将需求拆解为BDD场景（JSON数组）', 'parameters': [ { 'name': 'requirement', 'type': 'string', 'description': '需求文本', 'required': True } ], 'execute': bdd_tool_exec },
            { 'name': 'generate_code_project', 'description': '根据BDD与内部组件生成完整前端项目结构', 'parameters': [ { 'name': 'bdd', 'type': 'string', 'description': 'BDD场景JSON字符串', 'required': True } ], 'execute': generate_tool_exec }
        ])
        def forward_on_stream(evt: StreamEvent):
            e = evt.event
            if e and e.get('type') == 'tool_call_event' and e.get('data', {}).get('status') == 'end':
                tool_name = e.get('data', {}).get('tool_name')
                result = e.get('data', {}).get('result')
                if tool_name == 'create_coding_plan' and result and result.get('plan'):
                    steps = result['plan']['steps']
                    mapped = [TaskStep(id=s['id'], title=s['title'], status='pending', note=s.get('description')) for s in steps]
                    on_stream and on_stream(StreamEvent(sessionId=evt.sessionId, conversationId=evt.conversationId, event={ 'id': self.gen_id('task_plan'), 'role': 'assistant', 'type': 'task_plan_event', 'data': { 'step': [p.__dict__ for p in mapped] } }, timestamp=self._now()))
                if tool_name == 'decompose_bdd' and result and (result.get('features') or result.get('scenarios')):
                    feats = result.get('features') or [ { 'feature_id': 'feature_1', 'feature_title': 'General', 'description': '', 'scenarios': result.get('scenarios') or [] } ]
                    on_stream and on_stream(StreamEvent(sessionId=evt.sessionId, conversationId=evt.conversationId, event={ 'id': self.gen_id('bdd_event'), 'role': 'assistant', 'type': 'bdd_event', 'data': { 'features': feats } }, timestamp=self._now()))
                if tool_name == 'generate_code_project' and result and result.get('project'):
                    rag_sources = self.generator.get_rag_sources()
                    if rag_sources:
                        on_stream and on_stream(StreamEvent(sessionId=evt.sessionId, conversationId=evt.conversationId, event={ 'id': self.gen_id('rag_event'), 'role': 'assistant', 'type': 'rag_event', 'data': { 'sources': rag_sources } }, timestamp=self._now()))
            on_stream and on_stream(evt)
        await react.run_with_session(input_text, { 'sessionId': options.get('sessionId'), 'conversationId': options.get('conversationId'), 'onStream': forward_on_stream })
        if not final_project:
            final_project = { 'files': [], 'summary': 'No project generated' }
        return { 'finalAnswer': final_project }

    def _now(self) -> int:
        import time
        return int(time.time()*1000)
