import json
from typing import List, Dict, Any
from core.llm import BaseChatModel
from coder_agent.config.prompt import CODING_AGENT_PROMPTS
from aitypes import AgentConfig
from coder_agent.architect.architect_generator import ArchitectGenerator

class CodeGenerator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._rag_sources: List[Dict[str, Any]] = []
        self._rag_keys = set()

    async def generate(self, config: AgentConfig, bdd_scenarios: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        options = options or {}
        if options.get('onThought'): options['onThought']('Thought: 启动代码生成流程')
        if options.get('onThought'): options['onThought']('Action: 生成基础项目架构')
        options.get('onArchitectLog') and options['onArchitectLog']('开始调用 ArchitectGenerator 生成基础架构')
        arch = ArchitectGenerator(self.llm, config)
        base_arch = await arch.generate(bdd_scenarios, { 'onStream': options.get('onArchitectStream'), 'onLog': options.get('onArchitectLog') })
        base_arch = base_arch.strip() if base_arch else '[]'
        options.get('onArchitectLog') and options['onArchitectLog']('基础架构生成完成，长度: ' + str(len(base_arch)))
        options.get('onArchitecture') and options['onArchitecture'](base_arch)
        if options.get('onThought'): options['onThought']('Thought: 从BDD输入（支持 Feature 分组）中提取潜在组件关键词用于检索')
        kw_start = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_extract_keywords_{kw_start}', 'status': 'start', 'tool_name': 'extract_keywords', 'args': { 'input': 'bdd_scenarios' }, 'startedAt': kw_start })
        kw_bdd = await self._extract_keywords(bdd_scenarios)
        kw_arch = await self._extract_keywords(base_arch)
        keywords = list(dict.fromkeys([*(kw_bdd or []), *(kw_arch or [])]))
        kw_end = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_extract_keywords_{kw_start}', 'status': 'end', 'tool_name': 'extract_keywords', 'args': { 'input': 'bdd_scenarios' }, 'result': { 'keywords': keywords }, 'success': True, 'startedAt': kw_start, 'finishedAt': kw_end, 'durationMs': kw_end - kw_start })
        if options.get('onThought'): options['onThought']('Action: 获取可用内部组件列表')
        list_start = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_list_components_{list_start}', 'status': 'start', 'tool_name': 'list_internal_components', 'args': {}, 'startedAt': list_start })
        available = await self._fetch_available_components()
        list_end = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_list_components_{list_start}', 'status': 'end', 'tool_name': 'list_internal_components', 'args': {}, 'result': { 'available': available[:20] }, 'success': True, 'startedAt': list_start, 'finishedAt': list_end, 'durationMs': list_end - list_start })
        if options.get('onThought'): options['onThought']('Observation: 可用组件列表: ' + json.dumps(available[:8], ensure_ascii=False))
        sel_start = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_select_components_{sel_start}', 'status': 'start', 'tool_name': 'select_components', 'args': { 'keywords': keywords, 'available': available }, 'startedAt': sel_start })
        selected = self._select_components_from_bdd(keywords, available)
        sel_end = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_select_components_{sel_start}', 'status': 'end', 'tool_name': 'select_components', 'args': { 'keywords': keywords, 'available': available }, 'result': { 'selected': selected }, 'success': True, 'startedAt': sel_start, 'finishedAt': sel_end, 'durationMs': sel_end - sel_start })
        if options.get('onThought'): options['onThought']('Action: fetch_component_docs\nInput: { "components": ' + json.dumps(selected, ensure_ascii=False) + ' }')
        rag_context = await self._fetch_component_docs(selected, options)
        if options.get('onRagSources'): options['onRagSources'](self.get_rag_sources())
        if options.get('onThought'): options['onThought']('Observation: 已获取组件API与示例文档，开始代码生成')
        prompt = CODING_AGENT_PROMPTS['CODE_GENERATOR_PROMPT'].replace('{bdd_scenarios}', bdd_scenarios).replace('{base_architecture}', base_arch).replace('{rag_context}', rag_context)
        messages = [ { 'role': 'system', 'content': CODING_AGENT_PROMPTS['SYSTEM_PERSONA'] }, { 'role': 'user', 'content': prompt } ]
        gen_start = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_llm_generate_{gen_start}', 'status': 'start', 'tool_name': 'llm_generate_project', 'args': { 'model': 'chat', 'inputs': ['persona', 'prompt'] }, 'startedAt': gen_start })
        resp = await self.llm.invoke(messages)
        gen_end = self._now()
        if options.get('onToolCall'): options['onToolCall']({ 'id': f'tool_llm_generate_{gen_start}', 'status': 'end', 'tool_name': 'llm_generate_project', 'args': { 'model': 'chat' }, 'result': { 'length': len(resp.get('content') or '') }, 'success': True, 'startedAt': gen_start, 'finishedAt': gen_end, 'durationMs': gen_end - gen_start })
        content = resp.get('content') or ''
        try:
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            s = m.group(1) if m else content
            project = json.loads(s)
            try:
                flattened = self._flatten_features_to_scenarios(bdd_scenarios)
                matches = await self._compute_scenario_matches(flattened, [f.get('path') for f in project.get('files', [])])
                if options.get('onScenarioMatches') and matches:
                    options['onScenarioMatches'](matches)
            except Exception:
                pass
            return project
        except Exception:
            return { 'files': [ { 'path': 'src/components/GeneratedComponent.tsx', 'content': content } ], 'summary': 'Failed to parse structured output, returning raw content.' }

    def _now(self) -> int:
        import time
        return int(time.time()*1000)

    async def _extract_keywords(self, text: str) -> List[str]:
        prompt = f"Identify the UI components mentioned or implied in the following text. Return a comma-separated list of component names (e.g., \"Button, Table, DatePicker\").\n\nText:\n{text}"
        resp = await self.llm.invoke([ { 'role': 'user', 'content': prompt } ])
        content = resp.get('content') or ''
        return [s.strip() for s in content.split(',') if s.strip()]

    def _flatten_features_to_scenarios(self, input_str: str) -> str:
        try:
            data = json.loads(input_str)
            if isinstance(data, list) and data and isinstance(data[0], dict) and ('scenarios' in data[0]):
                scenarios = []
                for f in data:
                    if isinstance(f, dict) and isinstance(f.get('scenarios'), list):
                        scenarios.extend(f.get('scenarios'))
                return json.dumps(scenarios, ensure_ascii=False)
            if isinstance(data, list):
                return json.dumps(data, ensure_ascii=False)
        except Exception:
            pass
        return input_str

    async def _fetch_available_components(self) -> List[str]:
        import os
        import httpx
        base = os.environ.get('RAG_BASE_URL', 'http://192.168.21.101:3000')
        url = f"{base}/getComponentList"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers={ 'Content-Type': 'application/json' })
                if resp.status_code != 200:
                    return []
                data = resp.json()
                text = data.get('answer') or ''
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
                return [s.strip() for s in str(text).split('\n') if s.strip()]
        except Exception:
            return []

    def _select_components_from_bdd(self, keywords: List[str], available: List[str]) -> List[str]:
        s = set(a.lower() for a in available)
        selected = []
        for k in keywords:
            if k.lower() in s:
                selected.append(k)
        if not selected:
            return available[:3]
        return list(dict.fromkeys(selected))

    async def _fetch_component_docs(self, components: List[str], options: Dict[str, Any]) -> str:
        import os
        import httpx
        base = os.environ.get('RAG_BASE_URL', 'http://192.168.21.101:3000')
        url = f"{base}/query"
        context = ''
        for comp in components:
            for sec in ['API / Props','Usage Example']:
                started_at = self._now()
                tool_id = f"tool_rag_{comp}_{sec}_{started_at}"
                if options.get('onToolCall'):
                    options['onToolCall']({ 'id': tool_id, 'status': 'start', 'tool_name': 'search_component_docs', 'args': { 'query': '总结下这个组件的使用文档', 'metadataFilters': { 'component_name': comp, 'section': sec }, 'limit': 3 }, 'startedAt': started_at })
                try:
                    body = { 'query': '总结下这个组件的使用文档', 'metadataFilters': { 'component_name': comp, 'section': sec }, 'limit': 3 }
                    async with httpx.AsyncClient(timeout=15) as client:
                        resp = await client.post(url, headers={ 'Content-Type': 'application/json' }, json=body)
                    result = resp.json() if resp.status_code == 200 else { 'answer': '', 'sources': [] }
                    raw = result.get('answer') or ''
                    payload_str = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                    safe_payload = payload_str.replace('```','\`\`\`')
                    code_fence = 'tsx' if sec == 'Usage Example' else 'md'
                    context += f"\n--- {comp} ({sec}) ---\n\n```{code_fence}\n{safe_payload}\n```\n\n"
                    if options.get('onRagDoc'):
                        options['onRagDoc']({ 'component': comp, 'section': sec, 'content': payload_str })
                    src_list = result.get('sources') or []
                    if isinstance(src_list, list):
                        for s in src_list:
                            key = f"{str(s.get('metadata',{}).get('component_name','') or s.get('metadata',{}).get('title',''))}::{str(s.get('metadata',{}).get('section',''))}::{s.get('content','')}"
                            if key not in self._rag_keys:
                                self._rag_keys.add(key)
                                self._rag_sources.append(s)
                        if options.get('onRagUsed'):
                            options['onRagUsed']({ 'term': comp, 'components': [comp] })
                    finished = self._now()
                    if options.get('onToolCall'):
                        options['onToolCall']({ 'id': tool_id, 'status': 'end', 'tool_name': 'search_component_docs', 'args': { 'query': comp, 'metadataFilters': { 'component_name': comp, 'section': sec }, 'limit': 3 }, 'result': result, 'success': True, 'startedAt': started_at, 'finishedAt': finished, 'durationMs': finished - started_at })
                except Exception as err:
                    finished = self._now()
                    if options.get('onToolCall'):
                        options['onToolCall']({ 'id': tool_id, 'status': 'end', 'tool_name': 'search_component_docs', 'args': { 'query': comp, 'metadataFilters': { 'component_name': comp, 'section': sec }, 'limit': 3 }, 'result': { 'error': str(err) }, 'success': False, 'startedAt': started_at, 'finishedAt': finished, 'durationMs': finished - started_at })
        return context or 'No internal component documentation found.'

    def get_rag_sources(self) -> List[Dict[str, Any]]:
        return self._rag_sources

    async def _compute_scenario_matches(self, bdd_scenarios: str, file_paths: List[str]) -> List[Dict[str, Any]]:
        prompt = "Given BDD scenarios and a list of project file paths, select up to 3 most relevant file paths for each scenario and return JSON array [{\"scenarioId\":\"...\",\"paths\":[\"...\"]}].\nScenarios JSON:\n" + bdd_scenarios + "\n\nFile paths:\n" + "\n".join(file_paths)
        resp = await self.llm.invoke([ { 'role': 'user', 'content': prompt } ])
        content = resp.get('content') or ''
        try:
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            s = m.group(1) if m else content
            arr = json.loads(s)
            if isinstance(arr, list):
                return [ { 'scenarioId': str(x.get('scenarioId') or x.get('id') or ''), 'paths': [str(p) for p in (x.get('paths') or [])] } for x in arr ]
        except Exception:
            pass
        return []
