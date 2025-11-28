import json
from typing import Any, Dict, Optional
from core.llm import BaseChatModel
from aitypes import AgentConfig
from core.react_agent import ReActAgent
from coder_agent.config.prompt import CODING_AGENT_PROMPTS

class ArchitectGenerator:
    def __init__(self, llm: BaseChatModel, config: AgentConfig):
        self.llm = llm
        self.config = config

    async def generate(self, bdd: str, options: Optional[Dict[str, Any]] = None) -> str:
        options = options or {}
        async def create_architecture(tool_input: str) -> str:
            sys_prompt = CODING_AGENT_PROMPTS['ARCHITECT_GENERATOR_PROMPT']
            user_prompt = f"\n**User Prompt (用户输入)**\n任务：项目架构设计\n请分析以下 BDD 规范，并输出项目架构 JSON 结构。\n**BDD 规范：**\n{tool_input}\n"
            messages = [ { 'role': 'system', 'content': sys_prompt }, { 'role': 'user', 'content': user_prompt } ]
            resp = await self.llm.invoke(messages)
            raw = resp.get('content') or ''
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            s = m.group(1) if m else raw
            t = (s or '').strip()
            try:
                parsed = json.loads(t)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return t if t else '[]'

        agent = ReActAgent({ 'model': self.config.model, 'temperature': self.config.temperature, 'streamOutput': True, 'language': self.config.language, 'maxTokens': self.config.maxTokens, 'maxIterations': self.config.maxIterations, 'pauseAfterEachStep': False, 'autoPlanOnStart': False })
        async def is_valid_json_exec(inp: Dict[str, Any]) -> Dict[str, Any]:
            try:
                json.loads(inp.get('input') or '')
                return { 'content': 'true' }
            except Exception:
                return { 'content': 'false' }
        async def create_project_architecture_exec(inp: Dict[str, Any]) -> Dict[str, Any]:
            content = await create_architecture(inp.get('input') or bdd)
            return content
        agent.getToolRegistry().registerTools([
            { 'name': 'is_valid_json', 'description': '检查输入是否为有效 JSON 格式', 'parameters': [ { 'name': 'input', 'type': 'string', 'description': '要检查的 JSON 字符串', 'required': True } ], 'execute': is_valid_json_exec },
            { 'name': 'create_project_architecture', 'description': '创建项目的代码架构设计', 'parameters': [ { 'name': 'input', 'type': 'string', 'description': '用户需求BDD内容', 'required': True } ], 'execute': create_project_architecture_exec },
        ])
        prompt = (
            f"\n** 任务描述 **\n"
            f"根据BDD需求，创建前端项目架构，并在生成后验证该架构 JSON 的有效性。\n"
            f"请严格按以下步骤执行：\n"
            f"1) 使用工具 create_project_architecture 先生成项目架构 JSON 字符串；\n"
            f"2) 立刻使用工具 is_valid_json 验证第1步生成的架构 JSON（参数 input 传入第1步的完整 JSON 字符串）；\n"
            f"3) 若校验为 false，请重新生成架构并再次验证，直到返回 true；\n"
            f"4) 最终仅输出有效的项目架构 JSON（不包含任何描述或 Markdown）。\n\n"
            f"** BDD需求 **\n{bdd}\n\n"
        )
        options.get('onLog') and options['onLog']('ArchitectAgent: 开始生成项目架构')
        result = await agent.run_with_session(prompt, { 'onStream': options.get('onStream') })
        options.get('onLog') and options['onLog']('ArchitectAgent: 完成生成')
        fa = (result.get('finalAnswer') or '').strip()
        if fa:
            try:
                parsed = json.loads(fa)
                options.get('onLog') and options['onLog']('ArchitectAgent: 架构JSON有效')
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                options.get('onLog') and options['onLog']('ArchitectAgent: 架构JSON无效，尝试兜底')
                alt = await create_architecture(bdd)
                try:
                    parsed = json.loads(alt)
                    options.get('onLog') and options['onLog']('ArchitectAgent: 兜底架构JSON有效')
                    return json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    options.get('onLog') and options['onLog']('ArchitectAgent: 兜底架构仍非有效JSON，原样返回')
                    return alt
        options.get('onLog') and options['onLog']('ArchitectAgent: LLM最终答案为空，切换直接聊天兜底')
        alt = await create_architecture(bdd)
        try:
            parsed = json.loads(alt)
            options.get('onLog') and options['onLog']('ArchitectAgent: 兜底架构JSON有效')
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            options.get('onLog') and options['onLog']('ArchitectAgent: 兜底架构仍非有效JSON，原样返回')
            return alt
