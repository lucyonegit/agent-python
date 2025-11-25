import json
from core.llm import BaseChatModel
from coder_agent.config.prompt import CODING_AGENT_PROMPTS

class CodingPlanner:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def create_plan(self, input_text: str) -> dict:
        prompt = CODING_AGENT_PROMPTS['PLANNER_PROMPT'].replace('{input}', input_text)
        messages = [
            { 'role': 'system', 'content': CODING_AGENT_PROMPTS['SYSTEM_PERSONA'] },
            { 'role': 'user', 'content': prompt }
        ]
        resp = await self.llm.invoke(messages)
        content = resp.get('content') or ''
        try:
            import re
            m = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            json_str = m.group(1) if m else content
            return json.loads(json_str)
        except Exception:
            return { 'summary': 'Plan generation failed to parse, proceeding with default plan.', 'steps': [{ 'id': 'step_1', 'title': 'Implement Feature', 'description': input_text }] }

