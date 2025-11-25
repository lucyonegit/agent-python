import json
from core.llm import BaseChatModel
from coder_agent.config.prompt import CODING_AGENT_PROMPTS

class BDDDecomposer:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def decompose(self, requirement: str):
        prompt = CODING_AGENT_PROMPTS['BDD_DECOMPOSER_PROMPT'].replace('{requirement}', requirement)
        messages = [
            { 'role': 'system', 'content': CODING_AGENT_PROMPTS['SYSTEM_PERSONA'] },
            { 'role': 'user', 'content': prompt }
        ]
        resp = await self.llm.invoke(messages)
        content = resp.get('content') or ''
        try:
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            s = m.group(1) if m else content
            arr = json.loads(s)
            return arr if isinstance(arr, list) else []
        except Exception:
            return [{ 'id': 'scenario_1', 'title': 'Fallback scenario', 'given': ['User opens the page'], 'when': ['User interacts with the component'], 'then': ['Expected UI updates occur'] }]

