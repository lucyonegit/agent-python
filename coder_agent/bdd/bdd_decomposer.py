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
            if isinstance(arr, list):
                looks_like_feature = (len(arr) == 0) or (isinstance(arr[0], dict) and ('scenarios' in arr[0] or 'feature_id' in arr[0] or 'feature_title' in arr[0]))
                if looks_like_feature:
                    return arr
                scenarios = arr
                return [ { 'feature_id': 'feature_1', 'feature_title': 'General', 'description': '', 'scenarios': scenarios } ]
            return []
        except Exception:
            return [ { 'feature_id': 'feature_1', 'feature_title': 'General', 'description': '', 'scenarios': [ { 'id': 'scenario_1', 'title': 'Fallback scenario', 'given': ['User opens the page'], 'when': ['User enters valid input'], 'then': ['Expected UI updates occur'] } ] } ]
