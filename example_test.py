import asyncio
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent-python'))
from core.react_agent import ReActAgent

async def main():
    agent = ReActAgent()
    def handler(e):
        pass
    result = await agent.run_with_session('测试ReAct Python版本', { 'onStream': handler })
    print(result['finalAnswer'])

if __name__ == '__main__':
    asyncio.run(main())
