from typing import Any, Callable, Dict, List

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: Dict[str, Any]) -> None:
        name = tool.get('name')
        if not name:
            raise ValueError('Tool must have a name')
        if name in self.tools:
            raise ValueError(f'Tool "{name}" already exists')
        self.tools[name] = tool

    def register_tools(self, tools: List[Dict[str, Any]]) -> None:
        for t in tools:
            self.register_tool(t)

    def registerTools(self, tools: List[Dict[str, Any]]) -> None:
        self.register_tools(tools)

    def get_tool(self, name: str) -> Dict[str, Any]:
        return self.tools.get(name)

    def get_all_tools(self) -> Dict[str, Any]:
        return dict(self.tools)

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    async def execute_tool(self, name: str, input: Any) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if not tool:
            return {'success': False, 'result': None, 'error': f'Tool "{name}" not found'}
        params = tool.get('parameters', [])
        required = [p for p in params if p.get('required')]
        for p in required:
            if p['name'] not in input:
                return {'success': False, 'result': None, 'error': f'Required parameter "{p["name"]}" is missing'}
        execute: Callable[[Any], Any] = tool.get('execute')
        try:
            import asyncio
            if callable(execute):
                if asyncio.iscoroutinefunction(execute):
                    result = await execute(input)
                else:
                    result = execute(input)
            else:
                result = None
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'result': None, 'error': str(e)}

    def get_tools_description(self) -> str:
        descriptions = []
        for tool in self.tools.values():
            params = tool.get('parameters', [])
            params_str = '\n'.join([
                f"  - {p.get('name')}: {p.get('type')}{' (required)' if p.get('required') else ' (optional)'} - {p.get('description')}"
                for p in params
            ])
            descriptions.append(f"{tool.get('name')}: {tool.get('description')}\nParameters:\n{params_str}")
        return '\n\n'.join(descriptions)

    def unregister_tool(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    def clear(self) -> None:
        self.tools.clear()
