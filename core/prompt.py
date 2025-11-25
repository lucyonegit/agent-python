from typing import Optional

def create_planner_prompt(input: str) -> str:
    return (
        "你是规划器。请为下面的需求创建一个精炼的执行计划（2-5 步）。\n"
        "只返回一个紧凑的 JSON 数组，如：\n"
        "[\n  {\"title\":\"步骤 1 ...\"},\n  {\"title\":\"步骤 2 ...\"}\n]\n"
        "不要包含任何额外文本。\n"
        "用户目标如下：\n---\n"
        f"{input}\n---\n"
    )

def create_language_prompt(language: str) -> str:
    if language == 'chinese':
        return (
            "语言要求：\n  - 所有内容均使用中文\n  - 区块标签必须使用英文并保持一致：Thought / Action / Input / Final Answer"
        )
    if language == 'english':
        return (
            "语言要求：\n  - 所有内容均使用中文\n  - 区块标签必须使用英文并保持一致：Thought / Action / Input / Final Answer"
        )
    return (
        "语言要求：\n  - 所有内容均使用中文\n  - 区块标签必须使用英文并保持一致：Thought / Action / Input / Final Answer"
    )

def _gen_system_prompt(language: str, tools_description: Optional[str] = None) -> str:
    base = (
        "你是一个基于 ReAct（推理 + 执行动作）架构的智能体。严格遵守以下输出格式：\n\n"
        "格式：\n"
        "Thought: [简短推理，最多 1-2 句]\n"
        "Action: [tool_name] 或 Final Answer: [answer]\n"
        "Input: [JSON 对象，仅当执行 Action 时填写]\n\n"
        "规则：\n"
        "1. Thought 必须简洁\n"
        "2. 只能二选一：Action 或 Final Answer\n"
        "3. 需要信息时使用工具；完成当前计划步骤后再继续下一步\n"
        "4. 在所有计划步骤完成后再输出 Final Answer\n"
        "5. 严格按照上述格式输出，不要添加多余文本\n\n"
        f"{language}\n\n"
        f"{tools_description if tools_description else ''}\n\n"
        "注意：区块标签必须使用以下英文单词并保持一致：\"Thought\", \"Action\", \"Input\", \"Final Answer\"。"
    )
    return base

def create_system_prompt(language_prompt: str, tools_description: Optional[str] = None) -> str:
    base = _gen_system_prompt(language_prompt)
    if tools_description and tools_description.strip():
        return (
            f"{base}\n\nAvailable tools:\n{tools_description}\nUse tools when needed. If using a tool, output Action and Input."
        )
    return base

def create_pre_action_prompt(input: str) -> str:
    return f"请针对以下用户请求生成一段自然的确认语，说明你将开始执行任务：{input}\n要求：简短、自然、礼貌。"

def create_planner_prompt_with_tool(input: str) -> str:
    return (
        "你是规划器。请为下面的需求创建一个精炼的执行计划（2-5 步），在必要时可以使用工具获取信息。\n"
        "用户目标如下：\n---\n"
        f"{input}\n---\n"
    )
