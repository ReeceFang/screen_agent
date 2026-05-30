from __future__ import annotations
from copy import deepcopy
import json
from typing import Any

from agent import (
    agent,
    state as uia_tools_state,
    planer_agent,
    load_manual,
    PlanActions,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState
from langgraph.types import interrupt


""" state """


class AgentState(MessagesState):
    user_request: str
    actions: list[dict]
    current_step: int
    last_tool: dict[str, Any] | None


"""
actions:
[
	{
	    "id": 1,
	    "type": "analyze",
	    "instruction": "分析用户需求，提取目标、约束和成功标准",
	}.
	...
]

last_tool:
{
    "name": observe_window,
    "args": "",
}.
"""


DEFAULT_AGENT_STATE: AgentState = {
    "messages": [],
    "user_request": "",
    "actions": [],
    "current_step": 0,
    "last_tool": None,
}


def create_initial_state(patch: dict[str, Any] | None = None) -> AgentState:
    """基于默认 state 创建初始 state，并用局部 patch 覆盖。"""
    state = deepcopy(DEFAULT_AGENT_STATE)
    if patch:
        unknown_keys = set(patch) - set(DEFAULT_AGENT_STATE)
        if unknown_keys:
            raise ValueError(f"未知 state 字段: {sorted(unknown_keys)}")

        state.update(patch)
    return state


""" worker node"""


def worker_node(state):

    actions = state["actions"]
    current_step = state["current_step"]
    action = actions[current_step]
    message = action["instruction"]
    last_tool = None

    # 执行
    stream = agent.stream(
        {"messages": [HumanMessage(content=message)]},
    )
    for chunk in stream:
        if isinstance(chunk, AIMessage):
            last_tool = extract_last_tool_call(chunk) or last_tool
            print_ai_message(chunk)
            continue

        if not isinstance(chunk, dict):
            continue

        for node_output in chunk.values():
            if not isinstance(node_output, dict):
                continue
            messages = node_output.get("messages")
            if not messages:
                continue
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                last_tool = extract_last_tool_call(last_message) or last_tool
                print_ai_message(last_message)
            elif isinstance(last_message, ToolMessage):
                print_tool_message(last_message, message)

    return {
        "current_step": current_step + 1,
        "last_tool": last_tool,
    }


def extract_last_tool_call(message: AIMessage) -> dict[str, Any] | None:
    """从 AIMessage 中提取最后一次工具调用。"""
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None

    tool_call = tool_calls[-1]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args") or {}
    return {
        "name": tool_name,
        "args": uia_tools_state.display_tool_args(tool_args),
    }


def print_ai_message(message: AIMessage) -> None:
    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        print(f"\n[调用工具] {tool_call['name']} {tool_call.get('args') or {}}")

    if message.content:
        print(message.content, end="", flush=True)


def print_tool_message(message: ToolMessage, user_request: str) -> None:
    tool_name = getattr(message, "name", "") or "tool"
    content = str(message.content)

    if tool_name == "observe_window":
        try:
            data = json.loads(content)
            controls = data.get("controls", [])
            print(f"\n[工具结果] observe_window: 发现 {len(controls)} 个控件")
            matches = [
                control
                for control in controls
                if control.get("name")
                and (control["name"] in user_request or user_request in control["name"])
            ]
            if matches:
                print("[可能匹配]")
                for control in matches[:10]:
                    print(
                        "  "
                        f"{control.get('id')} "
                        f"{control.get('type')} "
                        f"{control.get('name')!r} "
                        f"actionable={control.get('actionable_hint')}"
                    )
            for control in controls[:20]:
                print(
                    "  "
                    f"{control.get('id')} "
                    f"{control.get('type')} "
                    f"{control.get('name')!r} "
                    f"actionable={control.get('actionable_hint')}"
                )
            if len(controls) > 20:
                print(f"  ... 还有 {len(controls) - 20} 个控件未显示")
            return
        except Exception:
            pass

    print(f"\n[工具结果] {tool_name}: {content[:500]}")


""" planer node """

manual_path = r"manual\推理.docx"


def planer_node(state):
    """计划节点。"""
    user_request = state["user_request"]

    manual = load_manual(manual_path)
    content = f"""
                    应用使用说明书：
                    {manual}

                    用户需求：
                    {user_request}
    """.strip()
    result = planer_agent.invoke(
        {"messages": [HumanMessage(content=content)]},
    )
    structured_response = result.get("structured_response")

    if structured_response is None:
        return {"actions": []}

    actions = None

    if isinstance(structured_response, PlanActions):
        actions = [action.model_dump() for action in structured_response.actions]

    if isinstance(structured_response, dict):
        actions = structured_response.get("actions")

    if not isinstance(actions, list):
        actions = []

    print(f"[计划结果]\n{actions}")

    return {"actions": actions}


""" route """


def route_after_plan(state) -> str:
    """计划后路由：没有动作则结束，有动作则进入 worker。"""
    actions = state.get("actions") or []
    if not actions:
        return END

    return "worker_node"


def route_after_human(state) -> str:
    """人工确认后路由：动作全部完成则结束，否则继续 worker。"""
    actions = state.get("actions") or []
    current_step = state.get("current_step", 0)

    if current_step >= len(actions):
        return END

    return "worker_node"


""" human node """


def human_node(state):
    """对需要人工确认的工具调用暂停图执行。"""
    last_tool = state.get("last_tool")
    if not should_confirm_last_tool(last_tool):
        return {}

    interrupt({"message": "请在选完成后按 y 确认"})

    print(f"[人工确认完成]")
    return {}


def should_confirm_last_tool(last_tool: dict[str, Any] | None) -> bool:
    """判断最后一次工具调用是否需要人工确认。"""
    if not last_tool:
        return False

    if last_tool.get("name") != "click_control":
        return False

    args = last_tool.get("args") or {}
    control_name = str(args.get("name") or "")
    return "打开" in control_name
