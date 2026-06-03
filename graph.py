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
    # 本次修改：用结构化字段保存 planner / worker / human 的执行记忆。
    plan_versions: list[dict]
    execution_log: list[dict]
    human_events: list[dict]
    last_worker_result: dict[str, Any] | None


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

plan_versions:
[
    {
        "version": 1,
        "reason": "initial_plan",
        "actions": [
            {
                "id": 1,
                "type": "click",
                "instruction": "点击推理界面",
            }
        ],
    }
]

execution_log:
[
    {
        "action_id": 1,
        "action_type": "click",
        "instruction": "点击推理界面",
        "status": "success",
        "tool_calls": [
            {"name": "observe_window", "args": {}},
            {"name": "click_control", "args": {"name": "推理界面"}},
        ],
        "tool_results": [
            {"name": "observe_window", "summary": "发现 86 个控件"},
            {"name": "click_control", "summary": "成功点击 TabItem「推理界面」", "ok": True},
        ],
        "summary": "成功点击 TabItem「推理界面」",
        "error": None,
    }
]

human_events:
[
    {
        "action_id": 2,
        "instruction": "点击打开图片",
        "reason": "点击打开类控件后需要人工确认",
        "response": "y",
        "status": "confirmed",  # confirmed 或 rejected
    }
]

last_worker_result:
{
    "action_id": 1,
    "action_type": "click",
    "instruction": "点击推理界面",
    "status": "success",
    "tool_calls": [],
    "tool_results": [],
    "summary": "成功点击 TabItem「推理界面」",
    "error": None,
}
"""


DEFAULT_AGENT_STATE: AgentState = {
    "messages": [],
    "user_request": "",
    "actions": [],
    "current_step": 0,
    "last_tool": None,
    # 本次修改：这些默认值会被后续节点持续追加执行记忆。
    "plan_versions": [],
    "execution_log": [],
    "human_events": [],
    "last_worker_result": None,
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
    # 本次修改：worker 仍然无记忆执行，但 Graph 会记录本轮工具调用和结果摘要。
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    seen_tool_call_ids: set[str] = set()
    final_worker_text = ""

    stream = agent.stream(
        {"messages": [HumanMessage(content=message)]},
    )
    for chunk in stream:
        # 处理一种更简单的返回形状
        if isinstance(chunk, AIMessage):
            # 获取最后一次的工具调用
            last_tool = extract_last_tool_call(chunk) or last_tool
            # 获取本条消息中的新工具调用摘要，避免重复记录同一次工具调用。
            tool_calls.extend(extract_tool_call_summaries(chunk, seen_tool_call_ids))
            # 获取 worker 最近一次的自然语言回复
            final_worker_text = latest_message_text(chunk, final_worker_text)
            print_ai_message(chunk)
            continue

        if not isinstance(chunk, dict):
            continue

        """
        chunk 可能的形状示例：
        chunk = {
            "model": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "observe_window",
                                "args": {},
                                "id": "call_1",
                            }
                        ],
                    )
                ]
            }
        }

        或

        chunk = {
            "tools": {
                "messages": [
                    ToolMessage(
                        name="click_control",
                        content='{"ok": true, "control": {"type": "TabItem", "name": "推理界面"}}',
                        tool_call_id="call_2",
                    )
                ]
            }
        }
        """
        for node_output in chunk.values():
            if not isinstance(node_output, dict):
                continue
            messages = node_output.get("messages")
            if not messages:
                continue
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                last_tool = extract_last_tool_call(last_message) or last_tool
                tool_calls.extend(
                    extract_tool_call_summaries(last_message, seen_tool_call_ids)
                )
                final_worker_text = latest_message_text(
                    last_message,
                    final_worker_text,
                )
                print_ai_message(last_message)
            elif isinstance(last_message, ToolMessage):
                tool_results.append(summarize_tool_message(last_message))
                print_tool_message(last_message, message)

    # 将 worker 的本轮执行压缩成结构化结果和 planner 可读摘要
    worker_result = build_worker_result(
        action=action,
        tool_calls=tool_calls,
        tool_results=tool_results,
        final_worker_text=final_worker_text,
    )
    execution_log = list(state.get("execution_log") or [])
    execution_log.append(worker_result)

    return {
        "current_step": current_step + 1,
        "last_tool": last_tool,
        "last_worker_result": worker_result,
        "execution_log": execution_log,
        "messages": [AIMessage(content=format_worker_memory_message(worker_result))],
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


# 以下辅助函数用于生成工具的简短摘要，避免把完整工具 JSON 写入 planner 记忆。
def extract_tool_call_summaries(
    message: AIMessage,
    seen_tool_call_ids: set[str],
) -> list[dict[str, Any]]:
    """提取本条 AIMessage 中的新工具调用摘要。"""
    summaries: list[dict[str, Any]] = []
    tool_calls = getattr(message, "tool_calls", None) or []

    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id")
        if tool_call_id:
            if tool_call_id in seen_tool_call_ids:
                continue
            seen_tool_call_ids.add(tool_call_id)

        tool_args = tool_call.get("args") or {}
        summaries.append(
            {
                "name": tool_call.get("name"),
                "args": uia_tools_state.display_tool_args(tool_args),
            }
        )

    return summaries


def latest_message_text(message: AIMessage, fallback: str) -> str:
    """记录 worker 最近一次自然语言回复。"""
    text = message_content_to_text(message.content)
    return text or fallback


def message_content_to_text(content: Any) -> str:
    """把 LangChain message content 处理成干净的字符串。"""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if content:
        return str(content).strip()

    return ""


def summarize_tool_message(message: ToolMessage) -> dict[str, Any]:
    """把 ToolMessage 压缩成 planner 可读的短结果。"""
    tool_name = getattr(message, "name", "") or "tool"
    content = str(message.content)

    try:
        data = json.loads(content)
    except Exception:
        return {
            "name": tool_name,
            "summary": content[:200],
        }

    if not isinstance(data, dict):
        return {
            "name": tool_name,
            "summary": content[:200],
        }

    if tool_name == "observe_window":
        control_count = data.get("control_count")
        if control_count is None:
            control_count = len(data.get("controls") or [])
        return {
            "name": tool_name,
            "summary": f"发现 {control_count} 个控件",
        }

    if tool_name == "click_control":
        ok = bool(data.get("ok"))
        control_label = format_control_label(data.get("control"))
        if ok:
            summary = f"成功点击 {control_label}"
        else:
            summary = data.get("error") or f"点击 {control_label} 失败"
        return {
            "name": tool_name,
            "summary": summary,
            "ok": ok,
        }

    if tool_name == "select_combobox_item":
        ok = bool(data.get("ok"))
        selected = data.get("selected") or data.get("best_match") or ""
        if ok:
            summary = f"选择 ComboBox 选项「{selected}」"
        else:
            summary = data.get("error") or f"选择 ComboBox 选项「{selected}」失败"
        return {
            "name": tool_name,
            "summary": summary,
            "ok": ok,
        }

    ok = data.get("ok")
    summary = data.get("message") or data.get("error") or content[:200]
    result = {
        "name": tool_name,
        "summary": str(summary),
    }
    if isinstance(ok, bool):
        result["ok"] = ok
    return result


def format_control_label(control: Any) -> str:
    """把工具结果中的 control 摘要格式化成短标签。"""
    if not isinstance(control, dict):
        return "未知控件"

    control_type = control.get("type") or "控件"
    control_name = (
        control.get("name")
        or control.get("automation_id")
        or control.get("id")
        or "未命名"
    )
    return f"{control_type}「{control_name}」"


def build_worker_result(
    action: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    final_worker_text: str,
) -> dict[str, Any]:
    """生成 execution_log 的单条 worker 执行记录。"""
    failed_results = [result for result in tool_results if result.get("ok") is False]
    if failed_results:
        status = "failed"
        error = failed_results[0].get("summary")
    elif looks_like_worker_failure(final_worker_text):
        status = "failed"
        error = final_worker_text
    elif tool_results or final_worker_text:
        status = "success"
        error = None
    else:
        status = "unknown"
        error = None

    if error:
        summary = error
    elif tool_results:
        summary = tool_results[-1].get("summary", "")
    else:
        summary = final_worker_text or "worker 未返回结果摘要。"

    return {
        "action_id": action.get("id"),
        "action_type": action.get("type"),
        "instruction": action.get("instruction"),
        "status": status,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "summary": summary,
        "error": error,
    }


# 没有失败工具结果时，用 worker 最终文本兜底判断失败
def looks_like_worker_failure(text: str) -> bool:
    """根据 worker 最终回复判断没有失败工具结果时的失败状态。"""
    failure_keywords = ("失败", "找不到", "没有找到", "缺少", "无法", "不能")
    return bool(text and any(keyword in text for keyword in failure_keywords))


def format_worker_memory_message(worker_result: dict[str, Any]) -> str:
    """生成写入 state['messages'] 的 worker 摘要。"""
    tool_names = [
        str(tool_call.get("name"))
        for tool_call in worker_result.get("tool_calls", [])
        if tool_call.get("name")
    ]
    tool_chain = " -> ".join(tool_names) if tool_names else "无"
    return (
        "[worker_result]\n"
        f"action {worker_result.get('action_id')}「"
        f"{worker_result.get('instruction')}」执行完成。\n"
        f"实际工具调用：{tool_chain}。\n"
        f"结果摘要：{worker_result.get('summary')}。"
    )


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

manual_path = r"manual\说明书.md"

import time


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
    # 本次修改：planner 再次被调用时可以读取此前写入 messages 的执行记忆。
    planner_messages = list(state.get("messages") or [])
    planner_messages.append(HumanMessage(content=content))
    # print("-" * 80)
    # print(f"[上下文]\n{planner_messages}")
    # print("-" * 80)
    # time.sleep(10)  # 等待用户阅读上下文
    result = planer_agent.invoke(
        {"messages": planner_messages},
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

    # 本次修改：记录 planner 初始计划版本，并写入一条 planner 可读记忆。
    plan_versions = list(state.get("plan_versions") or [])
    plan_versions.append(
        {
            "version": len(plan_versions) + 1,
            "reason": "initial_plan",
            "actions": actions,
        }
    )

    return {
        "actions": actions,
        "plan_versions": plan_versions,
        "messages": [
            HumanMessage(content=format_plan_request_memory_message(user_request)),
            AIMessage(content=format_plan_memory_message(actions)),
        ],
    }


# 本次修改：planner 计划摘要会进入全局 messages，后续 replanner 可直接读取。
def format_plan_request_memory_message(user_request: str) -> str:
    """生成写入 state['messages'] 的 planner 请求摘要。"""
    return (
        "[planner_request]\n"
        f"用户需求：{user_request}\n"
        "已向 planner 提供应用使用说明书。"
    )


def format_plan_memory_message(actions: list[dict[str, Any]]) -> str:
    """生成 planner 初始计划摘要。"""
    lines = ["[planner]", "已生成初始计划："]
    for action in actions:
        lines.append(
            f"{action.get('id')}. "
            f"{action.get('type')} - "
            f"{action.get('instruction')}"
        )
    return "\n".join(lines)


""" route """


def route_after_plan(state) -> str:
    """计划后路由：没有动作则结束，有动作则进入 worker。"""
    actions = state.get("actions") or []
    if not actions:
        return END

    return "worker_node"


def route_after_human(state) -> str:
    """人工确认后路由：动作全部完成则结束，否则继续 worker。"""
    last_human_event = latest_human_event(state)
    if last_human_event and last_human_event.get("status") == "rejected":
        return END

    actions = state.get("actions") or []
    current_step = state.get("current_step", 0)

    if current_step >= len(actions):
        return END

    return "worker_node"


# 本次修改：读取最近一次人工事件，让 route 能根据人工拒绝直接结束。
def latest_human_event(state) -> dict[str, Any] | None:
    """读取最近一次人工确认事件。"""
    human_events = state.get("human_events") or []
    if not human_events:
        return None
    return human_events[-1]


""" human node """


def human_node(state):
    """对需要人工确认的工具调用暂停图执行。"""
    last_tool = state.get("last_tool")
    if not should_confirm_last_tool(last_tool):
        return {}

    response = interrupt({"message": "请在选完成后按 y 确认"})

    response_text = str(response or "").strip().lower()
    confirmed = response_text == "y"
    if confirmed:
        print(f"[人工确认完成]")
    else:
        print(f"[人工阻止继续执行]")

    # 本次修改：把人工确认写入结构化记忆和 planner 可读 messages。
    last_worker_result = state.get("last_worker_result") or {}
    human_event = {
        "action_id": last_worker_result.get("action_id"),
        "instruction": last_worker_result.get("instruction"),
        "reason": "点击打开类控件后需要人工确认",
        "response": response_text,
        "status": "confirmed" if confirmed else "rejected",
    }
    human_events = list(state.get("human_events") or [])
    human_events.append(human_event)

    return {
        "human_events": human_events,
        "messages": [HumanMessage(content=format_human_memory_message(human_event))],
    }


# 本次修改：human 摘要只记录确认事实，不混入 worker 原始工具消息。
def format_human_memory_message(human_event: dict[str, Any]) -> str:
    """生成写入 state['messages'] 的人工确认摘要。"""
    if human_event.get("status") == "rejected":
        return (
            "[human]\n"
            f"action {human_event.get('action_id')}「"
            f"{human_event.get('instruction')}」后用户输入「"
            f"{human_event.get('response')}」，已阻止继续执行。"
        )

    return (
        "[human]\n"
        f"action {human_event.get('action_id')}「"
        f"{human_event.get('instruction')}」后用户已确认，可以继续执行后续步骤。"
    )


def should_confirm_last_tool(last_tool: dict[str, Any] | None) -> bool:
    """判断最后一次工具调用是否需要人工确认。"""
    if not last_tool:
        return False

    if last_tool.get("name") != "click_control":
        return False

    args = last_tool.get("args") or {}
    control_name = str(args.get("name") or "")
    return "打开" in control_name
