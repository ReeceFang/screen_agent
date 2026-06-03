from __future__ import annotations
import json
import sys
from uuid import uuid4

import pyautogui
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from graph import (
    AgentState,
    create_initial_state,
    worker_node,
    planer_node,
    human_node,
    route_after_plan,
    route_after_human,
)


def create_graph():
    """创建图。"""
    builder = StateGraph(AgentState)
    # 添加节点
    builder.add_node("planer_node", planer_node)
    builder.add_node("worker_node", worker_node)
    builder.add_node("human_node", human_node)
    # 添加普通边
    builder.add_edge(START, "planer_node")
    builder.add_edge("worker_node", "human_node")
    # 添加条件边
    builder.add_conditional_edges(
        "planer_node",
        route_after_plan,
        {END: END, "worker_node": "worker_node"},
    )
    builder.add_conditional_edges(
        "human_node",
        route_after_human,
        {END: END, "worker_node": "worker_node"},
    )

    graph = builder.compile(checkpointer=MemorySaver())
    return graph


def run_one(graph, message: str) -> None:
    """运行一次图；如果遇到 interrupt，则等待用户确认后 resume。"""
    state_patch = {
        "user_request": message,
    }
    # config = {"configurable": {"thread_id": f"run-{uuid4()}"}}
    config = {"configurable": {"thread_id": "thread_1"}}

    result = graph.invoke(create_initial_state(state_patch), config=config)
    while has_interrupt(result):
        print()
        print("[中断确认]")
        print_interrupts(result)
        answer = input("输入 y 确认继续：").strip().lower()
        result = graph.invoke(Command(resume=answer), config=config)

    print()
    # print("[图执行完成]")


def has_interrupt(result: dict) -> bool:
    """判断 LangGraph invoke 结果里是否包含 interrupt。"""
    return bool(result.get("__interrupt__"))


def print_interrupts(result: dict) -> None:
    """打印 interrupt payload，方便用户确认。"""
    interrupts = result.get("__interrupt__") or []
    for item in interrupts:
        payload = getattr(item, "value", item)
        if isinstance(payload, (dict, list)):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload)


def main() -> None:
    """程序入口"""
    if sys.platform != "win32":
        raise RuntimeError(
            "这个 UIA Agent 使用 Windows UI Automation，只能在 Windows 上运行。"
        )

    # PyAutoGUI 安全开关：
    # 鼠标移动到屏幕左上角会触发异常，从而紧急停止自动化操作
    pyautogui.FAILSAFE = True

    print("UIA tool-calling Agent 已启动。")
    graph = create_graph()

    while True:
        print("\n🤖 : 请问有什么可以帮助你的？")
        message = input("我 : ").strip()
        if not message:
            continue
        if message.lower() in {"q", "quit", "exit", "退出"}:
            break
        run_one(graph, message)


if __name__ == "__main__":
    main()
