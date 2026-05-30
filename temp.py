from __future__ import annotations

import os
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv
from pywinauto import Desktop


def safe_call(fn: Any, default: Any = "") -> Any:
    try:
        return fn()
    except Exception:
        return default


def control_info(control: Any) -> dict[str, Any]:
    info = control.element_info
    rect = safe_call(control.rectangle, None)
    if rect is None:
        rect_text = ""
    else:
        rect_text = f"[{rect.left}, {rect.top}, {rect.right}, {rect.bottom}]"

    return {
        "type": info.control_type or "",
        "name": (safe_call(control.window_text, "") or info.name or "").strip(),
        "automation_id": getattr(info, "automation_id", "") or "",
        "class_name": getattr(info, "class_name", "") or "",
        "enabled": safe_call(control.is_enabled, ""),
        "visible": safe_call(control.is_visible, ""),
        "offscreen": getattr(info, "is_offscreen", ""),
        "rect": rect_text,
    }


def print_control(control: Any, depth: int, index: int) -> None:
    item = control_info(control)
    indent = "  " * depth
    print(
        f"{indent}{index}. "
        f"type={item['type']!r} "
        f"name={item['name']!r} "
        f"automation_id={item['automation_id']!r} "
        f"class={item['class_name']!r} "
        f"enabled={item['enabled']} "
        f"visible={item['visible']} "
        f"offscreen={item['offscreen']} "
        f"rect={item['rect']}"
    )


def walk(control: Any, depth: int = 0, max_depth: int = 20) -> None:
    if depth > max_depth:
        return

    children = safe_call(control.children, [])
    for index, child in enumerate(children, start=1):
        print_control(child, depth, index)
        walk(child, depth + 1, max_depth)


def main() -> None:
    if sys.platform != "win32":
        raise RuntimeError("This script only runs on Windows.")

    load_dotenv()
    window_title = os.getenv("SCREEN_AGENT_WINDOW_TITLE")
    if not window_title:
        raise RuntimeError("Missing SCREEN_AGENT_WINDOW_TITLE in .env.")

    desktop = Desktop(backend="uia")
    window = desktop.window(title_re=f".*{re.escape(window_title)}.*")
    window.wait("exists enabled visible ready", timeout=10)

    try:
        window.set_focus()
    except Exception:
        try:
            window.wrapper_object().set_focus()
        except Exception:
            pass

    time.sleep(0.3)

    print(f"target window keyword: {window_title!r}")
    print(f"matched window title: {window.window_text()!r}")
    print("-" * 120)
    print_control(window, 0, 0)
    walk(window)


if __name__ == "__main__":
    main()
