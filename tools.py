from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pyautogui
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pywinauto import Desktop
from pywinauto.findwindows import ElementNotFoundError

load_dotenv()
window_title = os.getenv("SCREEN_AGENT_WINDOW_TITLE")


@dataclass
class ControlRecord:
    """保存一个 UIA 控件的 Python 内部记录。

    注意：
    - LLM 看不到真实的 control 对象。
    - LLM 只能看到 control_id，例如 c_1。
    - Python 通过 control_id 在 registry 里找回真实控件对象。
    """

    control_id: str
    control: Any
    summary: dict[str, Any]


class UIAToolsState:
    # observe_window 会保留这些类型的控件。
    # 这样可以避免把大量无意义布局节点都发给 LLM。
    INTERESTING_TYPES = {
        "Button",
        "CheckBox",
        "ComboBox",
        "Edit",
        "Hyperlink",
        "ListItem",
        "MenuItem",
        "RadioButton",
        "TabItem",
        "Text",
        "TreeItem",
        "DataItem",
        "Pane",
        "Document",
        "Group",
        "Custom",
        "ToolBar",
        "MenuBar",
        "Tab",
        "List",
        "Tree",
        "Table",
    }

    # 这些类型通常可以直接点击。
    # 这个信息会作为 actionable_hint 返回给 LLM。
    ACTIONABLE_TYPES = {
        "Button",
        "CheckBox",
        "TabItem",
        "ComboBox",
    }

    def __init__(
        self,
        max_controls: int = 180,
        focus_window: bool = True,
    ) -> None:
        # 当前仅支持 Windows。
        if sys.platform != "win32":
            raise RuntimeError("当前版本仅支持 Windows")

        self.window_title = window_title
        self.max_controls = max_controls
        self.focus_window = focus_window

        # 使用 pywinauto 的 UIA 后端连接桌面。
        self.desktop = Desktop(backend="uia")

        # 当前目标窗口。第一次 observe 时会自动连接。
        self.window: Any | None = None

        # 最近一次 observe_window 生成的控件注册表。
        # 形式大概是 {"c_1": ControlRecord(...), "c_2": ControlRecord(...)}。
        self.registry: dict[str, ControlRecord] = {}

    def connect_window(self) -> None:
        """查找并连接目标窗口。"""
        self.window = self.desktop.window(
            title_re=f".*{re.escape(self.window_title)}.*"
        )
        self.window.wait("exists enabled visible ready", timeout=5)
        self._activate_window()

    def _activate_window(self) -> None:
        """确保目标窗口已连接，并尽量切到前台。"""
        if self.window is None:
            self.connect_window()
            return

        if self.focus_window:
            try:
                self.window.set_focus()
            except Exception:
                try:
                    self.window.wrapper_object().set_focus()
                except Exception:
                    pass

        # 给系统一点时间完成窗口激活。
        time.sleep(0.2)

    def observe_window_impl(self) -> dict[str, Any]:
        """读取当前窗口控件，并刷新控件注册表。"""
        self._activate_window()
        assert self.window is not None

        # control_id 是短期有效的。
        # 每次 observe 都会重新生成 c_1、c_2、c_3...
        # 所以点击或选择后，如果界面变化，应重新 observe。
        self.registry.clear()

        try:
            current_window_title = self.window.window_text()
        except Exception:
            current_window_title = self.window_title or ""

        try:
            descendants = self.window.descendants()
        except ElementNotFoundError:
            # 如果窗口对象因为界面变化失效，就重新连接窗口再读取。
            self.connect_window()
            assert self.window is not None
            descendants = self.window.descendants()

        controls: list[dict[str, Any]] = []
        for control in descendants:
            if len(controls) >= self.max_controls:
                break

            summary = self._summarize_control(control)
            if summary is None:
                continue

            control_id = f"c_{len(controls) + 1}"
            summary["id"] = control_id

            self.registry[control_id] = ControlRecord(
                control_id=control_id,
                control=control,
                summary=summary,
            )
            controls.append(summary)

        return {
            "ok": True,
            "window_title": current_window_title,
            "control_count": len(controls),
            "controls": controls,
        }

    def click_control_impl(self, control_id: str) -> dict[str, Any]:
        """根据最近一次 observe_window 返回的 control_id 点击控件。"""
        record = self.registry.get(control_id)
        if record is None:
            return {
                "ok": False,
                "error": f"未知 control_id: {control_id!r}。请先调用 observe_window。",
            }

        self._activate_window()

        control = record.control
        summary = record.summary
        control_type = summary.get("type", "")

        if control_type not in self.ACTIONABLE_TYPES:
            return {
                "ok": False,
                "error": f"当前只支持点击可操作控件，收到的是 {control_type!r}。",
                "control": summary,
            }

        try:
            x, y = self._center(control)
            pyautogui.click(x, y)
            time.sleep(0.2)
            return self._success(
                "coordinate_click",
                record,
                extra={"x": x, "y": y},
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"操作 {control_id} 失败: {exc}",
                "control": summary,
            }

    def display_tool_args(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        """把工具内部参数转换成更适合外部状态记录的参数。"""
        control_id = tool_args.get("control_id")
        if not control_id:
            return tool_args

        record = self.registry.get(control_id)
        if record is None:
            return tool_args

        display_args = {
            key: value for key, value in tool_args.items() if key != "control_id"
        }
        display_args["name"] = (
            record.summary.get("name")
            or record.summary.get("automation_id")
            or control_id
        )
        return display_args

    def select_combobox_item_impl(
        self,
        control_id: str,
        item_pattern: str,
    ) -> dict[str, Any]:
        """选择 ComboBox 下拉框中的选项，优先正则匹配，其次宽松相似匹配。"""
        record = self.registry.get(control_id)
        if record is None:
            return {
                "ok": False,
                "error": f"未知 control_id: {control_id!r}。请先调用 observe_window。",
            }

        control = record.control
        summary = record.summary
        if summary.get("type") != "ComboBox":
            return {
                "ok": False,
                "error": f"{control_id} 不是 ComboBox，而是 {summary.get('type')!r}。",
                "control": summary,
            }

        self._activate_window()

        try:
            expand_method = self._expand_combobox(control)
            time.sleep(0.35)

            items = self._combobox_popup_items(control)
            available_items = [self._control_text(item) for item in items]
            available_items = [item for item in available_items if item]

            best_item: Any | None = None
            best_text = ""
            best_score = 0.0
            best_reason = ""

            for item in items:
                text = self._control_text(item)
                score, reason = self._match_item_score(item_pattern, text)
                if score > best_score:
                    best_item = item
                    best_text = text
                    best_score = score
                    best_reason = reason

            if best_item is None or best_score < 0.55:
                self._collapse_combobox(control)
                return {
                    "ok": False,
                    "error": f"展开 ComboBox 后没有找到足够相似的选项: {item_pattern!r}",
                    "available_items": available_items,
                    "best_match": best_text,
                    "best_score": round(best_score, 3),
                    "control": summary,
                }

            try:
                best_item.click_input()
            except Exception:
                x, y = self._center(best_item)
                pyautogui.click(x, y)

            time.sleep(0.25)
            return {
                "ok": True,
                "method": "select_combobox_item",
                "expand_method": expand_method,
                "control_id": control_id,
                "requested_pattern": item_pattern,
                "selected": best_text,
                "match_score": round(best_score, 3),
                "match_reason": best_reason,
                "available_items": available_items,
                "control": summary,
                "next_step_hint": "如果还要继续操作，请再次调用 observe_window 确认界面状态。",
            }
        except Exception as exc:
            self._collapse_combobox(control)
            return {
                "ok": False,
                "error": f"选择 ComboBox 选项失败: {exc}",
                "control": summary,
            }

    def _control_text(self, control: Any) -> str:
        """读取控件文本。"""
        try:
            info = control.element_info
            return (control.window_text() or info.name or "").strip()
        except Exception:
            return ""

    def _expand_combobox(self, control: Any) -> str:
        """展开 ComboBox，优先使用 UIA expand，失败时退回坐标点击。"""
        expand = getattr(control, "expand", None)
        if expand is not None:
            try:
                expand()
                return "expand"
            except Exception:
                pass

        try:
            rect = control.rectangle()
            x = rect.right - 10
            y = round((rect.top + rect.bottom) / 2)
            pyautogui.click(x, y)
            return "arrow_click"
        except Exception:
            x, y = self._center(control)
            pyautogui.click(x, y)
            return "center_click"

    def _collapse_combobox(self, control: Any) -> None:
        """收起 ComboBox；失败时忽略。"""
        collapse = getattr(control, "collapse", None)
        if collapse is not None:
            try:
                collapse()
            except Exception:
                pass

    def _combobox_popup_items(self, control: Any) -> list[Any]:
        """读取 ComboBox 展开后的候选项。"""
        items: list[Any] = []

        try:
            items.extend(control.descendants(control_type="ListItem"))
        except Exception:
            pass

        try:
            combo_rect = control.rectangle()
            for item in self.desktop.descendants(control_type="ListItem"):
                try:
                    rect = item.rectangle()
                    visible = item.is_visible()
                    offscreen = item.element_info.is_offscreen
                except Exception:
                    continue

                vertical_near = (
                    combo_rect.top - 30 <= rect.top <= combo_rect.bottom + 700
                )
                horizontal_overlap = (
                    rect.right >= combo_rect.left and rect.left <= combo_rect.right
                )
                if visible and not offscreen and vertical_near and horizontal_overlap:
                    items.append(item)
        except Exception:
            pass

        result: list[Any] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            text = self._control_text(item)
            rect_text = ""
            try:
                rect = item.rectangle()
                rect_text = f"{rect.left},{rect.top},{rect.right},{rect.bottom}"
            except Exception:
                pass

            key = (text, rect_text)
            if text and key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def _match_item_score(self, pattern: str, text: str) -> tuple[float, str]:
        """计算候选项和用户目标的匹配分数。"""
        pattern = (pattern or "").strip()
        text = (text or "").strip()
        if not pattern or not text:
            return 0.0, "empty"

        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return 1.0, "regex"
        except re.error:
            # 用户给的正则不合法时，不直接失败，继续走宽松匹配。
            pass

        pattern_norm = self._normalize_match_text(pattern)
        text_norm = self._normalize_match_text(text)
        if not pattern_norm or not text_norm:
            return 0.0, "empty_normalized"

        if pattern_norm == text_norm:
            return 0.95, "normalized_equal"

        if pattern_norm in text_norm or text_norm in pattern_norm:
            return 0.85, "contains"

        ratio = SequenceMatcher(None, pattern_norm, text_norm).ratio()
        return ratio, "similarity"

    def _normalize_match_text(self, text: str) -> str:
        """规范化匹配文本，忽略空格、下划线、短横线和加号。"""
        return re.sub(r"[\s_\-+]+", "", text).casefold()

    def _summarize_control(self, control: Any) -> dict[str, Any] | None:
        """把真实 UIA 控件压缩成 LLM 可读的 JSON 摘要。"""
        try:
            element_info = control.element_info
            control_type = element_info.control_type or ""
            name = (control.window_text() or element_info.name or "").strip()
            automation_id = getattr(element_info, "automation_id", "") or ""
            class_name = getattr(element_info, "class_name", "") or ""
        except Exception:
            return None

        if control_type not in self.INTERESTING_TYPES:
            return None

        try:
            rect = control.rectangle()
            left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
            width = max(0, right - left)
            height = max(0, bottom - top)
        except Exception:
            return None

        # 没有面积的控件通常不可见或不可操作。
        if width <= 0 or height <= 0:
            return None

        # 没有 name 和 automation_id 的控件通常对 LLM 没有意义。
        if not name and not automation_id:
            return None

        enabled = self._safe_bool(lambda: control.is_enabled(), default=True)
        visible = self._safe_bool(lambda: control.is_visible(), default=True)
        offscreen = self._safe_bool(lambda: element_info.is_offscreen, default=False)

        # 屏幕外控件不适合坐标点击，过滤掉。
        if offscreen:
            return None

        parent_name, parent_type = self._parent_info(control)

        # 避免超长文本占用太多上下文。
        if len(name) > 120:
            name = name[:117] + "..."

        return {
            "id": "",
            "type": control_type,
            "name": name,
            "automation_id": automation_id,
            "class_name": class_name,
            "enabled": enabled,
            "visible": visible and not offscreen,
            "rect": [left, top, right, bottom],
            "parent_name": parent_name,
            "parent_type": parent_type,
            "actionable_hint": control_type in self.ACTIONABLE_TYPES,
        }

    def _parent_info(self, control: Any) -> tuple[str | None, str | None]:
        """读取父控件信息，帮助 LLM 理解控件所在区域。"""
        try:
            parent = control.parent()
            if parent is None:
                return None, None

            info = parent.element_info
            name = (parent.window_text() or info.name or "").strip() or None
            control_type = info.control_type or None

            if name and len(name) > 80:
                name = name[:77] + "..."

            return name, control_type
        except Exception:
            return None, None

    def _center(self, control: Any) -> tuple[int, int]:
        """计算控件中心点，作为坐标点击的兜底方案。"""
        rect = control.rectangle()
        return (
            round((rect.left + rect.right) / 2),
            round((rect.top + rect.bottom) / 2),
        )

    def _safe_bool(self, fn: Any, default: bool) -> bool:
        """安全读取布尔属性，读取失败时返回默认值。"""
        try:
            return bool(fn())
        except Exception:
            return default

    def _success(
        self,
        method: str,
        record: ControlRecord,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """统一生成工具成功结果。"""
        result = {
            "ok": True,
            "method": method,
            "control_id": record.control_id,
            "control": record.summary,
            "message": (
                f"已操作 {record.control_id}: "
                f"{record.summary.get('type')} {record.summary.get('name')!r}"
            ),
            "next_step_hint": "如果还要继续操作，请再次调用 observe_window。",
        }
        if extra:
            result.update(extra)
        return result


class ClickControlInput(BaseModel):
    """click_control 的输入参数。"""

    control_id: str = Field(
        description=(
            "要操作的控件 ID。必须来自最近一次 observe_window 返回的 "
            "controls[*].id，例如 c_1、c_2、c_12。不要传控件对象、"
            "控件完整 JSON 或坐标。"
        )
    )


class SelectComboBoxItemInput(BaseModel):
    """select_combobox_item 的输入参数。"""

    control_id: str = Field(
        description=(
            "ComboBox 控件 ID。必须来自最近一次 observe_window 返回的 "
            "controls[*].id，且对应控件 type 必须是 ComboBox。"
        )
    )
    item_pattern: str = Field(
        description=(
            "要选择的下拉选项文本或正则表达式。支持部分匹配和相似匹配，"
            "例如 large、base、base.*plus、tiny|small。"
        )
    )


def make_uia_tools(state: UIAToolsState):
    """创建绑定到同一个 UIAToolsState 实例的 LangChain tools。"""

    @tool
    def observe_window() -> str:
        """观察当前目标窗口，并返回可供 Agent 推理和操作的 UIA 控件列表。

        返回 JSON 字符串，结构如下：
        {
          "ok": true,
          "window_title": "窗口标题",
          "control_count": 3,
          "controls": [
            {
              "id": "c_1",
              "type": "Button",
              "name": "保存",
              "automation_id": "btnSave",
              "class_name": "Button",
              "enabled": true,
              "visible": true,
              "rect": [left, top, right, bottom],
              "parent_name": "父控件名称",
              "parent_type": "Pane",
              "actionable_hint": true
            }
          ]
        }

        字段说明：
        - id：短期控件 ID。click_control 和 select_combobox_item 只能使用最近一次 observe_window 返回的 id。
        - type：UIA 控件类型，例如 Button、CheckBox、RadioButton、MenuItem、TabItem、ListItem、TreeItem、ComboBox、Hyperlink。
        - name：控件名称，通常是按钮文字、菜单文字、选项文字或可访问性名称。
        - automation_id：应用开发者设置的自动化 ID；如果有，通常比 name 更稳定。
        - class_name：底层控件类名，主要用于调试和区分控件来源。
        - enabled：控件当前是否可用。
        - visible：控件当前是否可见且不在屏幕外。
        - rect：控件屏幕矩形，仅用于理解布局，不要直接当作点击参数。
        - parent_name/parent_type：父控件信息，用于区分同名控件所在区域。
        - actionable_hint：该控件类型是否大概率可点击。

        使用规则：
        - 第一次执行任何点击或下拉选择前，必须先调用本工具。
        - 每次 click_control 或 select_combobox_item 后，如果还要继续操作，应重新调用本工具刷新控件 ID。
        - observe_window 只做观察，不会主动展开 ComboBox；需要选择下拉选项时，请调用 select_combobox_item。
        """
        return json.dumps(state.observe_window_impl(), ensure_ascii=False)

    @tool(args_schema=ClickControlInput)
    def click_control(control_id: str) -> str:
        """点击最近一次 observe_window 返回的某个可操作控件。

        适用控件：
        - Button、CheckBox、TabItem、ComboBox 等 ACTIONABLE_TYPES 中的控件。
        - 内部会取控件中心点，并用 pyautogui 执行真实鼠标点击。

        参数：
        - control_id：必须是最近一次 observe_window 返回的 controls[*].id，例如 "c_3"。

        不要传：
        - 控件对象
        - 控件完整 JSON
        - 坐标
        - 自己编造的 ID

        注意：
        - 如果用户要选择 ComboBox 的某个下拉选项，应优先使用 select_combobox_item，而不是先 click_control 展开再点 ListItem。
        - 操作完成后，如果还要继续操作，应再次调用 observe_window。
        """
        return json.dumps(state.click_control_impl(control_id), ensure_ascii=False)

    @tool(args_schema=SelectComboBoxItemInput)
    def select_combobox_item(control_id: str, item_pattern: str) -> str:
        """选择最近一次 observe_window 返回的某个 ComboBox 下拉选项。

        使用方式：
        - 先调用 observe_window，找到 type 为 ComboBox 的控件 id。
        - 再调用本工具，传入 control_id 和 item_pattern。
        - 本工具会临时展开该 ComboBox，读取候选 ListItem，选择最匹配的一项。

        匹配规则：
        - 优先把 item_pattern 当作正则表达式，用 re.search 忽略大小写匹配。
        - 如果正则不合法或没有匹配，会退回到规范化文本匹配。
        - 规范化会忽略空格、下划线、短横线和加号。
        - 最后会使用 SequenceMatcher 做相似度匹配，最低接受分数为 0.55。

        示例：
        - select_combobox_item("c_3", "large")
        - select_combobox_item("c_3", "base.*plus")
        - select_combobox_item("c_3", "tiny|small")

        返回值会包含：
        - selected：实际选择的选项文本。
        - available_items：展开后看到的候选项。
        - match_score / match_reason：匹配分数和匹配方式，便于调试。

        注意：
        - control_id 必须来自最近一次 observe_window。
        - 操作完成后，如果还要继续操作，应再次调用 observe_window。
        """
        return json.dumps(
            state.select_combobox_item_impl(control_id, item_pattern),
            ensure_ascii=False,
        )

    return [observe_window, click_control, select_combobox_item]


if __name__ == "__main__":
    """测试用例。"""

    state = UIAToolsState()
    start = time.time()
    window = json.dumps(state.observe_window_impl(), ensure_ascii=False)
    end = time.time()
    # print(window)
    print(f"耗时：{end - start:.2f} 秒")
