# Screen Agent

一个基于 Windows UI Automation 的桌面操作 Agent 原型。当前项目面向本机 Windows 桌面应用，通过大模型理解用户指令，结合应用使用说明书拆解操作步骤，再调用 UIA 工具完成窗口观察、控件点击和 ComboBox 选项选择。

## 当前进展

- 已接入 LangGraph，形成 `planner -> worker -> human confirm` 的执行流程。
- 已接入 DashScope 兼容 OpenAI 接口，当前模型配置为 `qwen-plus`。
- Planner 会读取 `manual/推理.docx`，根据说明书和用户需求生成结构化动作列表。
- Worker 已具备 3 个 UIA 工具：
  - `observe_window()`：观察目标窗口并返回可用控件摘要。
  - `click_control(control_id)`：点击最近一次观察到的可操作控件。
  - `select_combobox_item(control_id, item_pattern)`：展开 ComboBox 并按文本、正则或相似度选择选项。
- 已加入人工确认节点：当最后一次点击目标名称包含“打开”时，会中断流程并等待用户输入 `y` 后继续。
- 已提供 `temp.py` 作为 UIA 控件树调试脚本，便于检查目标窗口控件结构。

## 项目结构

```text
screen_agent/
├── main.py                 # 程序入口，创建并运行 LangGraph
├── graph.py                # AgentState、planner/worker/human 节点和路由逻辑
├── agent.py                # LLM、planner agent、worker agent、说明书读取逻辑
├── tools.py                # Windows UIA 工具实现
├── temp.py                 # 目标窗口控件树调试脚本
├── manual/
│   ├── 推理.docx
│   └── 点监督自动标注与推理验证系统-使用说明.docx
└── memory/
    └── checkpoint.db       # 本地运行产生的 checkpoint 数据
```

## 运行环境

当前版本只支持 Windows，因为底层依赖 Windows UI Automation 和真实鼠标点击。

建议环境：

- Windows 10/11
- Python 3.11 或 3.12
- 目标桌面应用已启动，并且窗口标题能被 `SCREEN_AGENT_WINDOW_TITLE` 匹配到

安装依赖：

```powershell
pip install python-dotenv pyautogui pywinauto python-docx pydantic langchain langchain-openai langgraph langgraph-checkpoint-sqlite
```

## 环境变量

在项目根目录创建 `.env`：

```env
SCREEN_AGENT_WINDOW_TITLE="目标检测平台"
DASHSCOPE_API_KEY="your_dashscope_api_key"
```

说明：

- `SCREEN_AGENT_WINDOW_TITLE`：目标窗口标题关键字，程序会用正则匹配包含该关键字的窗口。
- `DASHSCOPE_API_KEY`：DashScope API Key。不要把真实密钥提交到公开仓库。

## 使用方式

先打开目标桌面应用，并确保目标窗口可见。

启动主程序：

```powershell
python main.py
```

也可以调整每次观察窗口时最多返回的控件数量：

```powershell
python main.py --max-controls 180
```

程序启动后会进入交互模式：

```text
UIA tool-calling Agent 已启动。

🤖 : 请问有什么可以帮助你的？
我 :
```

输入自然语言任务即可，例如：

```text
点击推理界面，然后打开图片并开始推理
```

退出程序：

```text
q
quit
exit
退出
```

## 调试控件树

如果 Agent 找不到目标控件，可以先运行调试脚本查看当前窗口的 UIA 控件结构：

```powershell
python temp.py
```

脚本会读取 `.env` 中的 `SCREEN_AGENT_WINDOW_TITLE`，匹配目标窗口后打印控件类型、名称、automation_id、class、可见状态和坐标范围。

## 执行流程

1. `main.py` 创建 LangGraph。
2. 用户输入需求。
3. `planner_node` 读取 `manual/推理.docx`，让 planner agent 输出动作列表。
4. `worker_node` 按顺序把每个动作交给 worker agent。
5. Worker agent 根据需要调用 `observe_window`、`click_control` 或 `select_combobox_item`。
6. 如果触发人工确认，`human_node` 暂停流程，等待用户输入 `y`。
7. 所有动作完成后，本轮任务结束。

## 当前限制

- 当前仅支持 Windows。
- 当前工具层只实现了窗口观察、点击控件和 ComboBox 选择。
- Planner 的动作类型中预留了 `input`、`wait`、`verify` 等类型，但 worker 工具目前还没有专门的文本输入、等待检测或结果校验工具。
- `click_control` 目前只允许点击 `Button`、`CheckBox`、`TabItem`、`ComboBox` 类型控件。
- `control_id` 只在最近一次 `observe_window()` 后短期有效，界面变化后需要重新观察。
- 程序会执行真实鼠标操作，运行时不要随意移动窗口或遮挡目标控件。

## 安全提示

- PyAutoGUI 的 failsafe 已开启。鼠标移动到屏幕左上角会触发异常，可作为紧急停止手段。
- Agent 系统提示中禁止未明确授权的危险操作，例如删除、付款、格式化、关闭不保存等。
- `.env` 中可能包含真实 API Key，建议加入 `.gitignore` 并只保留 `.env.example` 作为配置模板。

