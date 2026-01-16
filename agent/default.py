import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from rich.console import Console
from typing import Any, Dict, List, Optional
from tools.base_tools import ToolRegistry, get_tool_registry, register_tool
from tools.file_system import FileSystemTool
from traj import TraceTrack

logger = logging.getLogger(name=__name__)
agent_console = Console()

class DefaultAgent:
    """
    This is the code agent.
    """
    def __init__(self, config) -> None:
        self.config = config
        self.max_turns = self.config.max_turns
        self.api_key = self.config.api_key
        self.api_url = self.config.api_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.model = self.config.model_name

        self.tool_registry = get_tool_registry()
        register_tool(tool=FileSystemTool(read_only=False))
        agent_console.print("✅ Tools registered\n", style="green")
        agent_console.print(self.tool_registry.get_registry_summary())

        

        self.system_message = (
            "You are SafeFlow DefaultAgent.\n"
            "You can use tools via function calling to operate on the real filesystem.\n"
            "When you need to create or modify files, call the file_system tool functions.\n"
            "Return concise final answers.\n"
            "Important:\n"
            "- Use function calls when interacting with the filesystem.\n"
            "- Prefer writing a complete working solution.\n"
        )
    
    def run(self, user_prompt: str, trace_track: TraceTrack):

        tools_schema = [
            {"type": "function", "function": f}
            for f in self.tool_registry.to_openai_functions(enabled_only=True)
        ]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_prompt},
        ]

        trace_track.step(f"[agent] start | model={self.model} | max_turns={self.max_turns}")
        trace_track.step(f"[agent] user_prompt:\n{user_prompt}")
        trace_track.step(f"[agent] tools_count={len(tools_schema)}")

        for turn in range(1, self.max_turns + 1):
            trace_track.step(f"[llm] turn={turn} request | messages={len(messages)}")

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
            )

            choice = resp.choices[0]
            msg = choice.message
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None) or []

            trace_track.step(
                f"[llm] turn={turn} response | finish_reason={getattr(choice, 'finish_reason', None)} | "
                f"tool_calls={len(tool_calls)} | content_preview={self._preview(content)}"
            )

            # 记录 assistant 消息（包含 tool_calls）
            assistant_entry: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_entry)

            # 没有 tool_calls：认为模型在给最终回答（提前退出）
            if not tool_calls:
                trace_track.step(f"[agent] done (no tool_calls) at turn={turn}")
                return {"success": True, "final": content, "messages": messages}

            # 有 tool_calls：逐个执行并把结果回填给模型
            for tc in tool_calls:
                fn_name = tc.function.name
                args_raw = tc.function.arguments or "{}"
                tool_name, function_name = self._route_function(fn_name)

                trace_track.step(
                    f"[tool] call | turn={turn} | tool_call_id={tc.id} | "
                    f"{tool_name}.{function_name} | args_preview={self._preview(args_raw, 1000)}"
                )

                # 解析参数（失败也要回填一个 tool error 给模型）
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object/dict")
                except Exception as e:
                    tool_result = {"success": False, "error": f"Bad tool arguments: {e}", "raw": args_raw}
                    trace_track.step(f"[tool] args_error | tool_call_id={tc.id} | error={e}")
                else:
                    tool_result = self.tool_registry.call_function(
                        tool_name=tool_name,
                        function_name=function_name,
                        parameters=args,
                    )
                    trace_track.step(
                        f"[tool] result | tool_call_id={tc.id} | success={tool_result.get('success')} | "
                        f"result_preview={self._preview(json.dumps(tool_result, ensure_ascii=False), 1000)}"
                    )

                # 回填 tool message 给 LLM
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

                # 如果模型调用了 finish_task：立刻提前退出
                # （finish_task 的结果也已经回填给模型了，但我们不需要再问下一轮）
                if tool_name == "file_system" and function_name == "finish_task":
                    trace_track.step(f"[agent] done (finish_task) at turn={turn}")
                    return {
                        "success": True,
                        "final": content,  # 可能为 None；你也可以改成固定字符串
                        "messages": messages,
                    }

        trace_track.step(f"[agent] failed: reached max_turns={self.max_turns}")
        return {
            "success": False,
            "error": f"Reached max_turns={self.max_turns} without finish_task or final response.",
            "messages": messages,
        }
    # ----------------- helpers -----------------

    def _route_function(self, openai_function_name: str) -> Tuple[str, str]:
        """
        Map OpenAI tool-call function name -> (tool_name, function_name)

        Your registry is keyed by tool_name then function_name.
        OpenAI tool-call uses only function name in schema; so default to 'file_system'.
        If you later choose to emit names like 'file_system.write_file', this supports it.
        """
        if "." in openai_function_name:
            tool_name, fn = openai_function_name.split(".", 1)
            return tool_name, fn
        return "file_system", openai_function_name
    
    @staticmethod
    def _preview(text: Optional[str], limit: int = 400) -> str:
        if text is None:
            return "None"
        s = str(text)
        if len(s) <= limit:
            return s
        return s[:limit] + f"...(truncated, {len(s)} chars)"
            
        


