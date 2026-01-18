import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from rich.console import Console
from tools.abs_tools import ToolRegistry
from tools.file_system import FileSystemTool
from tools.windowed_editor import WindowedEditorTool
from tools.planning_tool import PlanningTool
from tools.base_tools import BaseTools
from tools.env_management import EnvManagementTool
from traj import TraceTrack

logger = logging.getLogger(name=__name__)
agent_console = Console()

class DefaultAgent:
    """
    This is the code agent.
    """
    def __init__(self, config, agent_name: str, item_id: str, context_manager=None) -> None:
        self.config = config
        self.max_turns = self.config.max_turns
        self.api_key = self.config.api_key
        self.api_url = self.config.api_url
        self.agent_name = agent_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.model = self.config.model_name
        self.item_id = item_id
        self.trace_track = TraceTrack(root_dir=self.config.output_dir, agent_name=agent_name, item_id=item_id)

        # Reference to context manager (optional)
        self.context_manager = context_manager

        # Initialize tool registry with all available tools
        self.tool_registry = ToolRegistry()

        # Register all tools with same item_id
        self.tool_registry.register_tool(FileSystemTool(item_id=item_id))
        self.tool_registry.register_tool(WindowedEditorTool(item_id=item_id))
        self.tool_registry.register_tool(PlanningTool(item_id=item_id))
        self.tool_registry.register_tool(BaseTools(item_id=item_id))
        self.tool_registry.register_tool(EnvManagementTool(item_id=item_id))

        agent_console.print("✅ All Tools registered\n", style="green")
        agent_console.print(self.tool_registry.get_registry_summary())

        # Track initialization state at agent level
        self.session_initialized = False
        self.initialization_attempted = False

        # Enhanced state awareness
        self._last_context_check = None

        self.system_message = (
            "You are SafeFlow, an intelligent agent for coding and task management.\n\n"
            "CORE PRINCIPLES:\n"
            "- ALWAYS check your current work context before starting operations\n"
            "- Use absolute file paths for all operations\n"
            "- Be systematic and thorough in your approach\n"
            "- Use appropriate tools for each type of operation\n"
            "- Maintain awareness of your initialization status and work location\n"
            "- Verify task completion before calling base_tools__finish_task()\n\n"
            "## Available Tools (Clear Boundaries)\n"
            "- **base_tools**: ONLY initialization, workspace setup, git clone, task finish\n"
            "- **windowed_editor**: ONLY file editing with precise line-number control + file info\n"
            "- **file_system**: ONLY file discovery, glob search, semantic search, symbol search (AST)\n"
            "- **env_management**: ONLY bash commands, compilation, quality checks, testing\n"
            "- **planning_tool**: ONLY task planning, progress tracking, plan management\n\n"
            "## Initialization Flow\n"
            "1. Check status: `base_tools__session_state(action='check')`\n"
            "2. If needed: `base_tools__complete_initialization()` (ONCE only)\n"
            "3. Always work relative to your established work_root\n\n"

            "## Tool Usage Guidelines\n"
            "- **CRITICAL: Use ABSOLUTE paths**: All tools require absolute paths except base_tools__set_work_root\n"
            "- **Get work_root**: Use base_tools__session_state(action='get_context') to get your work_root\n"
            "- **Build absolute paths**: Combine work_root + filename (e.g., '/path/to/work/file.py')\n"
            "- **File editing**: windowed_editor for precise line-based changes (absolute paths only)\n"
            "- **File discovery**: file_system for finding and analyzing files (absolute paths only)\n"
            "- **Code verification**: env_management for bash, compilation, testing (absolute working_dir)\n"
            "- **Planning**: planning_tool for task management (absolute plan file paths only)\n\n"

            "## 🔥 Code Verification Protocol\n"
            "**Before presenting code to users, ALWAYS:**\n"
            "1. **Syntax**: `env_management__execute_bash('python -m py_compile file.py')`\n"
            "2. **Quality** (if available): `env_management__code_quality_check('flake8', '.')`\n"
            "3. **Fix & Re-test**: Iterate until everything works correctly\n\n"
            # "4. **Execute**: `env_management__execute_bash('python file.py')` - verify it runs!\n"


            "## Communication Rules\n"
            "- **NEVER empty content**: Always explain your actions in the content field\n"
            "- **Tool rationale**: Describe what each tool call will accomplish\n"
            "- **Systematic approach**: Complete verification before marking tasks done\n"
        )

    def get_current_work_context(self) -> Dict[str, Any]:
        """Get current work context from context manager."""
        if self.context_manager:
            return self.context_manager.get_current_context()

        # Fallback: try to get from base_tools
        for tool_name, tool in self.tool_registry.tools.items():
            if tool_name == "base_tools" and hasattr(tool, '_get_session_work_root'):
                return {
                    "work_root": tool._get_session_work_root(),
                    "initialized": tool._is_session_initialized() if hasattr(tool, '_is_session_initialized') else False,
                    "message": "Context from base_tools (limited info)"
                }

        return {"work_root": None, "message": "No context available"}

    def before_tool_call(self, tool_name: str, function_name: str) -> str:
        """Get context information before tool calls."""
        context = self.get_current_work_context()
        context_summary = ""

        if self.context_manager:
            context_summary = self.context_manager.get_context_summary_for_agent()
        else:
            context_summary = f"Work Root: {context.get('work_root', 'Not set')}"

        return context_summary

    def update_context_after_tool_call(self, tool_name: str, function_name: str,
                                     tool_result: Dict[str, Any], parameters: Dict[str, Any]) -> None:
        """Update context after tool calls."""
        if not self.context_manager:
            return

        # Update context based on tool usage
        if tool_name == "base_tools":
            if function_name == "base_tools__set_work_root" and tool_result.get("success"):
                work_root = tool_result.get("result", {}).get("work_root")
                if work_root:
                    self.context_manager.update_work_context(work_root=work_root)

            elif function_name == "base_tools__complete_initialization" and tool_result.get("success"):
                result = tool_result.get("result", {})
                work_root = result.get("work_root")
                task_analysis = result.get("task_analysis", {})

                self.context_manager.update_work_context(
                    work_root=work_root,
                    initialization_status="completed"
                )

                # Track completed steps
                for step in result.get("steps_completed", []):
                    self.context_manager.add_completed_step(step)

        elif tool_name == "windowed_editor" or tool_name == "file_system":
            # Track file operations
            if "path" in parameters:
                file_path = parameters["path"]
                operation = function_name.split("__")[-1] if "__" in function_name else function_name
                self.context_manager.track_active_file(file_path, operation)

    def get_enhanced_system_message(self) -> str:
        """Get system message enhanced with current context."""
        base_message = self.system_message

        if self.context_manager:
            context_summary = self.context_manager.get_context_summary_for_agent()

            enhanced_message = f"{base_message}\n\nCURRENT WORK CONTEXT:\n{context_summary}\n\n"

            # Add plan reminder if needed
            if self.context_manager.needs_plan_reminder():
                plans_summary = ", ".join([p.get("type", "unknown") for p in self.context_manager.current_plans[-3:]])
                enhanced_message += f"📋 REMINDER: You have active plans: {plans_summary}\n\n"

            return enhanced_message

        return base_message

    def _route_function(self, openai_function_name: str) -> Tuple[str, str]:
        """
        Map OpenAI tool-call function name -> (tool_name, function_name)

        Your registry is keyed by tool_name then function_name.
        OpenAI tool-call uses only function name in schema; so default to 'file_system'.
        If you later choose to emit names like 'file_system.write_file', this supports it.
        """
        fn = openai_function_name
        for tool_name, tool in self.tool_registry.tools.items():
            # tool.functions keys are callable names on that tool
            if fn in getattr(tool, "functions", {}):
                return tool_name, fn
        raise ValueError(f"No tool found for function {fn}")

    def _enhance_user_prompt_with_initialization(self, user_prompt: str) -> str:
        """
        Enhance user prompt with initialization requirements if needed.
        """
        # Check if user is explicitly asking for reinitialization
        reinit_keywords = ["reinitialize", "reset workspace", "start over", "force_reinit"]
        if any(keyword in user_prompt.lower() for keyword in reinit_keywords):
            self.session_initialized = False  # Reset state for forced reinit
            self.initialization_attempted = False
            return user_prompt  # Let user handle reinitialization themselves

        # If already initialized at agent level, don't require initialization again
        if self.session_initialized:
            return user_prompt

        # Check base_tools state as backup
        base_tools = None
        for tool_name, tool in self.tool_registry.tools.items():
            if tool_name == "base_tools":
                base_tools = tool
                break

        if base_tools and hasattr(base_tools, '_is_session_initialized'):
            if base_tools._is_session_initialized():
                self.session_initialized = True  # Sync agent state
                return user_prompt  # Already initialized

        # Only enhance prompt if initialization hasn't been attempted yet
        if not self.initialization_attempted:
            self.initialization_attempted = True
            return (
                f"USER TASK: {user_prompt}\n\n"
                "INITIALIZATION REQUIREMENT: You must initialize your workspace before proceeding.\n"
                "1. First call: base_tools__check_initialization_status()\n"
                "2. If not initialized, call: base_tools__complete_initialization(task_blob=\"" + user_prompt.replace('"', '\\"') + "\")\n"
                "3. IMMEDIATELY after initialization, create a detailed plan using planning_tool__create_plan() to break down the task into steps\n"
                "4. Then proceed with executing the plan step by step.\n\n"
                "Remember: You MUST create a plan after initialization and before starting work on the task.")
        else:
            # Initialization was attempted but maybe not completed, just return original prompt
            return user_prompt

    def _check_initialization_completion(self, function_name: str, tool_result: Dict[str, Any]) -> None:
        """
        Monitor tool calls to detect initialization completion and plan creation.
        """
        # Check if initialization-related functions completed successfully
        if tool_result.get("success") and function_name in [
            "base_tools__complete_initialization",
            "base_tools__set_work_root"
        ]:
            # Mark as initialized if complete_initialization succeeded
            if function_name == "base_tools__complete_initialization":
                self.session_initialized = True
                if self.context_manager:
                    self.context_manager.record_memory(
                        "milestone",
                        "Workspace initialization completed successfully - plan creation required next"
                    )

            # Also mark as initialized if set_work_root succeeded (backup check)
            elif function_name == "base_tools__set_work_root":
                result_data = tool_result.get("result", {})
                if result_data.get("session_initialized"):
                    self.session_initialized = True


        # Monitor plan creation
        elif function_name == "planning_tool__create_plan" and tool_result.get("success"):
            if self.context_manager:
                # Extract plan content from result
                plan_data = tool_result.get("result", {})
                plan_content = plan_data.get("plan_content", "Plan created successfully")

                # Record the plan in context manager
                self.context_manager.record_plan(
                    plan_content=plan_content,
                    plan_type="initial",
                    metadata={
                        "function_name": function_name,
                        "result": plan_data
                    }
                )
    def run(self, user_prompt: str):
        trace_track = self.trace_track

        # Record task start with context manager if available
        if self.context_manager:
            self.context_manager.record_memory(
                "task_start",
                f"DefaultAgent started task: {user_prompt}"
            )
        tools_schema = [
            {"type": "function", "function": f}
            for f in self.tool_registry.to_openai_functions(enabled_only=True)
        ]

        messages: List[Dict[str, Any]] = []

        def push(msg: Dict[str, Any]):
            """同时写入：给模型的 messages + 轨迹 trace(json, 带 time_stamp) + context_manager memory"""
            messages.append(msg)
            trace_track.add_message(msg)

            # Also record in context manager if available
            if self.context_manager:
                self.context_manager.add_default_agent_message(msg)

        # 初始化消息 - 使用增强的系统消息
        enhanced_system_message = self.get_enhanced_system_message()
        push({"role": "system", "content": enhanced_system_message})

        # 检查并引导初始化
        enhanced_prompt = self._enhance_user_prompt_with_initialization(user_prompt)
        push({"role": "user", "content": enhanced_prompt})

        # 如果你仍想额外记录调试信息，可以用 save_json，而不是 step
        trace_track.save_json(f"{self.agent_name}_tools_schema.json", tools_schema)

        for turn in range(1, self.max_turns + 1):

            # Check if plan reminder is needed (context manager)
            if self.context_manager:
                reminder_data = self.context_manager.check_plan_reminder_needed()
                if reminder_data:
                    # Add reminder message to conversation
                    reminder_msg = {"role": "system", "content": reminder_data["message"]}
                    push(reminder_msg)

            # Use context_manager's memory management to get appropriately sized messages
            # This ensures we don't exceed token limits while preserving important context
            actual_messages_for_llm = messages
            if self.context_manager:
                try:
                    # Get summarized messages if needed, otherwise original messages
                    summarized_messages = self.context_manager.get_summarized_messages()
                    if summarized_messages != self.context_manager.default_agent_messages:
                        # Context manager applied summarization, use the summarized version
                        actual_messages_for_llm = summarized_messages
                        agent_console.print(f"🧠 Using summarized messages: {len(summarized_messages)} vs {len(messages)}", style="blue")
                except Exception as e:
                    logger.warning(f"Failed to get summarized messages, using original: {e}")
                    actual_messages_for_llm = messages

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=actual_messages_for_llm,
                tools=tools_schema,
                tool_choice="auto",
            )
            messages = actual_messages_for_llm

            choice = resp.choices[0]
            msg = choice.message
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None) or []
            # assistant message（带 tool_calls 时 content 可能为 None，这没问题）
            assistant_entry: Dict[str, Any] = {"role": "assistant", "content": content}
            agent_console.print(f"Agent in Turn {turn}: {content}", style="green")

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

            push(assistant_entry)

            # 1) 没有 tool_calls，进入下一轮，因为结束也得tool_calss
            if not tool_calls:
                continue

            # 2) 有 tool_calls：逐个执行工具
            for tc in tool_calls:
                fn_name = tc.function.name
                args_raw = tc.function.arguments or "{}"
                try:
                    tool_name, function_name = self._route_function(fn_name)
                except Exception as e:
                    tool_result = {"success": False, "error": f"Bad tool names: {e}", "raw": fn_name}

                # 解析参数（失败也要回填 tool 消息）
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object/dict")
                except Exception as e:
                    tool_result = {"success": False, "error": f"Bad tool arguments: {e}", "raw": args_raw}
                else:
                    tool_result = self.tool_registry.call_function(
                        tool_name=tool_name,
                        function_name=function_name,
                        parameters=args,
                    )

                # tool message：为了贴合你给的示例，加上 name 字段（可选但推荐）
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,  # 例如 "write_file" / "finish_task"
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                push(tool_msg)

                # Monitor initialization completion
                self._check_initialization_completion(function_name, tool_result)

                # Update context after tool call
                self.update_context_after_tool_call(tool_name, function_name, tool_result, args)

                # 3) 如果调用 finish_task：提前退出（推荐只用它作为"完成信号"）
                if function_name == "base_tools__finish_task":
                    # Record task completion with context manager if available
                    if self.context_manager:
                        self.context_manager.record_memory(
                            "task_completed",
                            content or "Task finished by DefaultAgent"
                        )
                    return {"success": True, "final": content, "messages": messages}

        return {
            "success": False,
            "error": f"Reached max_turns={self.max_turns} without finish_task or final response.",
            "messages": messages,
        }