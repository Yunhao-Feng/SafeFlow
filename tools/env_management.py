"""
Environment Management Tools - Bash Execution and System Operations

Provides bash command execution capabilities with proper error handling,
timeout management, and interaction detection.
"""

import subprocess
import os
import sys
import signal
import threading
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from .abs_tools import Tool, ToolCategory, tool_function, ToolParameter


class EnvManagementTool(Tool):
    """
    Environment management and bash execution tool.

    Provides safe bash command execution with timeout handling,
    error capture, and interaction detection.
    """

    def __init__(
        self,
        item_id: str,
        name: str = "env_management",
        description: str = "Environment management and bash command execution"
    ):
        super().__init__(
            name=name,
            description=description,
            category=ToolCategory.ENV_MANAGEMENT
        )
        self.item_id = item_id
        self.default_timeout = 30  # Default 30 second timeout
        self.max_timeout = 300     # Maximum 5 minute timeout

    @tool_function(
        description="Execute bash commands with timeout and error handling. Supports most shell operations like installing packages, compiling, running scripts, file operations, etc.",
        parameters=[
            ToolParameter("command", "string", "The bash command to execute", required=True),
            ToolParameter("working_dir", "string", "Working directory for command execution (absolute path)", required=False),
            ToolParameter("timeout_seconds", "integer", "Timeout in seconds (default: 30, max: 300)", required=False, default=30),
            ToolParameter("capture_output", "boolean", "Whether to capture stdout/stderr (default: true)", required=False, default=True),
            ToolParameter("shell_env", "object", "Additional environment variables as key-value pairs", required=False),
        ],
        returns="Command execution result with output, error, and status",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__execute_bash(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 30,
        capture_output: bool = True,
        shell_env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Execute bash command with comprehensive error handling."""

        try:
            # Validate and sanitize inputs
            if not command or not isinstance(command, str):
                return {"success": False, "error": "Command cannot be empty"}

            # Validate timeout
            timeout_seconds = max(1, min(timeout_seconds, self.max_timeout))

            # Setup working directory
            if working_dir:
                work_path = Path(working_dir).resolve()
                if not work_path.exists():
                    return {"success": False, "error": f"Working directory does not exist: {working_dir}"}
                cwd = str(work_path)
            else:
                cwd = os.getcwd()

            # Setup environment
            env = os.environ.copy()
            if shell_env:
                env.update(shell_env)

            # Log command execution
            exec_info = {
                "command": command,
                "working_dir": cwd,
                "timeout": timeout_seconds,
                "pid": None
            }

            # Execute command
            if capture_output:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )

                stdout = result.stdout.strip() if result.stdout else ""
                stderr = result.stderr.strip() if result.stderr else ""
                return_code = result.returncode

            else:
                # For commands that need to show real-time output
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                exec_info["pid"] = process.pid

                try:
                    stdout, _ = process.communicate(timeout=timeout_seconds)
                    return_code = process.returncode
                    stderr = ""
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, _ = process.communicate()
                    return {
                        "success": False,
                        "error": f"Command timed out after {timeout_seconds} seconds",
                        "stdout": stdout.strip() if stdout else "",
                        "stderr": "Timeout",
                        "return_code": -1,
                        "execution_info": exec_info
                    }

            # Determine success/failure
            success = return_code == 0

            # Prepare result
            result_data = {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "execution_info": exec_info
            }

            # Add error message if command failed
            if not success:
                if stderr:
                    result_data["error"] = f"Command failed (exit code {return_code}): {stderr}"
                else:
                    result_data["error"] = f"Command failed with exit code {return_code}"

            return result_data

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout_seconds} seconds",
                "stdout": "",
                "stderr": "Timeout",
                "return_code": -1,
                "execution_info": exec_info
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Command failed: {e}",
                "stdout": e.stdout.strip() if e.stdout else "",
                "stderr": e.stderr.strip() if e.stderr else "",
                "return_code": e.returncode,
                "execution_info": exec_info
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "execution_info": exec_info
            }

    @tool_function(
        description="Check if a command exists and is available in the system PATH",
        parameters=[
            ToolParameter("command_name", "string", "Name of the command to check (e.g., 'python', 'git', 'npm')", required=True),
        ],
        returns="Command availability status and path information",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__check_command(self, command_name: str) -> Dict[str, Any]:
        """Check if a command is available in the system."""

        try:
            # Use 'which' or 'where' depending on the system
            if os.name == 'nt':  # Windows
                check_cmd = f"where {command_name}"
            else:  # Unix-like
                check_cmd = f"which {command_name}"

            result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                command_path = result.stdout.strip()
                return {
                    "success": True,
                    "result": {
                        "available": True,
                        "command_name": command_name,
                        "path": command_path,
                        "message": f"Command '{command_name}' is available at: {command_path}"
                    }
                }
            else:
                return {
                    "success": True,
                    "result": {
                        "available": False,
                        "command_name": command_name,
                        "path": None,
                        "message": f"Command '{command_name}' is not available in PATH"
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to check command availability: {str(e)}"
            }

    @tool_function(
        description="Get current environment information including PATH, working directory, and system details",
        parameters=[
            ToolParameter("include_path", "boolean", "Include PATH environment variable details", required=False, default=True),
            ToolParameter("include_env_vars", "array", "List of specific environment variables to include", required=False),
        ],
        returns="Current environment information",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__get_env_info(
        self,
        include_path: bool = True,
        include_env_vars: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get current environment information."""

        try:
            env_info = {
                "current_working_directory": os.getcwd(),
                "system_platform": sys.platform,
                "python_executable": sys.executable,
                "python_version": sys.version,
            }

            if include_path:
                env_info["path"] = os.environ.get("PATH", "").split(os.pathsep)

            if include_env_vars:
                env_info["custom_env_vars"] = {}
                for var in include_env_vars:
                    env_info["custom_env_vars"][var] = os.environ.get(var)

            return {
                "success": True,
                "result": env_info
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get environment info: {str(e)}"
            }

    @tool_function(
        description="Execute bash command with automatic interaction detection and handling",
        parameters=[
            ToolParameter("command", "string", "The bash command to execute", required=True),
            ToolParameter("working_dir", "string", "Working directory for command execution", required=False),
            ToolParameter("timeout_seconds", "integer", "Timeout in seconds (default: 60 for interactive commands)", required=False, default=60),
            ToolParameter("auto_confirm", "boolean", "Automatically confirm 'yes/no' prompts with 'yes'", required=False, default=False),
            ToolParameter("expected_prompts", "object", "Dictionary of expected prompts and their responses", required=False),
        ],
        returns="Command execution result with interaction handling",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__execute_interactive(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 60,
        auto_confirm: bool = False,
        expected_prompts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Execute bash command with basic interaction handling."""

        try:
            # For now, we'll use a simple approach - run with non-interactive flags where possible
            # and provide clear feedback about interaction requirements

            # Common non-interactive flags for popular commands
            interactive_fixes = {
                "apt": " -y",
                "apt-get": " -y",
                "yum": " -y",
                "pip": " --quiet",
                "npm": " --silent",
                "conda": " --yes"
            }

            modified_command = command
            for cmd_prefix, flag in interactive_fixes.items():
                if command.strip().startswith(cmd_prefix) and flag not in command:
                    if auto_confirm:
                        modified_command = command + flag
                        break

            # Setup working directory
            cwd = working_dir if working_dir else os.getcwd()

            # Execute with timeout
            result = subprocess.run(
                modified_command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                input="" if auto_confirm else None  # Provide empty input to avoid hanging
            )

            success = result.returncode == 0

            result_data = {
                "success": success,
                "stdout": result.stdout.strip() if result.stdout else "",
                "stderr": result.stderr.strip() if result.stderr else "",
                "return_code": result.returncode,
                "original_command": command,
                "executed_command": modified_command,
                "interaction_handling": {
                    "auto_confirm": auto_confirm,
                    "command_modified": modified_command != command
                }
            }

            if not success:
                # Check for common interaction indicators in stderr
                stderr_lower = result.stderr.lower() if result.stderr else ""
                if any(indicator in stderr_lower for indicator in ["password", "confirm", "y/n", "[y/n]"]):
                    result_data["error"] = f"Command requires interaction (exit code {result.returncode}). Consider using auto_confirm=true or providing expected_prompts."
                    result_data["interaction_detected"] = True
                else:
                    result_data["error"] = f"Command failed (exit code {result.returncode}): {result.stderr}"
                    result_data["interaction_detected"] = False

            return result_data

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout_seconds} seconds - likely waiting for input",
                "interaction_detected": True,
                "suggestion": "Command may require interaction. Try using auto_confirm=true or check if the command supports non-interactive flags."
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}"
            }

    @tool_function(
        description="Run code quality checks using flake8, pylint, black, isort, or mypy",
        parameters=[
            ToolParameter("tool", "string", "Quality tool to run", required=True,
                         enum_values=["flake8", "pylint", "black", "isort", "mypy", "bandit"]),
            ToolParameter("target_path", "string", "File or directory path to check", required=True),
            ToolParameter("fix_mode", "boolean", "Apply automatic fixes where possible (black, isort)", required=False, default=False),
            ToolParameter("config_file", "string", "Path to config file (optional)", required=False),
            ToolParameter("extra_args", "string", "Additional command line arguments", required=False),
        ],
        returns="Code quality analysis results with suggestions",
        category=ToolCategory.ENV_MANAGEMENT,
    )
    def env_management__code_quality_check(
        self,
        tool: str,
        target_path: str,
        fix_mode: bool = False,
        config_file: Optional[str] = None,
        extra_args: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run code quality checks and provide improvement suggestions."""
        try:
            # Validate target path
            target = Path(target_path)
            if not target.exists():
                return {"success": False, "error": f"Target path does not exist: {target_path}"}

            # Build command based on tool
            commands = {
                "flake8": self._build_flake8_command,
                "pylint": self._build_pylint_command,
                "black": self._build_black_command,
                "isort": self._build_isort_command,
                "mypy": self._build_mypy_command,
                "bandit": self._build_bandit_command
            }

            if tool not in commands:
                return {"success": False, "error": f"Unsupported tool: {tool}"}

            # Check if tool is available
            check_result = self.env_management__check_command(tool)
            if not check_result["result"]["available"]:
                return {
                    "success": False,
                    "error": f"Tool '{tool}' is not installed. Install with: pip install {tool}",
                    "suggestion": f"Run: pip install {tool}"
                }

            # Build and execute command
            command = commands[tool](target_path, fix_mode, config_file, extra_args)
            result = self.env_management__execute_bash(command, timeout_seconds=300)

            # Parse and format results
            analysis = self._parse_quality_results(tool, result, target_path)

            return {
                "success": True,
                "result": {
                    "tool": tool,
                    "target_path": target_path,
                    "command_executed": command,
                    "raw_output": result,
                    "analysis": analysis
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Code quality check failed: {str(e)}"
            }

    def _build_flake8_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        """Build flake8 command."""
        cmd = f"flake8 {target_path}"
        if config_file:
            cmd += f" --config={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_pylint_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        """Build pylint command."""
        cmd = f"pylint {target_path}"
        if config_file:
            cmd += f" --rcfile={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        else:
            cmd += " --score=yes --reports=yes"
        return cmd

    def _build_black_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        """Build black command."""
        cmd = f"black"
        if not fix_mode:
            cmd += " --check --diff"
        cmd += f" {target_path}"
        if config_file:
            cmd += f" --config={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_isort_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        """Build isort command."""
        cmd = f"isort"
        if not fix_mode:
            cmd += " --check-only --diff"
        cmd += f" {target_path}"
        if config_file:
            cmd += f" --settings-path={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_mypy_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        """Build mypy command."""
        cmd = f"mypy {target_path}"
        if config_file:
            cmd += f" --config-file={config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        return cmd

    def _build_bandit_command(self, target_path: str, fix_mode: bool, config_file: Optional[str], extra_args: Optional[str]) -> str:
        """Build bandit command (security analysis)."""
        cmd = f"bandit"
        if Path(target_path).is_dir():
            cmd += " -r"
        cmd += f" {target_path}"
        if config_file:
            cmd += f" -c {config_file}"
        if extra_args:
            cmd += f" {extra_args}"
        else:
            cmd += " -f json"  # JSON output for better parsing
        return cmd

    def _parse_quality_results(self, tool: str, result: Dict[str, Any], target_path: str) -> Dict[str, Any]:
        """Parse code quality tool results and provide structured analysis."""
        analysis = {
            "tool": tool,
            "success": result["success"],
            "issues": [],
            "summary": {},
            "suggestions": []
        }

        if not result["success"] and result["return_code"] == 0:
            # Some tools return non-zero for style issues, not actual failures
            result["success"] = True

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if tool == "flake8":
            analysis.update(self._parse_flake8_output(stdout, stderr))
        elif tool == "pylint":
            analysis.update(self._parse_pylint_output(stdout, stderr))
        elif tool == "black":
            analysis.update(self._parse_black_output(stdout, stderr))
        elif tool == "isort":
            analysis.update(self._parse_isort_output(stdout, stderr))
        elif tool == "mypy":
            analysis.update(self._parse_mypy_output(stdout, stderr))
        elif tool == "bandit":
            analysis.update(self._parse_bandit_output(stdout, stderr))

        return analysis

    def _parse_flake8_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse flake8 output."""
        issues = []
        for line in stdout.splitlines():
            if ":" in line:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    issues.append({
                        "file": parts[0],
                        "line": parts[1],
                        "column": parts[2],
                        "message": parts[3].strip(),
                        "type": "style"
                    })

        return {
            "issues": issues,
            "summary": {"total_issues": len(issues)},
            "suggestions": ["Fix style issues to improve code readability"] if issues else ["Code style looks good!"]
        }

    def _parse_pylint_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pylint output."""
        issues = []
        score = None

        lines = stdout.splitlines()
        for line in lines:
            if line.startswith("Your code has been rated at"):
                score_match = re.search(r"rated at ([\d.-]+)/10", line)
                if score_match:
                    score = float(score_match.group(1))

            if ":" in line and any(marker in line for marker in ["C:", "R:", "W:", "E:", "F:"]):
                parts = line.split(":", 4)
                if len(parts) >= 5:
                    issues.append({
                        "file": parts[0],
                        "line": parts[1],
                        "type": parts[3].strip(),
                        "message": parts[4].strip()
                    })

        suggestions = []
        if score is not None:
            if score < 7.0:
                suggestions.append(f"Code quality score is {score}/10. Consider addressing pylint warnings.")
            else:
                suggestions.append(f"Good code quality score: {score}/10")

        return {
            "issues": issues,
            "summary": {"total_issues": len(issues), "score": score},
            "suggestions": suggestions
        }

    def _parse_black_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse black output."""
        if "would reformat" in stdout or "reformatted" in stdout:
            return {
                "issues": [{"type": "formatting", "message": "Code needs formatting"}],
                "summary": {"needs_formatting": True},
                "suggestions": ["Run black with fix_mode=true to format code automatically"]
            }
        return {
            "issues": [],
            "summary": {"needs_formatting": False},
            "suggestions": ["Code formatting looks good!"]
        }

    def _parse_isort_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse isort output."""
        if "would reorder" in stdout or "Fixing" in stdout:
            return {
                "issues": [{"type": "import_order", "message": "Import statements need reordering"}],
                "summary": {"needs_import_sorting": True},
                "suggestions": ["Run isort with fix_mode=true to sort imports automatically"]
            }
        return {
            "issues": [],
            "summary": {"needs_import_sorting": False},
            "suggestions": ["Import order looks good!"]
        }

    def _parse_mypy_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse mypy output."""
        issues = []
        for line in stdout.splitlines():
            if ": error:" in line or ": warning:" in line:
                parts = line.split(":", 3)
                if len(parts) >= 3:
                    issues.append({
                        "file": parts[0],
                        "line": parts[1],
                        "type": "type_error" if "error" in line else "type_warning",
                        "message": parts[2].strip()
                    })

        return {
            "issues": issues,
            "summary": {"type_errors": len([i for i in issues if i["type"] == "type_error"])},
            "suggestions": ["Add type annotations to improve code safety"] if issues else ["Type checking passed!"]
        }

    def _parse_bandit_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse bandit output."""
        issues = []
        try:
            if stdout.startswith("{"):
                data = json.loads(stdout)
                for result in data.get("results", []):
                    issues.append({
                        "file": result.get("filename", ""),
                        "line": str(result.get("line_number", "")),
                        "type": "security",
                        "severity": result.get("issue_severity", ""),
                        "confidence": result.get("issue_confidence", ""),
                        "message": result.get("issue_text", "")
                    })
        except json.JSONDecodeError:
            # Fall back to text parsing
            for line in stdout.splitlines():
                if ">> Issue:" in line:
                    issues.append({
                        "type": "security",
                        "message": line.strip()
                    })

        return {
            "issues": issues,
            "summary": {"security_issues": len(issues)},
            "suggestions": ["Review security issues carefully"] if issues else ["No security issues detected!"]
        }