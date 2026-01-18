import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.abs_tools import Tool, ToolCategory, ToolParameter, tool_function

def _ensure_under(base: Path, target: Path) -> Path:
    base = base.resolve()
    target = target.resolve()
    try:
        target.relative_to(base)
    except Exception:
        raise PermissionError(f"Path escapes base directory: base={base}, target={target}")
    return target

def _summarize_key_files(
    root: Path,
    max_chars_each: int = 2000,
    max_entries: int = 20,
) -> Dict[str, str]:
    """
    Summarize a small set of "likely important" top-level files.
    - Case-insensitive matching (README.md/readme.md etc.)
    - Not limited to a fixed name list: uses heuristics
    - max_entries limits how many files we read
    """
    root = Path(root)
    result: Dict[str, str] = {}
    if not root.exists():
        return result

    # Heuristics: keywords and config-y filenames
    keyword_names = {
        "readme", "changelog", "changes", "license", "copying",
        "contributing", "contributors", "code_of_conduct", "security",
        "install", "usage", "getting_started",
    }
    exact_names = {
        "pyproject.toml", "setup.cfg", "setup.py", "requirements.txt",
        "pipfile", "pipfile.lock", "poetry.lock",
        "pytest.ini", "tox.ini", "noxfile.py",
        "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ".gitignore", ".gitattributes",
    }
    config_exts = {".toml", ".cfg", ".ini", ".yml", ".yaml", ".json", ".txt", ".md", ".rst"}

    def score(p: Path) -> int:
        # higher = more important
        name = p.name
        lower = name.lower()
        stem_lower = p.stem.lower()

        s = 0
        if lower in exact_names:
            s += 100
        if stem_lower in keyword_names:
            s += 80
        if lower.startswith("readme"):
            s += 90
        if lower.startswith("license") or lower.startswith("copying"):
            s += 70
        if p.suffix.lower() in config_exts:
            s += 10
        # prefer shorter names and top-level (we only scan top-level anyway)
        s -= len(name) // 20
        return s

    # Only scan one layer (top-level)
    files: List[Path] = [p for p in root.iterdir() if p.is_file()]

    # Filter: likely-text / config-ish files
    candidates: List[Path] = []
    for p in files:
        lower = p.name.lower()
        if lower in exact_names:
            candidates.append(p)
            continue
        if p.suffix.lower() in config_exts and (p.stem.lower() in keyword_names or lower.startswith(("readme", "license", "changelog", "contributing"))):
            candidates.append(p)
            continue
        # also allow common project config files by extension (but not everything)
        if p.suffix.lower() in {".toml", ".cfg", ".ini", ".yml", ".yaml", ".json"}:
            candidates.append(p)

    # Rank and take max_entries
    ranked: List[Tuple[int, Path]] = sorted(((score(p), p) for p in candidates), key=lambda x: x[0], reverse=True)
    chosen = [p for _, p in ranked[:max_entries]]

    for p in chosen:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            result[p.name] = _truncate(text, max_chars_each)
        except Exception:
            continue

    return result

def _truncate(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n...[truncated {len(s) - max_chars} chars]"


def _list_one_layer(dir_path: Path, include_hidden: bool, max_entries: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not dir_path.exists():
        return out
    if not dir_path.is_dir():
        return out

    i = 0
    for p in sorted(dir_path.iterdir(), key=lambda x: x.name):
        if i >= max_entries:
            break
        if not include_hidden and p.name.startswith("."):
            continue
        out.append(
            {
                "name": p.name,
                "type": "directory" if p.is_dir() else "file",
                "absolute_path": str(p.resolve()),
            }
        )
        i += 1
    return out



class BaseTools(Tool):
    """
    Minimal, non-overlapping orchestration tools
    """

    # Class-level initialization tracking (shared across instances with same item_id)
    _initialized_sessions: Dict[str, bool] = {}
    _session_work_roots: Dict[str, str] = {}

    def __init__(
        self,
        item_id: str,
        name: str = "base_tools",
        description: str = "Initialization/workspace/git orchestration tools",
        command_timeout_sec: int = 300,
        max_cmd_output_chars: int = 200_000,
    ):
        super().__init__(name=name, description=description, category=ToolCategory.BASE_TOOLS)
        self.item_id = item_id
        self.command_timeout_sec = command_timeout_sec
        self.max_cmd_output_chars = max_cmd_output_chars

        self._task_blob_preview: Optional[str] = None
        self.work_root: Optional[Path] = None

    def _is_session_initialized(self) -> bool:
        """Check if current session (item_id) is already initialized."""
        return self._initialized_sessions.get(self.item_id, False)

    def _mark_session_initialized(self, work_root: str = None):
        """Mark current session as initialized."""
        self._initialized_sessions[self.item_id] = True
        if work_root:
            self._session_work_roots[self.item_id] = work_root

    def _get_session_work_root(self) -> Optional[str]:
        """Get work root for current session if exists."""
        return self._session_work_roots.get(self.item_id)
    

    def base_tools__initialize_context(
        self,
        task_blob: str,
        preview_chars: int = 4000,
        include_hidden: bool = False,
        max_entries: int = 50,
    ) -> Dict[str, Any]:

        cwd = Path.cwd().resolve()
        self._task_blob_preview = _truncate(task_blob, int(preview_chars))

        return {
            "success": True,
            "result": {
                "item_id": self.item_id,
                "cwd": str(cwd),
                "cwd_entries": _list_one_layer(cwd, include_hidden=include_hidden, max_entries=int(max_entries)),
                "work_root": str(self.work_root) if self.work_root else None,
                "task_blob_preview": self._task_blob_preview,
                "capabilities": {
                    "git_available": shutil.which("git") is not None,
                },
            },
        }
    
    @tool_function(
        description="Set the working root directory for subsequent operations. Path is resolved relative to cwd if not absolute.",
        parameters=[
            ToolParameter("path", "string", "Work root path, e.g. '.', './work', or absolute path.", required=True),
            ToolParameter("create", "boolean", "Create directory if it does not exist.", required=False, default=True),
            ToolParameter("must_be_dir", "boolean", "If true, reject if path is not a directory.", required=False, default=True),
            ToolParameter("include_hidden", "boolean", "Include hidden entries in listing.", required=False, default=False),
            ToolParameter("max_entries", "integer", "Max entries in work_root listing (one layer).", required=False, default=50),
        ],
        returns="Selected work_root and its one-layer listing.",
        category=ToolCategory.BASE_TOOLS,
    )

    def base_tools__set_work_root(
        self,
        path: str,
        create: bool = False,
        must_be_dir: bool = True,
        include_hidden: bool = False,
        max_entries: int = 50,
    ) -> Dict[str, Any]:
        try:
            p = Path(path)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            else:
                p = p.resolve()
            
            if p.exists():
                if must_be_dir and not p.is_dir():
                    return {"success": False, "error": f"work_root is not a directory: {p}"}
            else:
                if not create:
                    return {"success": False, "error": f"work_root does not exist: {p}. Set create=true to create it."}
                p.mkdir(parents=True, exist_ok=True)
            
            self.work_root = p

            # Mark session as initialized when work_root is set
            self._mark_session_initialized(str(p))

            return {
                "success": True,
                "result": {
                    "work_root": str(self.work_root),
                    "work_root_entries": _list_one_layer(self.work_root, include_hidden=include_hidden, max_entries=int(max_entries)),
                    "session_initialized": True,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    

    @tool_function(
        description="Summarize a few key files under work_root (README/pyproject/setup.cfg/package.json etc).",
        parameters=[
            ToolParameter("max_chars_each", "integer", "Max chars to return per file.", required=False, default=2000),
            ToolParameter("max_files", "integer", "Max entries of files in work_root you want to summary.", required=False, default=20)
        ],
        returns="Dict of filename -> head text.",
        category=ToolCategory.BASE_TOOLS,
    )
    def base_tools__summarize_work_root(self, max_chars_each: int = 2000, max_files: int = 20) -> Dict[str, Any]:
        if self.work_root is None:
            return {"success": False, "error": "work_root not set. Call base_tools__set_work_root first."}
        try:
            return {"success": True, "result": {"summaries": _summarize_key_files(root=self.work_root, max_chars_each=int(max_chars_each), max_entries=max_files)}}
        except Exception as e:
            return {"success": False, "error": str(e)}


    @tool_function(
        description="Controlled git clone into work_root/dest_subdir. Only https and allowlisted hosts.",
        parameters=[
            ToolParameter("repo", "string", "owner/repo or https://github.com/owner/repo(.git)", required=True),
            ToolParameter("dest_subdir", "string", "Destination subdir under work_root.", required=False, default="."),
            ToolParameter("checkout", "string", "Optional revision/commit to checkout.", required=False, default=""),
            ToolParameter("depth", "integer", "Shallow clone depth (0=full).", required=False, default=0),

            # new knobs
            ToolParameter("recurse_submodules", "boolean", "Clone submodules.", required=False, default=False),
            ToolParameter("shallow_submodules", "boolean", "If recurse_submodules, shallow submodules.", required=False, default=True),
            ToolParameter("partial_clone", "boolean", "Use partial clone filter=blob:none when depth>0.", required=False, default=True),
            ToolParameter("fetch_unshallow_on_demand", "boolean", "If checkout missing, deepen/unshallow automatically.", required=False, default=True),
        ],
        returns="Clone result.",
        category=ToolCategory.BASE_TOOLS,
    )
    def base_tools__git_clone(
        self,
        repo: str,
        dest_subdir: str = "repo",
        checkout: str = "",
        depth: int = 0,
        recurse_submodules: bool = False,
        shallow_submodules: bool = True,
        partial_clone: bool = True,
        fetch_unshallow_on_demand: bool = True,
    ) -> Dict[str, Any]:
        if self.work_root is None:
            return {"success": False, "error": "work_root not set. Call base_tools__set_work_root first."}
        if shutil.which("git") is None:
            return {"success": False, "error": "git not found on PATH."}

        repo = repo.strip()
        if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo):
            repo_url = f"https://github.com/{repo}.git"
        else:
            repo_url = repo

        m = re.match(r"^https://([^/]+)/", repo_url)
        if not m:
            return {"success": False, "error": f"Only https clone allowed. Got: {repo_url}"}
        host = m.group(1).lower()

        dest = _ensure_under(self.work_root, (self.work_root / (dest_subdir or "repo")))
        if dest.exists() and any(dest.iterdir()):
            return {"success": False, "error": f"Destination not empty: {dest}"}
        dest.mkdir(parents=True, exist_ok=True)

        cmd = ["git", "clone"]
        if recurse_submodules:
            cmd.append("--recurse-submodules")
            if shallow_submodules and depth and depth > 0:
                cmd.append("--shallow-submodules")

        # shallow clone knobs
        if depth and depth > 0:
            cmd += ["--depth", str(depth)]
            if partial_clone:
                # reduces blob transfer for big repos; checkout may later need blobs -> fetched on demand by git
                cmd += ["--filter=blob:none"]

        cmd += [repo_url, str(dest)]

        def _run(cmd_, cwd, timeout):
            cp = subprocess.run(
                cmd_,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return cp, _truncate(cp.stdout, self.max_cmd_output_chars), _truncate(cp.stderr, self.max_cmd_output_chars)

        def _has_object(rev: str) -> bool:
            # works for commits/tags/branches if present locally
            cp, _, _ = _run(["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=dest, timeout=30)
            return cp.returncode == 0

        try:
            cp, stdout, stderr = _run(cmd, cwd=self.work_root, timeout=self.command_timeout_sec)
            if cp.returncode != 0:
                return {"success": False, "error": f"git clone failed ({cp.returncode})", "meta": {"stdout": stdout, "stderr": stderr}}

            if checkout:
                # If shallow clone doesn't contain the target commit, fetch it (or unshallow).
                if not _has_object(checkout):
                    if not fetch_unshallow_on_demand:
                        return {"success": False, "error": f"checkout target not present in shallow clone: {checkout}"}

                    # 1) try fetching just that commit (often works and is efficient)
                    cp_f, so_f, se_f = _run(["git", "fetch", "origin", checkout], cwd=dest, timeout=self.command_timeout_sec)

                    # 2) if still missing, deepen/unshallow (more compatible)
                    if cp_f.returncode != 0 or not _has_object(checkout):
                        if depth and depth > 0:
                            # unshallow gives full history
                            cp_u, so_u, se_u = _run(["git", "fetch", "--unshallow", "origin"], cwd=dest, timeout=self.command_timeout_sec)
                        else:
                            cp_u, so_u, se_u = _run(["git", "fetch", "--all", "--tags"], cwd=dest, timeout=self.command_timeout_sec)

                        # if unshallow failed, propagate useful logs
                        if not _has_object(checkout):
                            return {
                                "success": False,
                                "error": f"Unable to fetch checkout target: {checkout}",
                                "meta": {
                                    "clone_stdout": stdout, "clone_stderr": stderr,
                                    "fetch_stdout": so_f, "fetch_stderr": se_f,
                                    "deepen_stdout": so_u if 'so_u' in locals() else "",
                                    "deepen_stderr": se_u if 'se_u' in locals() else "",
                                }
                            }

                # detach is safer for commit SHA
                cp2, so2, se2 = _run(["git", "checkout", "--detach", checkout], cwd=dest, timeout=self.command_timeout_sec)
                if cp2.returncode != 0:
                    return {"success": False, "error": f"git checkout failed ({cp2.returncode})", "meta": {"stdout": so2, "stderr": se2}}

            # If we want submodules AND we checked out an older commit, ensure submodules match that commit.
            if recurse_submodules:
                cp_sm, so_sm, se_sm = _run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=dest,
                    timeout=self.command_timeout_sec,
                )
                if cp_sm.returncode != 0:
                    return {"success": False, "error": f"git submodule update failed ({cp_sm.returncode})", "meta": {"stdout": so_sm, "stderr": se_sm}}

            cp3, _, _ = _run(["git", "rev-parse", "HEAD"], cwd=dest, timeout=10)
            head = cp3.stdout.strip() if cp3.returncode == 0 else None

            return {
                "success": True,
                "result": {
                    "repo_url": repo_url,
                    "dest": str(dest),
                    "head": head,
                    "checked_out": checkout or None,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "git clone timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def base_tools__analyze_task_type(self, task_blob: str) -> Dict[str, Any]:
        """Detect SWE vs regular task and recommend workspace setup."""
        try:
            import json
            import re

            # Try to parse as JSON (SWE-bench format)
            swe_analysis = {"is_swe_task": False, "confidence": "high"}

            try:
                task_data = json.loads(task_blob)

                # Check for SWE-bench indicators
                swe_indicators = [
                    "instance_id", "repo", "base_commit", "patch",
                    "test_patch", "problem_statement", "hints_text"
                ]

                found_indicators = [key for key in swe_indicators if key in task_data]

                if len(found_indicators) >= 3:  # Strong SWE indication
                    swe_analysis = {
                        "is_swe_task": True,
                        "confidence": "high",
                        "repo": task_data.get("repo"),
                        "instance_id": task_data.get("instance_id"),
                        "base_commit": task_data.get("base_commit"),
                        "found_indicators": found_indicators
                    }

            except json.JSONDecodeError:
                # Check for SWE patterns in text
                swe_patterns = [
                    r"instance_id[:\s]+\w+",
                    r"repo[:\s]+[\w\-_/]+",
                    r"base_commit[:\s]+[a-f0-9]{7,}",
                    r"problem_statement[:\s]",
                ]

                found_patterns = []
                for pattern in swe_patterns:
                    if re.search(pattern, task_blob, re.IGNORECASE):
                        found_patterns.append(pattern)

                if len(found_patterns) >= 2:
                    swe_analysis = {
                        "is_swe_task": True,
                        "confidence": "medium",
                        "found_patterns": found_patterns
                    }

            # Generate workspace recommendations
            cwd = Path.cwd().resolve()
            cwd_entries = list(cwd.iterdir()) if cwd.exists() else []
            cwd_empty = len(cwd_entries) == 0

            if swe_analysis["is_swe_task"]:
                if cwd_empty:
                    recommendation = {
                        "strategy": "clone_to_cwd",
                        "work_root": str(cwd),
                        "reason": "Empty cwd, can clone SWE repo directly"
                    }
                else:
                    recommendation = {
                        "strategy": "clone_to_subdir",
                        "work_root": str(cwd / "swe_workspace"),
                        "reason": "Non-empty cwd, clone to subdirectory"
                    }
                recommendation["requires_clone"] = True
                recommendation["repo_info"] = {
                    "repo": swe_analysis.get("repo"),
                    "base_commit": swe_analysis.get("base_commit")
                }
            else:
                # Regular task
                has_code = any(p.suffix in {'.py', '.js', '.java', '.cpp', '.go'}
                             for p in cwd_entries if p.is_file())

                if has_code:
                    recommendation = {
                        "strategy": "use_existing_cwd",
                        "work_root": str(cwd),
                        "reason": "Existing code project in cwd"
                    }
                elif cwd_empty:
                    recommendation = {
                        "strategy": "use_cwd_as_workspace",
                        "work_root": str(cwd),
                        "reason": "Empty cwd, use as workspace"
                    }
                else:
                    recommendation = {
                        "strategy": "create_subdir",
                        "work_root": str(cwd / "workspace"),
                        "reason": "Non-empty cwd, create workspace subdir"
                    }
                recommendation["requires_clone"] = False

            return {
                "success": True,
                "result": {
                    "task_analysis": swe_analysis,
                    "current_cwd": {
                        "path": str(cwd),
                        "empty": cwd_empty,
                        "entry_count": len(cwd_entries)
                    },
                    "recommendation": recommendation
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Task analysis failed: {e}"}


    @tool_function(
        description="Complete initialization workflow: analyze task, set workspace, clone repo if needed, summarize.",
        parameters=[
            ToolParameter("task_blob", "string", "Raw task description or SWE-bench JSON", required=True),
            ToolParameter("force_reinit", "boolean", "Force re-initialization even if already initialized", required=False, default=False),
        ],
        returns="Complete initialization result",
        category=ToolCategory.BASE_TOOLS,
    )
    def base_tools__complete_initialization(
        self,
        task_blob: str,
        force_reinit: bool = False
    ) -> Dict[str, Any]:
        """Execute complete initialization workflow."""
        try:
            # Step 1: Check if already initialized
            if self._is_session_initialized() and not force_reinit:
                existing_work_root = self._get_session_work_root()
                return {
                    "success": True,
                    "result": {
                        "already_initialized": True,
                        "work_root": existing_work_root,
                        "message": "Session already initialized. Use force_reinit=true to reinitialize."
                    }
                }

            # Step 2: Initialize context (explore current directory)
            context_result = self.base_tools__initialize_context(task_blob)
            if not context_result["success"]:
                return {"success": False, "error": f"Context initialization failed: {context_result['error']}"}

            cwd_info = context_result["result"]

            # Step 3: Analyze task type and get workspace recommendation
            analysis_result = self.base_tools__analyze_task_type(task_blob)
            if not analysis_result["success"]:
                return {"success": False, "error": f"Task analysis failed: {analysis_result['error']}"}

            analysis = analysis_result["result"]
            recommendation = analysis["recommendation"]

            # Step 4: Set work root based on recommendation
            work_root_result = self.base_tools__set_work_root(
                path=recommendation["work_root"],
                create=True
            )
            if not work_root_result["success"]:
                return {"success": False, "error": f"Work root setup failed: {work_root_result['error']}"}

            work_root = work_root_result["result"]["work_root"]

            # Step 5: Clone repository if needed (SWE tasks)
            clone_result = None
            if recommendation.get("requires_clone") and recommendation.get("repo_info", {}).get("repo"):
                repo_info = recommendation["repo_info"]

                # Determine destination subdirectory
                if recommendation["strategy"] == "clone_to_cwd":
                    dest_subdir = "."
                else:
                    dest_subdir = "repo"

                clone_result = self.base_tools__git_clone(
                    repo=repo_info["repo"],
                    dest_subdir=dest_subdir,
                    checkout=repo_info.get("base_commit", "")
                )

                if not clone_result["success"]:
                    return {"success": False, "error": f"Git clone failed: {clone_result['error']}"}

            # Step 6: Summarize workspace (especially important after cloning)
            summary_result = self.base_tools__summarize_work_root()
            workspace_summary = {}
            if summary_result["success"]:
                workspace_summary = summary_result["result"]["summaries"]

            # Compile final result
            result = {
                "initialization_completed": True,
                "task_analysis": analysis,
                "work_root": work_root,
                "workspace_strategy": recommendation["strategy"],
                "cwd_info": cwd_info,
                "workspace_summary": workspace_summary,
                "steps_completed": [
                    "context_initialization",
                    "task_analysis",
                    "workspace_setup"
                ]
            }

            if clone_result:
                result["git_clone"] = clone_result["result"]
                result["steps_completed"].append("repository_clone")

            result["steps_completed"].append("workspace_summary")

            # Explicitly mark session as initialized after all steps complete
            self._mark_session_initialized(work_root)

            return {
                "success": True,
                "result": result
            }

        except Exception as e:
            return {"success": False, "error": f"Complete initialization failed: {e}"}

    @tool_function(
        description="Unified session state management: check, save, or get current work context",
        parameters=[
            ToolParameter("action", "string", "Action: 'check', 'save', 'get_context'", required=True),
            ToolParameter("work_root", "string", "Work root path for saving state", required=False),
            ToolParameter("current_task", "string", "Current task description", required=False),
            ToolParameter("completed_steps", "array", "List of completed steps", required=False),
        ],
        returns="Session state information",
        category=ToolCategory.BASE_TOOLS,
    )
    def base_tools__session_state(
        self,
        action: str,
        work_root: str = None,
        current_task: str = None,
        completed_steps: List[str] = None
    ) -> Dict[str, Any]:
        """Unified session state management with persistent context tracking."""
        try:
            if action == "check":
                # Return comprehensive session status
                initialized = self._is_session_initialized()
                current_work_root = self._get_session_work_root()

                # Try to load additional state from context file if it exists
                context_file = Path.cwd() / f".safeflow_context_{self.item_id}.json"
                additional_context = {}
                if context_file.exists():
                    try:
                        import json
                        with open(context_file, 'r') as f:
                            additional_context = json.load(f)
                    except Exception:
                        pass

                return {
                    "success": True,
                    "result": {
                        "initialized": initialized,
                        "item_id": self.item_id,
                        "work_root": current_work_root,
                        "message": "Already initialized" if initialized else "Not initialized",
                        "last_activity": additional_context.get("last_activity"),
                        "current_task": additional_context.get("current_task"),
                        "completed_steps": additional_context.get("completed_steps", [])
                    }
                }

            elif action == "save":
                # Save current state including work_root and task info
                if work_root:
                    self._mark_session_initialized(work_root)

                # Save extended context to file
                context_file = Path.cwd() / f".safeflow_context_{self.item_id}.json"
                import json
                from datetime import datetime

                context_data = {
                    "item_id": self.item_id,
                    "initialized": True,
                    "work_root": work_root or self._get_session_work_root(),
                    "current_task": current_task,
                    "completed_steps": completed_steps or [],
                    "last_activity": datetime.now().isoformat()
                }

                with open(context_file, 'w') as f:
                    json.dump(context_data, f, indent=2)

                return {
                    "success": True,
                    "result": {
                        "saved": True,
                        "context_file": str(context_file),
                        "work_root": context_data["work_root"]
                    }
                }

            elif action == "get_context":
                # Get full current context for agent awareness
                current_work_root = self._get_session_work_root()
                context_file = Path.cwd() / f".safeflow_context_{self.item_id}.json"

                context_data = {
                    "work_root": current_work_root,
                    "initialized": self._is_session_initialized(),
                    "item_id": self.item_id
                }

                if context_file.exists():
                    try:
                        import json
                        with open(context_file, 'r') as f:
                            saved_context = json.load(f)
                            context_data.update(saved_context)
                    except Exception:
                        pass

                return {
                    "success": True,
                    "result": context_data
                }

            else:
                return {
                    "success": False,
                    "error": f"Invalid action '{action}'. Must be 'check', 'save', or 'get_context'"
                }

        except Exception as e:
            return {"success": False, "error": f"Session state operation failed: {e}"}

    @tool_function(
        description="Verify current environment and dependencies",
        parameters=[
            ToolParameter("requirements_file", "string", "Path to requirements file to check", required=False),
            ToolParameter("check_python", "boolean", "Check Python version and environment", required=False, default=True),
        ],
        returns="Environment verification result",
        category=ToolCategory.BASE_TOOLS,
    )
    def base_tools__verify_environment(
        self,
        requirements_file: str = None,
        check_python: bool = True
    ) -> Dict[str, Any]:
        """Verify current environment and dependencies."""
        import sys
        import importlib.util

        try:
            result = {
                "work_root": self._get_session_work_root(),
                "current_directory": str(Path.cwd()),
                "environment_ok": True,
                "issues": []
            }

            if check_python:
                result["python_version"] = sys.version
                result["python_executable"] = sys.executable

            if requirements_file:
                requirements_path = Path(requirements_file)
                if not requirements_path.exists():
                    result["issues"].append(f"Requirements file not found: {requirements_file}")
                    result["environment_ok"] = False
                else:
                    result["requirements_file"] = str(requirements_path)
                    result["installed_packages"] = []
                    result["missing_packages"] = []

                    # Read requirements
                    try:
                        with open(requirements_path, 'r') as f:
                            requirements = [line.strip() for line in f.readlines()
                                          if line.strip() and not line.startswith('#')]

                        for req in requirements:
                            # Simple package name extraction (ignoring version constraints)
                            package_name = req.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()

                            try:
                                spec = importlib.util.find_spec(package_name)
                                if spec is not None:
                                    result["installed_packages"].append(package_name)
                                else:
                                    result["missing_packages"].append(package_name)
                                    result["issues"].append(f"Missing package: {package_name}")
                            except (ImportError, ModuleNotFoundError):
                                result["missing_packages"].append(package_name)
                                result["issues"].append(f"Missing package: {package_name}")

                        if result["missing_packages"]:
                            result["environment_ok"] = False
                            result["installation_command"] = f"pip install -r {requirements_file}"

                    except Exception as e:
                        result["issues"].append(f"Failed to parse requirements file: {e}")
                        result["environment_ok"] = False

            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Environment verification failed: {e}"}

    @tool_function(
        description="Enhanced task completion with verification",
        parameters=[
            ToolParameter("message", "string", "Completion message", required=True),
            ToolParameter("verify_task", "boolean", "Verify task completion before finishing", required=False, default=True),
            ToolParameter("expected_files", "array", "List of expected files that should exist", required=False),
            ToolParameter("expected_functionality", "string", "Description of functionality that should work", required=False),
        ],
        returns="Task completion result with verification",
        category=ToolCategory.BASE_TOOLS,
    )
    def base_tools__finish_task(
        self,
        message: str,
        verify_task: bool = True,
        expected_files: List[str] = None,
        expected_functionality: str = None
    ) -> Dict[str, Any]:
        """Enhanced task completion with verification."""
        try:
            if not verify_task:
                # Skip verification, finish immediately
                return {
                    "success": True,
                    "result": {
                        "done": True,
                        "message": message,
                        "item_id": self.item_id,
                        "verification_skipped": True
                    }
                }

            # Perform task verification
            verification_result = self._verify_task_completion(expected_files, expected_functionality)

            if not verification_result["completed"]:
                return {
                    "success": False,
                    "error": f"Task not fully completed: {', '.join(verification_result['missing_items'])}",
                    "result": {
                        "verification_failed": True,
                        "missing_items": verification_result["missing_items"],
                        "suggestions": verification_result.get("next_steps", []),
                        "message": "Task verification failed - please complete missing items before finishing"
                    }
                }

            # Task verification passed
            return {
                "success": True,
                "result": {
                    "done": True,
                    "message": message,
                    "item_id": self.item_id,
                    "verification_passed": True,
                    "verified_items": verification_result.get("completed_items", [])
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Task completion failed: {e}"}

    def _verify_task_completion(
        self,
        expected_files: List[str] = None,
        expected_functionality: str = None
    ) -> Dict[str, Any]:
        """Internal method to verify task completion."""
        missing_items = []
        completed_items = []

        # Check expected files exist
        if expected_files:
            for file_path in expected_files:
                full_path = Path(file_path)
                if not full_path.is_absolute():
                    work_root = self._get_session_work_root()
                    if work_root:
                        full_path = Path(work_root) / file_path

                if full_path.exists():
                    completed_items.append(f"File exists: {file_path}")
                else:
                    missing_items.append(f"Missing file: {file_path}")

        # Basic functionality checks could be added here
        # For now, we mainly focus on file existence

        return {
            "completed": len(missing_items) == 0,
            "missing_items": missing_items,
            "completed_items": completed_items,
            "next_steps": [f"Create or fix: {item}" for item in missing_items]
        }
