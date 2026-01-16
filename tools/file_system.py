"""
此处应该实现文件系统的目录查看，内容查看，内容编辑，文件的搜索，文件内容的检索
"""

"""
File System Tool for SafeFlow.
This tool provides file system operations that agents can use to read,
write, and navigate files during sessions.
"""

import glob
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from tools.base_tools import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(name=__name__)

class FileSystemTool(Tool):
    """
    File System Tool for SafeFlow.
    This tool provides file system operations that agents can use to read,
    write, and navigate files during sessions.
    """
    def __init__(self, 
        name: str = "file_system",
        description: str = "Tool for reading, editing, and navigating files and directories",
        max_file_size: int = 10, # 10 MB
        max_content_size = 8_000_000, # max chars of a file to read
        read_only: bool = False,
        ):
        super().__init__(
            name=name,
            description=description,
            category=ToolCategory.FILE_SYSTEM
        )

        self.max_file_size = max_file_size * 1024 * 1024
        self.read_only = read_only
        self.max_content_size = max_content_size

    
    @tool_function(
        description="Read the contents of a readable file.",
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the file to read", required=True),
            ToolParameter(name="encoding", type="string", description="File encoding (default: utf-8)", required=False, default="utf-8")
        ],
        returns="Content of the file as a string",
        category=ToolCategory.FILE_SYSTEM
    )
    def read_file(self, path: str, encoding: str = "utf-8"):
        """
        Read a file from the filesystem
        """
        max_content_size = self.max_content_size

        def _read_text_file(p: str) -> str:
            with open(p, "r", encoding=encoding, errors="replace") as f:
                content = f.read()
            if len(content) > max_content_size:
                logger.warning(f"File {p} too large ({len(content)} chars), truncating to {max_content_size}")
                content = (
                    content[:max_content_size]
                    + f"\n\n[Content truncated - file was {len(content)} characters, showing first {max_content_size}]"
                )
            return content
        
        def _exists_file(p: str) -> bool:
            try:
                return os.path.isfile(p)
            except OSError:
                return False
        

        logger.debug(msg=f"FileSystemTool.read_file (REAL FS): Requested path: {path}")
        # 1) Exact match
        if _exists_file(path):
            logger.debug(f"Read file from disk (exact match): {path}")
            return _read_text_file(path)
        
        # 2) Path normalization / guessing
        normalizations_to_try = [path]
        # single <-> double slash conversions
        if "/" in path and "//" not in path:
            normalizations_to_try.append(path.replace("/", "//"))
        if "//" in path:
            normalizations_to_try.append(path.replace("//", "/"))
        
        # backslash conversions
        normalizations_to_try.extend([
            path.replace("\\", "/"),
            path.replace("\\", "//"),
        ])

        # prefix tweaks
        if not path.startswith("/"):
            normalizations_to_try.extend(["/" + path, "//" + path])
        
        # suffix-based tries (progressively shorter)
        if "/" in path:
            parts = path.split("/")
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                suffix_double = "//".join(parts[i:])
                normalizations_to_try.extend([suffix, suffix_double, "/" + suffix, "//" + suffix])

        # last N components (1..3)
        parts2 = [p for p in path.split("/") if p]
        if len(parts2) > 1:
            for i in range(1, min(4, len(parts2))):
                suffix = "/".join(parts2[-i:])
                suffix_double = "//".join(parts2[-i:])
                normalizations_to_try.extend([suffix, suffix_double, "/" + suffix, "//" + suffix])

        # de-dup preserve order
        normalizations_to_try = list(dict.fromkeys(normalizations_to_try))

        for candidate in normalizations_to_try:
            if _exists_file(candidate):
                logger.debug(f"Read file from disk (normalized {path} -> {candidate})")
                return _read_text_file(candidate)
            else:
                logger.debug(f"Normalization attempt failed: {candidate} not found")

        # 3) Fuzzy matching by filename: search in a reasonable root
        # Choose search roots:
        # - If path is absolute: search under its anchor/drive (may be huge; limit depth by using ** with glob)
        # - If relative: search under current working directory
        requested = Path(path)
        filename = requested.name

        # Helper: choose roots
        if requested.is_absolute():
            # On Unix: "/" ; on Windows: "C:\"
            roots = [str(requested.anchor or os.path.sep)]
        else:
            roots = [os.getcwd()]

        # Limit search scope a bit: also include directory part if provided
        if requested.parent and str(requested.parent) not in ("", "."):
            parent_candidate = requested.parent
            if not parent_candidate.is_absolute():
                roots.insert(0, str(Path(os.getcwd()) / parent_candidate))

        # de-dup roots
        roots = list(dict.fromkeys(roots))

        if filename:
            logger.debug(f"Trying fuzzy matching on disk for filename: {filename}; roots={roots}")

            # We'll collect a small set of candidates to keep it fast
            fuzzy_candidates = []
            for root in roots:
                try:
                    pattern = os.path.join(root, "**", filename)
                    matches = glob.glob(pattern, recursive=True)
                    for m in matches:
                        if os.path.isfile(m):
                            fuzzy_candidates.append(m)
                    # stop early if too many
                    if len(fuzzy_candidates) > 200:
                        fuzzy_candidates = fuzzy_candidates[:200]
                        break
                except Exception as e:
                    logger.debug(f"Fuzzy glob failed under root {root}: {e}")

            # Try to pick best match: if _path_structures_match exists, use it; else first match
            for m in fuzzy_candidates:
                try:
                    if hasattr(self, "_path_structures_match"):
                        if self._path_structures_match(path, m):
                            logger.debug(f"Read file from disk (fuzzy match {path} -> {m})")
                            return _read_text_file(m)
                    else:
                        logger.debug(f"Read file from disk (fuzzy match {path} -> {m})")
                        return _read_text_file(m)
                except Exception as e:
                    logger.debug(f"Fuzzy candidate rejected {m}: {e}")

        # 4) Agent-friendly error
        # Provide actionable hints: cwd, attempted normalizations, and a small directory listing
        cwd = os.getcwd()

        # List a few nearby files if possible
        nearby = []
        try:
            base_dir = str(requested.parent) if str(requested.parent) not in ("", ".") else cwd
            if not os.path.isabs(base_dir):
                base_dir = str(Path(cwd) / base_dir)
            if os.path.isdir(base_dir):
                entries = sorted(os.listdir(base_dir))
                # show up to 30 items
                nearby = entries[:30]
        except Exception:
            pass

        helpful_error = f"❌ FILE NOT FOUND ON DISK: '{path}'\n\n"
        helpful_error += f"Working directory (cwd): {cwd}\n"
        helpful_error += f"Encoding: {encoding}\n\n"
        helpful_error += "Tried these path variants (first 15 shown):\n"
        for p in normalizations_to_try[:15]:
            helpful_error += f"   • {p}\n"
        if len(normalizations_to_try) > 15:
            helpful_error += f"   ... and {len(normalizations_to_try) - 15} more\n"

        if nearby:
            helpful_error += f"\nDirectory listing for: {base_dir}\n"
            helpful_error += "\n".join([f"   • {x}" for x in nearby]) + "\n"

        if filename:
            helpful_error += (
                "\nHints:\n"
                f" - Ensure the file exists and the path is correct.\n"
                f" - If you only know the filename, try searching for '{filename}' in the project.\n"
            )

        logger.info(f"read_file failed for path={path}")
        raise FileNotFoundError(helpful_error)

            

    @tool_function(
        description="Write content to a writable file. Creates itself and its parent directories if needed.",
        parameters=[
            ToolParameter("path", "string", "Path to the file to write", required=True),
            ToolParameter("content", "string", "Content to write to the file", required=True),
            ToolParameter("encoding", "string", "File encoding (default: utf-8)", required=False, default="utf-8"),
            ToolParameter("create_directories", "boolean", "Create parent directories if they don't exist", required=False, default=True),
        ],
        returns="Success message with file information.",
        category=ToolCategory.FILE_SYSTEM,
    )
    def write_file(self, path: str, content: str, encoding: str = "utf-8", create_directories: bool = True) -> str:
        """
        Directly write to the real filesystem:
        - No temp workspace indirection
        - No safety/sandbox permission check
        - Optionally auto-create parent directories
        - Enforces self.max_file_size if present
        """
        if self.read_only:
            raise PermissionError("FileSystemTool is in read-only mode")
        file_path = Path(path)

        # Optional size guard (keep your existing behavior if configured)
        max_size = getattr(self, "max_file_size", None)
        if max_size is not None:
            content_size = len(content.encode(encoding, errors="replace"))
            if content_size > max_size:
                raise ValueError(f"Content too large: {content_size} bytes (max: {max_size})")
        
        if create_directories:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(file_path, "w", encoding=encoding, errors="replace") as f:
            f.write(content)

        logger.info(f"Wrote file to disk: {str(file_path)} ({len(content)} characters)")
        return f"Successfully wrote {len(content)} characters to {str(file_path)}"


    @tool_function(
        description="List files and directories in a directory.",
        parameters=[
            ToolParameter("path", "string", "Path to the directory to list", required=True),
            ToolParameter("include_hidden", "boolean", "Include hidden files (starting with .)", required=False, default=False),
            ToolParameter("recursive", "boolean", "List files recursively", required=False, default=False),
        ],
        returns="List of files and directories with metadata",
        category=ToolCategory.FILE_SYSTEM,
    )
    def list_directory(self, path: str, include_hidden: bool = False, recursive: bool = False) -> List[Dict[str, Any]]:
        """
        List contents of a real filesystem directory.

        Changes vs original:
        - Removed sandbox permission checks (_safe_path_operation / _is_path_allowed)
        - Removed scenario_context/project_files virtual listing

        Kept:
        - include_hidden filtering
        - recursive listing
        - per-item metadata (size/mtime/permissions/extension)
        - per-item error tolerance (skip unreadable entries)
        """
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            if not dir_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            items: List[Dict[str, Any]] = []

            iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()

            for item in iterator:
                # Skip hidden files/dirs if not requested
                if not include_hidden and item.name.startswith("."):
                    continue

                try:
                    stat = item.stat()
                    item_info: Dict[str, Any] = {
                        "name": item.name,
                        "path": str(item.relative_to(dir_path)),
                        "absolute_path": str(item.resolve()),
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": stat.st_mtime,
                        "permissions": oct(stat.st_mode)[-3:],
                    }

                    if item.is_file():
                        item_info["extension"] = item.suffix

                    items.append(item_info)

                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not access {item}: {e}")
                    continue

            logger.debug(f"Listed directory: {path} ({len(items)} items)")
            return items

        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            raise FileNotFoundError(f"Error listing directory {path}: {e}")
    

    @tool_function(
        description="Check if a file or directory exists.",
        parameters=[
            ToolParameter("path", "string", "Path to check", required=True)
        ],
        returns="Boolean indicating whether the path exists and metadata if it exists.",
        category=ToolCategory.FILE_SYSTEM,
    )
    def check_path_exists(self, path: str):
        """
        Check if a path exists
        """
        try:
            file_path = Path(path)
            exists = file_path.exists()
            
            result: Dict[str, Any]= {"exists": exists}
            
            if exists:
                stat = file_path.stat()
                result.update({
                    "type": "directory" if file_path.is_dir() else "file",
                    "size": stat.st_size if file_path.is_file() else None,
                    "modified": stat.st_mtime,
                    "permissions": oct(stat.st_mode)[-3:]
                })
                
                if file_path.is_file():
                    result["extension"] = file_path.suffix
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking path {path}: {e}")
            return {
                "exists": False,
                "error": str(e)
            }


    @tool_function(
        description="Search for files matching a glob pattern in the filesystem",
        parameters=[
            ToolParameter("pattern", "string", "Glob pattern to search for", required=True),
            ToolParameter("directory", "string", "Directory to search in (default: current directory)", required=False, default="."),
            ToolParameter("recursive", "boolean", "Search recursively", required=False, default=True),
        ],
        returns="List of matching files with metadata",
        category=ToolCategory.FILE_SYSTEM,
    )
    def search_files_glob(self, pattern: str, directory: str = ".", recursive: bool = True):
        """
        Search for files/dirs in the real filesystem matching a glob pattern.
        """
        try:
            search_path = Path(directory)

            if not search_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            if not search_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {directory}")

            matches: List[Dict[str, Any]] = []

            iterator = search_path.rglob(pattern) if recursive else search_path.glob(pattern)
            
            for match in iterator:
                try:
                    stat = match.stat()
                    match_info: Dict[str, Any] = {
                        "name": match.name,
                        "path": str(match.relative_to(search_path)),
                        "absolute_path": str(match.resolve()),
                        "type": "directory" if match.is_dir() else "file",
                        "size": stat.st_size if match.is_file() else None,
                        "modified": stat.st_mtime,
                    }
                    if match.is_file():
                        match_info["extension"] = match.suffix

                    matches.append(match_info)

                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not access {match}: {e}")
                    continue

            logger.debug(f"Found {len(matches)} matches for pattern '{pattern}' in {directory}")
            return matches
        
        except Exception as e:
            logger.error(f"Error searching for pattern {pattern} in {directory}: {e}")
            raise


    @tool_function(
        description="Get the current working directory",
        parameters=[],
        returns="Current working directory path",
        category=ToolCategory.FILE_SYSTEM,
    )
    def get_current_directory(self):
        """
        Get the current directory
        """
        try:
            current_dir = Path.cwd()
            
            # # Check if current directory is allowed
            # if not self._is_path_allowed(current_dir):
            #     raise PermissionError("Current directory is outside allowed directories")
            
            return str(current_dir)
            
        except Exception as e:
            logger.error(f"Error getting current directory: {e}")
            raise
    

    @tool_function(
        description="Finish the task. Call this only when you think all the task is done.",
        parameters=[],
        returns="A status dict indicating the agent requested to finish.",
        category=ToolCategory.FILE_SYSTEM,
    )
    def finish_task(self):
        return {
            "done": True,
            "message": "finish_task called"
        }
    

    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "readonly_mode": self.read_only,
            "max_file_size": self.max_file_size,
            "functions": len(self.functions)
        }
    
    def _path_structures_match(self, requested_path: str, available_path: str) -> bool:
        """Check if two paths represent the same file with different slash formats"""
        # Normalize both paths to compare structure
        req_normalized = requested_path.replace("//", "/").replace("\\", "/").strip("/")
        avail_normalized = available_path.replace("//", "/").replace("\\", "/").strip("/")
        
        # Split into components
        req_parts = [p for p in req_normalized.split("/") if p]
        avail_parts = [p for p in avail_normalized.split("/") if p]
        
        # If the requested path is a suffix of the available path, it's a match
        if len(req_parts) <= len(avail_parts):
            return req_parts == avail_parts[-len(req_parts):]
        
        return False
        