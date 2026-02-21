"""
Extracted tool implementations for RealWorldJunior.
These tools mirror the exact functionality of the ai_scrum_team developer tools.
"""

import logging
import os
import subprocess
import ast
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _safe_workspace_path(workspace: Path, path_str: str) -> Path:
    """Ensure path stays within workspace."""
    try:
        # Strip leading slashes to prevent pathlib from treating it as absolute
        clean_path = path_str.lstrip("/")
        # Also remove any workspace prefix if they hallucinate absolute path
        workspace_str = str(workspace.resolve())
        if clean_path.startswith(workspace_str.lstrip("/")):
            clean_path = clean_path[len(workspace_str.lstrip("/")):]
            clean_path = clean_path.lstrip("/")
            
        target = (workspace / clean_path).resolve()
        if not str(target).startswith(workspace_str):
            return workspace  # Fallback to root
        return target
    except Exception:
        return workspace

def _truncate(text: str, max_chars: int = 12000) -> str:
    """Truncate long text from the middle."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [TRUNCATED] ...\n" + text[-half:]

def _syntax_check_content(content: str, ext: str) -> tuple[bool, str]:
    """Basic syntax check before saving."""
    if ext == ".py":
        try:
            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, str(e)
    return True, ""

class Tools:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def file_read(self, path: str, offset: int = 1, limit: Optional[int] = None) -> str:
        # Handle LLM hallucinations where offset is passed as a string list "[236, 247]"
        if isinstance(offset, str) and offset.strip().startswith("[") and offset.strip().endswith("]"):
            try:
                import ast
                offset = ast.literal_eval(offset.strip())
            except Exception:
                pass
                
        # Handle LLM hallucinations where offset is passed as a list [start, end]
        if isinstance(offset, list):
            if len(offset) == 2:
                try:
                    limit = int(offset[1]) - int(offset[0]) + 1
                    offset = int(offset[0])
                except (ValueError, TypeError):
                    offset = 1
            elif len(offset) > 0:
                try:
                    offset = int(offset[0])
                except (ValueError, TypeError):
                    offset = 1
            else:
                offset = 1
                
        try:
            offset = int(offset)
        except (ValueError, TypeError):
            return f"Error: 'offset' parameter must be an integer, got {type(offset).__name__} '{offset}'"
            
        file_path = _safe_workspace_path(self.workspace, path)
        if not file_path.exists():
            return "Error: file not found: %s" % path
            
        all_lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        total = len(all_lines)
        
        if total == 0:
            return "File: %s -- File is empty." % path
            
        start = max(1, offset)
        end = start + limit - 1 if limit is not None else total
        
        numbered = [
            "%6d|%s" % (i + 1, all_lines[i])
            for i in range(start - 1, min(end, total))
        ]
        header = "File: %s (%d lines)\n" % (path, total)
        return header + _truncate("".join(numbered))

    def file_write(self, path: str, content: str) -> str:
        file_path = _safe_workspace_path(self.workspace, path)
        existed = file_path.exists()
        
        ok, err = _syntax_check_content(content, file_path.suffix.lower())
        if not ok:
            return "Write rejected -- syntax error in content:\n%s" % err
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        
        rel = file_path.relative_to(self.workspace)
        if existed:
            return "Wrote %s (%d chars). WARNING: Overwrote existing file." % (rel, len(content))
        return "Created %s (%d chars)" % (rel, len(content))

    def file_patch(self, path: str, old_string: str, new_string: str, replace_all: bool = False, use_regex: bool = False) -> str:
        import re
        file_path = _safe_workspace_path(self.workspace, path)
        if not file_path.exists():
            return "Error: file '%s' does not exist." % path
            
        content = file_path.read_text(encoding="utf-8")
        content_norm = content.replace("\r\n", "\n")
        old_norm = old_string.replace("\r\n", "\n")
        new_norm = new_string.replace("\r\n", "\n")
        
        if old_norm == new_norm:
            return "Error: old_string and new_string are identical. Nothing to change."
            
        if use_regex:
            try:
                pattern = re.compile(old_norm, flags=re.MULTILINE)
            except re.error as e:
                return f"Error compiling regex: {e}"
            
            occurrences = len(pattern.findall(content_norm))
            if occurrences == 0:
                return "Error: regex pattern not found in %s." % path
            if occurrences > 1 and not replace_all:
                return "Error: regex pattern matches %d locations in %s." % (occurrences, path)
                
            new_content = pattern.sub(new_norm, content_norm, count=0 if replace_all else 1)
        else:
            occurrences = content_norm.count(old_norm)
            if occurrences == 0:
                # Try to find a partial match to give a helpful hint
                first_line = old_norm.strip().split("\n")[0].strip()
                hint_str = ""
                if first_line:
                    hint_matches = []
                    for i, line in enumerate(content_norm.split("\n")):
                        if first_line in line:
                            hint_matches.append(f"Line {i+1}: {line.strip()}")
                            if len(hint_matches) >= 3:
                                break
                    if hint_matches:
                        hint_str = "\nHint: Did you mean one of these lines?\n" + "\n".join(hint_matches)
                return "Error: old_string not found in %s. Check for exact whitespace/indentation.%s" % (path, hint_str)
                
            if occurrences > 1 and not replace_all:
                return "Error: old_string matches %d locations in %s." % (occurrences, path)
                
            if replace_all:
                new_content = content_norm.replace(old_norm, new_norm)
            else:
                new_content = content_norm.replace(old_norm, new_norm, 1)
                
        if new_content == content_norm:
            return "Error: Replacement resulted in no changes to the file content."
            
        # Optional: verify syntax after patching
        ok, err = _syntax_check_content(new_content, file_path.suffix.lower())
        if not ok:
            return "Edit rejected -- syntax error in result:\n%s" % err
            
        file_path.write_text(new_content, encoding="utf-8")
        
        # VERIFICATION: Read back from disk to guarantee patch applied exactly as intended
        verify_content = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if verify_content != new_content:
            return "Error: Patch failed verification on disk write."
            
        replaced = occurrences if replace_all else 1
        return "Patched %s successfully (%d replacement%s)." % (path, replaced, "s" if replaced > 1 else "")

    def shell(self, command: str, timeout: int = 60) -> str:
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(self.workspace),
        }
        
        try:
            result = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return "Error: command timed out after %ds." % timeout
            
        parts = ["exit_code=%d" % result.returncode]
        if result.stdout:
            parts.append("stdout:\n%s" % _truncate(result.stdout))
        if result.stderr:
            parts.append("stderr:\n%s" % _truncate(result.stderr))
            
        return "\n".join(parts)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob_filter: Optional[str] = None,
        context_lines: int = 0,
        case_insensitive: bool = False,
        output_mode: str = "content",
    ) -> str:
        return grep(self.workspace, pattern, path, glob_filter, context_lines, case_insensitive, output_mode)
        
    def glob_search(self, pattern: str, path: Optional[str] = None) -> str:
        return glob_search(self.workspace, pattern, path)

import re
import os
from pathlib import Path
from typing import Optional, Any
from fnmatch import fnmatch

_BINARY_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip", ".tar", ".gz",
    ".mp3", ".mp4", ".mov", ".wav", ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".class", ".woff", ".woff2", ".ttf", ".eot", ".ico"
})
_SECRET_FILENAMES = frozenset({".env", "secrets.json", "credentials.json", "id_rsa"})
_IGNORE_DIRS = frozenset({".git", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__", ".next", ".nuxt", "dist", "build"})

def _is_secret_file(path: Path) -> bool:
    return path.name in _SECRET_FILENAMES

def _should_skip_dir(dirname: str) -> bool:
    return dirname in _IGNORE_DIRS or dirname.endswith(".egg-info")

def _truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [TRUNCATED] ...\n" + text[-half:]

def grep(
    workspace: Path,
    pattern: str,
    path: Optional[str] = None,
    glob_filter: Optional[str] = None,
    context_lines: int = 0,
    case_insensitive: bool = False,
    output_mode: str = "content",
) -> str:
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return "Error: invalid regex '%s' -- %s" % (pattern, e)

    search_root = workspace
    if path:
        try:
            target = (workspace / path).resolve()
            if not str(target).startswith(str(workspace.resolve())):
                search_root = workspace
            else:
                search_root = target
        except Exception:
            search_root = workspace

        if not search_root.exists():
            return "Error: path '%s' not found." % path

    if search_root.is_file():
        search_files = [search_root]
    else:
        search_files = []
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if not _should_skip_dir(d)]
            for filename in files:
                fpath = Path(root) / filename
                if _is_secret_file(fpath) or fpath.suffix in _BINARY_EXTS:
                    continue
                if glob_filter and not fnmatch(filename, glob_filter):
                    continue
                search_files.append(fpath)

    matches_by_file = {}
    file_lines_cache = {}
    total_matches = 0

    for fpath in search_files:
        try:
            lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel = str(fpath.relative_to(workspace))
        for i, line in enumerate(lines):
            if regex.search(line):
                if rel not in matches_by_file:
                    matches_by_file[rel] = []
                    file_lines_cache[rel] = lines
                matches_by_file[rel].append((i + 1, line.strip()))
                total_matches += 1
                if total_matches >= 1000:
                    break
        if total_matches >= 1000:
            break

    if not matches_by_file:
        return "No matches found for '%s'." % pattern

    if output_mode == "files":
        return "\n".join(sorted(matches_by_file.keys()))

    if output_mode == "count":
        lines_out = [
            "%s: %d matches" % (f, len(m))
            for f, m in sorted(matches_by_file.items())
        ]
        return "\n".join(lines_out)

    results = []
    ctx = max(0, context_lines)
    for filepath, file_matches in sorted(matches_by_file.items()):
        all_lines = file_lines_cache.get(filepath, [])
        total_file_lines = len(all_lines)

        if ctx == 0:
            for line_no, line_text in file_matches:
                results.append("%s:%d: %s" % (filepath, line_no, line_text))
        else:
            shown_lines = set()
            for line_no, _ in file_matches:
                start = max(1, line_no - ctx)
                end = min(total_file_lines, line_no + ctx)
                for n in range(start, end + 1):
                    if n not in shown_lines:
                        shown_lines.add(n)
                        prefix = ":" if n == line_no else "-"
                        results.append(
                            "%s%s%d%s %s"
                            % (filepath, prefix, n, prefix, all_lines[n - 1].rstrip())
                        )
                if end < total_file_lines:
                    results.append("--")

    output = "\n".join(results)
    if total_matches >= 1000:
        output += "\n...(at least %d matches, results capped)" % total_matches
    return _truncate(output)

def glob_search(workspace: Path, pattern: str, path: Optional[str] = None) -> str:
    search_root = workspace
    if path:
        try:
            target = (workspace / path).resolve()
            if not str(target).startswith(str(workspace.resolve())):
                search_root = workspace
            else:
                search_root = target
        except Exception:
            search_root = workspace
            
        if not search_root.exists():
            return "Error: directory '%s' not found." % path

    glob_pattern = pattern
    if not glob_pattern.startswith("**/") and "/" not in glob_pattern:
        glob_pattern = "**/" + glob_pattern

    matches = []
    for match_path in search_root.glob(glob_pattern):
        if not match_path.is_file():
            continue
        rel_parts = match_path.relative_to(workspace).parts
        if any(_should_skip_dir(p) for p in rel_parts):
            continue
        if _is_secret_file(match_path) or match_path.suffix in _BINARY_EXTS:
            continue
        matches.append(match_path)

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not matches:
        return "No files matching '%s'." % pattern

    results = [str(m.relative_to(workspace)) for m in matches[:100]]
    output = "\n".join(results)
    if len(matches) > 100:
        output += "\n...(%d total matches, showing first 100)" % len(matches)
    return output

