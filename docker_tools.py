"""
Docker-based Tools for running the Dev Arena Agent inside a SWE-bench container.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _truncate(text: str, max_chars: int = 12000) -> str:
    """Truncate long text from the middle."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [TRUNCATED] ...\n" + text[-half:]

class DockerTools:
    """
    Implements the same API as `Tools` but executes everything via `docker exec`.
    This allows our agent to work exactly the same way, but inside an isolated container.
    """
    def __init__(self, container_name: str, workspace_dir: str = "/testbed",
                 fail_to_pass: list[str] | None = None,
                 pass_to_pass: list[str] | None = None):
        self.container_name = container_name
        self.workspace_dir = workspace_dir
        self._fail_to_pass = fail_to_pass or []
        self._pass_to_pass = pass_to_pass or []

    def _exec(self, cmd: str) -> subprocess.CompletedProcess:
        # We run as root or whatever the default user is in the swebench container.
        # Run as a login shell (-lc) to ensure `.bashrc` / Conda environments are fully loaded.
        return subprocess.run(
            ["docker", "exec", "-w", self.workspace_dir, self.container_name, "bash", "-lc", cmd],
            capture_output=True, text=True
        )

    def _clean_path(self, path: str) -> str:
        if path.startswith(self.workspace_dir + "/"):
            return path[len(self.workspace_dir) + 1:]
        if path.startswith("/"):
            return path.lstrip("/")
        return path

    def file_read(self, path: str, offset: int = 1, limit: Optional[int] = None) -> str:
        path = self._clean_path(path)
        
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
            
        # Use python inside the container to read safely
        script = f"""
import sys
try:
    with open('{path}', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        total = len(lines)
        if total == 0:
            print("File: {path} -- File is empty.")
            sys.exit(0)
        
        start = max(1, {offset})
        end = start + {limit if limit is not None else 'total'} - 1
        end = min(end, total)
        
        print(f"File: {path} ({{total}} lines)")
        for i in range(start - 1, end):
            print(f"{{i + 1:6d}}|{{lines[i]}}", end='')
except Exception as e:
    print(f"Error: {{e}}")
"""
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(script)
            temp_path = f.name
            
        subprocess.run(["docker", "cp", temp_path, f"{self.container_name}:/tmp/read_file.py"], capture_output=True)
        os.remove(temp_path)
        
        res = self._exec("python /tmp/read_file.py")
        if res.returncode != 0:
            return f"Error reading file: {res.stderr}"
        return _truncate(res.stdout)

    def file_write(self, path: str, content: str) -> str:
        path = self._clean_path(path)
        # Write to a temp file locally, then docker cp
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        # Ensure parent dir exists
        dirname = os.path.dirname(path)
        if dirname:
            self._exec(f"mkdir -p {dirname}")
            
        res = subprocess.run(
            ["docker", "cp", temp_path, f"{self.container_name}:{self.workspace_dir}/{path}"],
            capture_output=True, text=True
        )
        os.remove(temp_path)
        
        if res.returncode != 0:
            return f"Error writing file: {res.stderr}"
        return f"Created {path} ({len(content)} chars)"

    def file_patch(self, path: str, old_string: str, new_string: str, replace_all: bool = False, use_regex: bool = False) -> str:
        path = self._clean_path(path)
        import re
        # We'll read the file locally, patch it, and push it back.
        res = self._exec(f"cat {path}")
        if res.returncode != 0:
            return f"Error: file '{path}' does not exist or cannot be read."
            
        content = res.stdout
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
                return f"Error: regex pattern not found in {path}."
            if occurrences > 1 and not replace_all:
                return f"Error: regex pattern matches {occurrences} locations in {path}."
            
            new_content = pattern.sub(new_norm, content_norm, count=0 if replace_all else 1)
        else:
            occurrences = content_norm.count(old_norm)
            if occurrences == 0:
                # Provide a helpful hint if possible by finding the first line of old_string
                first_line = old_norm.strip().split("\n")[0].strip()
                if first_line:
                    hint_matches = []
                    for i, line in enumerate(content_norm.split("\n")):
                        if first_line in line:
                            hint_matches.append(f"Line {i+1}: {line.strip()}")
                            if len(hint_matches) >= 3:
                                break
                    if hint_matches:
                        hint = "\nHint: Did you mean one of these lines?\n" + "\n".join(hint_matches)
                        return f"Error: old_string not found in {path}. Check for whitespace or indentation mismatches.{hint}"
                return f"Error: old_string not found in {path}. Check for exact whitespace/indentation."
            
            if occurrences > 1 and not replace_all:
                return f"Error: old_string matches {occurrences} locations in {path}."
                
            if replace_all:
                new_content = content_norm.replace(old_norm, new_norm)
            else:
                new_content = content_norm.replace(old_norm, new_norm, 1)
                
        if new_content == content_norm:
            return "Error: Replacement resulted in no changes to the file content."
            
        self.file_write(path, new_content)
        replaced = occurrences if replace_all else 1
        return f"Patched {path} successfully ({replaced} replacement{'s' if replaced > 1 else ''})."

    def run_tests(self) -> str:
        """Run the official FAIL_TO_PASS tests and return a clean pass/fail summary."""
        if not self._fail_to_pass:
            return "run_tests: no test IDs configured for this task."

        # Run only the tests that should flip from FAIL to PASS
        test_ids = " ".join(self._fail_to_pass)
        cmd = f"cd {self.workspace_dir} && python -m pytest {test_ids} -x --tb=short -q 2>&1"
        res = self._exec(cmd)
        output = _truncate((res.stdout or "") + (res.stderr or ""))
        
        passed = (res.returncode == 0)

        # Also run pass_to_pass tests to catch regressions
        regression_summary = ""
        if self._pass_to_pass:
            p2p_ids = " ".join(self._pass_to_pass[:20])  # cap to avoid huge runs
            p2p_res = self._exec(
                f"cd {self.workspace_dir} && python -m pytest {p2p_ids} --tb=no -q 2>&1"
            )
            regression_summary = "\n\n--- REGRESSION CHECK (PASS_TO_PASS) ---\n" + _truncate(
                (p2p_res.stdout or "") + (p2p_res.stderr or ""), max_chars=3000
            )
            if p2p_res.returncode != 0:
                passed = False
                
        status_header = "[CI ORACLE: ALL TESTS PASSED]\n" if passed else "[CI ORACLE: TESTS FAILED]\n"
        return status_header + output + regression_summary

    def shell(self, command: str, timeout: int = 60, env: Optional[dict] = None) -> str:
        # Run via docker exec with timeout
        # Using python to enforce timeout inside docker to avoid leaving zombie processes
        env_str = repr(env) if env else "None"
        script = f"""
import subprocess
import os
try:
    custom_env = os.environ.copy()
    env_updates = {env_str}
    if env_updates:
        custom_env.update(env_updates)
        
    result = subprocess.run(
        {repr(command)},
        shell=True,
        capture_output=True,
        text=True,
        timeout={timeout},
        env=custom_env
    )
    print(f"exit_code={{result.returncode}}")
    if result.stdout:
        print("stdout:")
        print(result.stdout)
    if result.stderr:
        print("stderr:")
        print(result.stderr)
except subprocess.TimeoutExpired:
    print("Error: command timed out after {timeout}s.")
except Exception as e:
    print(f"Error: {{e}}")
"""
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(script)
            temp_path = f.name
            
        subprocess.run(["docker", "cp", temp_path, f"{self.container_name}:/tmp/run_cmd.py"], capture_output=True)
        os.remove(temp_path)
        
        res = self._exec("python /tmp/run_cmd.py")
        return _truncate(res.stdout or res.stderr)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob_filter: Optional[str] = None,
        context_lines: int = 0,
        case_insensitive: bool = False,
        output_mode: str = "content",
    ) -> str:
        flags = "-i" if case_insensitive else ""
        ctx = f"-C {context_lines}" if context_lines > 0 else ""
        files_only = "-l" if output_mode == "files" else ""
        count_only = "-c" if output_mode == "count" else ""
        line_numbers = "-n" if output_mode == "content" else ""
        
        path_arg = path if path else "."
        # Simple grep via shell
        # Note: grep handles basic patterns. For complex ones, we rely on standard grep
        cmd = f"grep -r {flags} {ctx} {files_only} {count_only} {line_numbers} {repr(pattern)} {path_arg}"
        res = self._exec(cmd)
        return _truncate(res.stdout)
        
    def glob_search(self, pattern: str, path: Optional[str] = None) -> str:
        path_arg = path if path else "."
        cmd = f"find {path_arg} -path {repr('*' + pattern.replace('**/', '') + '*')} -not -path '*/.git/*' -not -path '*/__pycache__/*'"
        res = self._exec(cmd)
        return _truncate(res.stdout)
