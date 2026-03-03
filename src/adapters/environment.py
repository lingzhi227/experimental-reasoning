"""Environment Adapters — benchmark-specific execution environments.

Each benchmark needs its own environment adapter that translates
ER actions into benchmark-specific API calls and returns structured results.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-container REPL server script (embedded as string, written to work_dir)
# ---------------------------------------------------------------------------

_ER_SERVER_SCRIPT = r'''#!/usr/bin/env python3
"""ER JSON-line REPL server — runs inside Docker container.

Protocol:
  stdin  -> {"code": "..."}           (one JSON object per line)
  stdout <- {"output": "...", "error": "...", "answer": "..."}

Real stdout/stderr from exec'd code are captured via StringIO;
only the JSON protocol uses the actual pipe.
"""
import io
import json
import os
import sys
import traceback

# Persistent namespace across steps (variables survive)
_namespace = {"__builtins__": __builtins__}

# Change to workspace
os.chdir("/workspace")

# Suppress warnings by default
import warnings
warnings.filterwarnings("ignore")

MAX_OUTPUT = 50_000  # 50 KB stdout cap

def run_one(code: str) -> dict:
    old_out, old_err = sys.stdout, sys.stderr
    cap_out, cap_err = io.StringIO(), io.StringIO()
    try:
        sys.stdout, sys.stderr = cap_out, cap_err
        exec(code, _namespace)
        out = cap_out.getvalue()
        err = cap_err.getvalue()
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n... [output truncated at 50KB]"
        result = {"output": out, "error": err if err else None}
        # Extract _answer if set
        if "_answer" in _namespace:
            result["answer"] = str(_namespace["_answer"])
        return result
    except Exception as e:
        return {
            "output": cap_out.getvalue()[:MAX_OUTPUT],
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
    finally:
        sys.stdout, sys.stderr = old_out, old_err

# Main loop: read JSON lines from stdin, write JSON lines to stdout
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        sys.stdout.write(json.dumps({"output": "", "error": f"Invalid JSON: {exc}"}) + "\n")
        sys.stdout.flush()
        continue

    code = request.get("code", "")
    result = run_one(code)
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
'''


class LocalPythonEnvironment:
    """Execute Python code locally and capture output.

    Used for benchmarks that require code execution
    (ScienceAgentBench, BixBench, MLE-bench).
    """

    def __init__(
        self,
        allowed_imports: list[str] | None = None,
        timeout_seconds: int = 120,
        working_dir: str | None = None,
    ) -> None:
        self.allowed_imports = allowed_imports
        self.timeout_seconds = timeout_seconds
        self.working_dir = working_dir
        self._namespace: dict[str, Any] = {}

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute code in a local Python environment."""
        code = action.get("code")
        if not code:
            return {"output": "", "error": "No code provided", "metrics": {}}

        return await asyncio.to_thread(self._run_code, code)

    def _run_code(self, code: str) -> dict[str, Any]:
        """Run code synchronously, capturing stdout/stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_out = io.StringIO()
        captured_err = io.StringIO()

        try:
            sys.stdout = captured_out
            sys.stderr = captured_err

            exec(code, self._namespace)

            stdout_text = captured_out.getvalue()
            stderr_text = captured_err.getvalue()

            return {
                "output": stdout_text,
                "error": stderr_text if stderr_text else None,
                "metrics": self._namespace.get("_metrics", {}),
            }

        except Exception as e:
            return {
                "output": captured_out.getvalue(),
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                "metrics": {},
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def reset(self) -> None:
        """Reset the execution namespace."""
        self._namespace.clear()


class DockerPythonEnvironment:
    """Execute Python code inside a Docker container via JSON-line REPL.

    A persistent Python process runs inside the container, reading JSON
    code requests from stdin and writing JSON results to stdout.
    Variables persist across calls (same as LocalPythonEnvironment).
    """

    def __init__(
        self,
        docker_image: str,
        work_dir: str | Path,
        timeout_seconds: int = 300,
    ) -> None:
        self.docker_image = docker_image
        self.work_dir = Path(work_dir)
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen | None = None
        self._container_name = f"er_repl_{os.getpid()}"

        # Write the server script to work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        server_path = self.work_dir / ".er_server.py"
        server_path.write_text(_ER_SERVER_SCRIPT)

        # Start the Docker container with the REPL server
        cmd = [
            "docker", "run", "--rm", "-i",
            "--name", self._container_name,
            "-v", f"{self.work_dir.resolve()}:/workspace",
            self.docker_image,
            "python", "-u", "/workspace/.er_server.py",
        ]
        logger.info("Starting Docker REPL: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send code to the Docker REPL and return the result."""
        code = action.get("code")
        if not code:
            return {"output": "", "error": "No code provided", "metrics": {}}

        if self._process is None or self._process.poll() is not None:
            return {"output": "", "error": "Docker REPL process is not running", "metrics": {}}

        return await asyncio.to_thread(self._send_code, code)

    def _send_code(self, code: str) -> dict[str, Any]:
        """Send code to REPL (blocking), read JSON result."""
        proc = self._process
        if proc is None or proc.stdin is None or proc.stdout is None:
            return {"output": "", "error": "Docker REPL not available", "metrics": {}}

        request = json.dumps({"code": code}) + "\n"
        try:
            proc.stdin.write(request.encode())
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return {"output": "", "error": f"Docker REPL stdin broken: {e}", "metrics": {}}

        # Read one JSON line with timeout
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        try:
            ready = sel.select(timeout=self.timeout_seconds)
            if not ready:
                # Timeout — kill and report
                self._kill_container()
                return {
                    "output": "",
                    "error": f"Code execution timed out after {self.timeout_seconds}s",
                    "metrics": {},
                }

            line = proc.stdout.readline()
            if not line:
                stderr_out = ""
                if proc.stderr:
                    stderr_out = proc.stderr.read().decode(errors="replace")[:2000]
                return {
                    "output": "",
                    "error": f"Docker REPL process exited unexpectedly. stderr: {stderr_out}",
                    "metrics": {},
                }

            result = json.loads(line.decode())
            metrics = {}
            if "answer" in result:
                metrics["answer"] = result["answer"]
            return {
                "output": result.get("output", ""),
                "error": result.get("error"),
                "metrics": metrics,
            }
        except json.JSONDecodeError as e:
            return {"output": "", "error": f"Invalid JSON from REPL: {e}", "metrics": {}}
        except Exception as e:
            return {"output": "", "error": f"Docker REPL error: {e}", "metrics": {}}
        finally:
            sel.close()

    def _kill_container(self) -> None:
        """Force-kill the Docker container."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", self._container_name],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
        self._process = None

    def cleanup(self) -> None:
        """Gracefully shut down the Docker REPL."""
        proc = self._process
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._kill_container()
        except Exception:
            self._kill_container()
        finally:
            self._process = None

        # Clean up server script
        server_path = self.work_dir / ".er_server.py"
        if server_path.exists():
            server_path.unlink()

    def reset(self) -> None:
        """Reset by restarting the container."""
        self.cleanup()
        self.__init__(self.docker_image, self.work_dir, self.timeout_seconds)


class ToolCallingEnvironment:
    """Execute tool calls against a registered set of tools.

    Used for benchmarks that use tool-calling interfaces
    (tau-bench).
    """

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register_tool(self, name: str, func: Any) -> None:
        """Register a callable tool."""
        self.tools[name] = func

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute tool calls."""
        tool_calls = action.get("tool_calls", [])
        if not tool_calls:
            return {"output": "", "error": "No tool calls provided", "metrics": {}}

        results = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {})

            if name not in self.tools:
                results.append({
                    "tool": name,
                    "error": f"Unknown tool: {name}",
                })
                continue

            try:
                if asyncio.iscoroutinefunction(self.tools[name]):
                    result = await self.tools[name](**args)
                else:
                    result = self.tools[name](**args)
                results.append({"tool": name, "result": result})
            except Exception as e:
                results.append({"tool": name, "error": str(e)})

        return {
            "output": results,
            "error": None,
            "metrics": {},
        }


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = responses or []
        self._call_count = 0
        self.call_history: list[dict[str, Any]] = []

    async def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        self.call_history.append(action)

        if self._call_count < len(self.responses):
            result = self.responses[self._call_count]
        else:
            result = {"output": "mock output", "error": None, "metrics": {}}

        self._call_count += 1
        return result
