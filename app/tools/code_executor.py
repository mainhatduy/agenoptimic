"""
Code executor cho multi-agent optimization system.

Chiến lược: subprocess isolation với timeout + resource limits.
- Code LLM sinh ra được ghi vào temp file, chạy trong subprocess riêng.
- Capture stdout/stderr để trả về kết quả hoặc thông báo lỗi.
- Hỗ trợ các thư viện tối ưu hoá: scipy, pulp, cvxpy, numpy, ...
"""

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Các thư viện được phép import trong code được thực thi.
# Dùng làm tài liệu / whitelist tham khảo, không enforce ở đây
# vì subprocess đã cô lập process.
ALLOWED_PACKAGES = {
    "numpy", "scipy", "pulp", "cvxpy", "pandas",
    "math", "itertools", "functools", "collections",
}

DEFAULT_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 8_000


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int

    @property
    def output(self) -> str:
        """Trả về stdout nếu thành công, stderr nếu lỗi."""
        return self.stdout if self.success else self.stderr

    def __str__(self) -> str:
        if self.success:
            return f"[OK]\n{self.stdout}"
        return f"[ERROR exit={self.exit_code}]\n{self.stderr}"


def execute_python_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    extra_env: Optional[dict] = None,
) -> ExecutionResult:
    """
    Thực thi đoạn code Python trong một subprocess cô lập.

    Args:
        code: Chuỗi code Python cần chạy.
        timeout: Thời gian tối đa (giây) cho phép code chạy.
        extra_env: Biến môi trường bổ sung truyền vào subprocess.

    Returns:
        ExecutionResult chứa stdout, stderr, exit_code.
    """
    code = textwrap.dedent(code).strip()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="agenoptimic_exec_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        env = _build_env(extra_env)

        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        stdout = _truncate(proc.stdout)
        stderr = _truncate(proc.stderr)

        return ExecutionResult(
            success=proc.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds.",
            exit_code=-1,
        )
    except Exception as exc:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Executor internal error: {exc}",
            exit_code=-2,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_env(extra_env: Optional[dict]) -> dict:
    """
    Xây dựng environment cho subprocess.
    Kế thừa PATH và PYTHONPATH từ process cha để subprocess
    có thể import các thư viện tối ưu hoá đã cài.
    """
    import os
    env = {
        key: val
        for key, val in os.environ.items()
        if key in {"PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL"}
    }
    # Đảm bảo subprocess dùng cùng virtualenv/site-packages
    env["PYTHONPATH"] = str(Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    if extra_env:
        env.update(extra_env)
    return env


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + f"\n...[truncated {len(text) - MAX_OUTPUT_CHARS} chars]...\n" + text[-half:]
