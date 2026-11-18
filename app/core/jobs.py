"""白名单作业运行器:只能跑注册表里的 make 目标,argv 写死在本模块,前端只传作业名。

- start_new_session=True 起独立会话,stop 时 killpg 连 make → uv → python 一起收;
- 日志覆盖写 log/acceptance/<job>.log,页面轮询 tail 回显尾部;
- 绝不 shell=True,绝不接受任何来自请求的命令片段。
"""
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "log" / "acceptance"


class UnknownJob(KeyError):
    """作业名不在白名单。"""


class JobRunning(RuntimeError):
    """同名作业还在跑,拒绝重入。"""


class JobNotRunning(RuntimeError):
    """作业未在运行,无可停止。"""


@dataclass(frozen=True)
class JobSpec:
    name: str
    target: str          # Makefile 里真实存在的目标;argv 固定为 ["make", target]
    heavy: bool = False  # 重作业(调 LLM / 清库),页面需二次确认
    description: str = ""


JOBS: dict[str, JobSpec] = {spec.name: spec for spec in [
    JobSpec("kb-preview", "kb-preview", description="预览建库材料切块(dry-run)"),
    JobSpec("kb-build", "kb-build", description="建库材料切块入库(pending)"),
    JobSpec("kb-vectorize", "kb-vectorize", description="补向量化 pending 块"),
    JobSpec("kb-mine", "kb-mine", heavy=True, description="对话挖知识(调 LLM)"),
    JobSpec("kb-reset", "kb-reset", heavy=True, description="清空知识库(MySQL+Milvus)"),
    JobSpec("seed-conv", "seed-conv", description="灌合成对话 seed"),
    JobSpec("eval-rag", "eval-rag", heavy=True,
            description="RAG 评估(四策略对照):需 Milvus + 已建库 + 上游可用,分钟级"),
]}

_procs: dict[str, subprocess.Popen] = {}


def _spec(name: str) -> JobSpec:
    spec = JOBS.get(name)
    if spec is None:
        raise UnknownJob(name)
    return spec


def status(name: str) -> dict:
    spec = _spec(name)
    proc = _procs.get(name)
    running = proc is not None and proc.poll() is None
    return {
        "name": spec.name,
        "target": spec.target,
        "heavy": spec.heavy,
        "description": spec.description,
        "running": running,
        "started": proc is not None,
        "pid": proc.pid if proc is not None else None,
        "exit_code": proc.poll() if proc is not None else None,
        "log": str(LOG_DIR / f"{spec.name}.log"),
    }


def list_jobs() -> list[dict]:
    return [status(name) for name in JOBS]


def start(name: str) -> dict:
    spec = _spec(name)
    if status(name)["running"]:
        raise JobRunning(name)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{spec.name}.log"
    with open(log_path, "w", encoding="utf-8") as log_file:  # 覆盖写,不追加
        proc = subprocess.Popen(
            ["make", spec.target],            # argv 只来自注册表,与终端敲的是同一条命令
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _procs[name] = proc
    return status(name)


def stop(name: str) -> dict:
    _spec(name)
    proc = _procs.get(name)
    if proc is None or proc.poll() is not None:
        raise JobNotRunning(name)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # 收整个进程组
    except ProcessLookupError:
        pass  # 刚好自己退出了,视同已停止
    return status(name)


def tail(name: str, lines: int = 50) -> str:
    _spec(name)
    log_path = LOG_DIR / f"{name}.log"
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(content.splitlines()[-lines:])
