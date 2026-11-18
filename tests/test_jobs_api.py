import re
import signal

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import jobs
from app.main import app

pytestmark = pytest.mark.asyncio(loop_scope="session")


class FakePopen:
    """替身进程:记录 argv/kwargs,poll() 可控返回码;绝不真跑 make。"""
    instances: list["FakePopen"] = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4321
        self._rc = None
        FakePopen.instances.append(self)

    def poll(self):
        return self._rc


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    FakePopen.instances = []
    monkeypatch.setattr(jobs, "_procs", {})
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path / "acceptance")
    monkeypatch.setattr(jobs.subprocess, "Popen", FakePopen)


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---- Step 2: 作业运行器的安全边界 ----

async def test_unknown_job_is_rejected(client):
    """作业名不在白名单 → 404;命令片段没有任何地方能传进来。"""
    async with client as c:
        assert (await c.post("/api/jobs/not-a-job")).status_code == 404
        assert (await c.get("/api/jobs/not-a-job")).status_code == 404
        assert (await c.post("/api/jobs/not-a-job/stop")).status_code == 404
        # shell 元字符混进作业名也只能得到 404,不会拼进命令
        assert (await c.post("/api/jobs/kb-build;rm -rf .")).status_code == 404
        assert (await c.post("/api/jobs/kb-build$(evil)")).status_code == 404
        assert (await c.post("/api/jobs/../../etc/passwd")).status_code == 404
    assert FakePopen.instances == []  # 一个进程都没起


async def test_argv_never_comes_from_request(client):
    """请求体里塞 argv/cmd 一律无视:进程只跑注册表写死的 ["make", <target>]。"""
    async with client as c:
        r = await c.post("/api/jobs/kb-build", json={"argv": ["rm", "-rf", "/"], "cmd": "evil"})
    assert r.status_code == 200
    assert len(FakePopen.instances) == 1
    proc = FakePopen.instances[0]
    assert proc.argv == ["make", "kb-build"]
    assert not proc.kwargs.get("shell", False)  # 绝不 shell=True
    assert proc.kwargs["start_new_session"] is True
    assert str(proc.kwargs["cwd"]) == str(jobs.ROOT)


async def test_running_job_refuses_reentry(client):
    """同名作业还在跑 → 409,别让两份进程抢同一批产物。"""
    async with client as c:
        r1 = await c.post("/api/jobs/kb-build")
        assert r1.status_code == 200
        assert r1.json()["running"] is True
        r2 = await c.post("/api/jobs/kb-build")
        assert r2.status_code == 409
        assert len(FakePopen.instances) == 1
        # 跑完(返回码 0)后允许再跑
        FakePopen.instances[0]._rc = 0
        r3 = await c.post("/api/jobs/kb-build")
        assert r3.status_code == 200
        assert len(FakePopen.instances) == 2


async def test_every_job_target_exists_in_makefile():
    """注册表里每个作业都得是 Makefile 里真有的目标,免得按钮指向不存在的配方。"""
    makefile = (jobs.ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {m.group(1) for m in re.finditer(r"^([A-Za-z0-9][A-Za-z0-9_-]*):", makefile, re.M)}
    assert targets, "Makefile 解析不出任何目标"
    for name, spec in jobs.JOBS.items():
        assert spec.target in targets, f"作业 {name} 指向不存在的 make 目标 {spec.target}"


# ---- 运行器行为 ----

async def test_registry_heavy_flags_and_list(client):
    assert jobs.JOBS["kb-mine"].heavy is True
    assert jobs.JOBS["kb-reset"].heavy is True
    assert jobs.JOBS["eval-rag"].heavy is True   # 分钟级评估重活,页面二次确认
    assert jobs.JOBS["kb-build"].heavy is False
    async with client as c:
        r = await c.get("/api/jobs")
    assert r.status_code == 200
    listed = r.json()["jobs"]
    assert {j["name"] for j in listed} == set(jobs.JOBS)
    by_name = {j["name"]: j for j in listed}
    assert by_name["kb-mine"]["heavy"] is True
    assert by_name["kb-reset"]["heavy"] is True
    assert all(j["running"] is False for j in listed)


async def test_stop_kills_whole_process_group(client, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(jobs.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    async with client as c:
        assert (await c.post("/api/jobs/kb-mine")).status_code == 200
        r = await c.post("/api/jobs/kb-mine/stop")
        assert r.status_code == 200
        assert calls == [(4321, signal.SIGTERM)]  # killpg 连 make → uv → python 一起收
        # 已结束的作业再 stop → 409
        FakePopen.instances[0]._rc = 0
        assert (await c.post("/api/jobs/kb-mine/stop")).status_code == 409


async def test_status_and_log_tail(client):
    async with client as c:
        r0 = await c.get("/api/jobs/kb-build")
        assert r0.status_code == 200
        assert r0.json()["running"] is False and r0.json()["log_tail"] == ""

        assert (await c.post("/api/jobs/kb-build")).status_code == 200
        log_path = jobs.LOG_DIR / "kb-build.log"
        assert log_path.exists()  # 日志落 log/acceptance/<job>.log
        log_path.write_text("\n".join(f"line-{i}" for i in range(60)), encoding="utf-8")
        r = await c.get("/api/jobs/kb-build")
        d = r.json()
        assert d["running"] is True and d["pid"] == 4321
        tail_lines = d["log_tail"].splitlines()
        assert len(tail_lines) == 50  # 只回显尾部
        assert tail_lines[-1] == "line-59" and tail_lines[0] == "line-10"
        # 结束后状态带返回码
        FakePopen.instances[0]._rc = 2
        r2 = await c.get("/api/jobs/kb-build")
        assert r2.json()["running"] is False and r2.json()["exit_code"] == 2


async def test_log_is_overwritten_on_restart(client):
    async with client as c:
        await c.post("/api/jobs/kb-build")
        log_path = jobs.LOG_DIR / "kb-build.log"
        log_path.write_text("上一轮的旧日志", encoding="utf-8")
        FakePopen.instances[0]._rc = 0
        await c.post("/api/jobs/kb-build")
    assert log_path.read_text(encoding="utf-8") == ""  # 覆盖写,不追加
