import pathlib

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings

_DDL_FILES = [
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch03-ddl.sql",
]
# 删除顺序:先子表后父表;knowledge_chunks 自引用 FK 靠 FOREIGN_KEY_CHECKS=0 兜
_TABLES = ["messages", "tickets", "conversations", "faq",
           "qa_extraction_staging", "knowledge_chunks"]


def _create_table_stmts() -> list[str]:
    stmts: list[str] = []
    for ddl in _DDL_FILES:
        raw = ddl.read_text(encoding="utf-8")
        sql = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("--"))
        stmts += [s.strip() for s in sql.split(";") if s.strip() and "CREATE TABLE" in s.upper()]
    return stmts

# asyncmy 连接绑定创建它的事件循环:DB 相关 fixture 与测试统一跑在 session 级 loop
# (DB 测试文件需声明 pytestmark = pytest.mark.asyncio(loop_scope="session"))。
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _test_engine():
    # 用 server-url(不带库名)建 mewhelp_test 库
    server_url = settings.test_database_url.rsplit("/", 1)[0]
    admin = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text("DROP DATABASE IF EXISTS mewhelp_test"))
        await conn.execute(text("CREATE DATABASE mewhelp_test CHARACTER SET utf8mb4"))
    await admin.dispose()

    engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
    # 跑 DDL 建表(以 sql/ch02-ddl.sql、sql/ch03-ddl.sql 为权威;按 ; 分割只跑 CREATE TABLE 句)
    async with engine.begin() as conn:
        for s in _create_table_stmts():
            await conn.execute(text(s))
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture(loop_scope="session")
async def db_session_factory(_test_engine, monkeypatch):
    """把 repository 用到的 async_session 指向测试库。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory

@pytest_asyncio.fixture(loop_scope="session")
async def db_clean(_test_engine):
    """每个测试后清空全部表(关外键检查以便 TRUNCATE)。"""
    yield
    async with _test_engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in _TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {t}"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
