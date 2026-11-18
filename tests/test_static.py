"""静态页可达性:后台各页能打开且都引了共用外壳 admin.js(前端逻辑在浏览器里验)。"""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture()
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                       follow_redirects=True)


@pytest.mark.parametrize("path", ["/kb", "/admin", "/rag-eval",
                                  "/static/kb.html", "/static/admin.html", "/static/rageval.html"])
async def test_admin_pages_reachable_and_mount_shell(client, path):
    async with client as c:
        r = await c.get(path)
    assert r.status_code == 200
    assert "/static/admin.js" in r.text


@pytest.mark.parametrize("path", ["/static/admin.js", "/static/acceptance.css", "/static/acceptance.js"])
async def test_shared_assets_reachable(client, path):
    async with client as c:
        r = await c.get(path)
    assert r.status_code == 200


async def test_chat_page_has_admin_entry(client):
    async with client as c:
        r = await c.get("/static/index.html")
    assert r.status_code == 200
    assert "后台管理" in r.text
    assert 'href="/admin"' in r.text
