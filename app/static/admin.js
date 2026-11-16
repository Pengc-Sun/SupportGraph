/* 后台共用导航外壳:mountAdminNav(active)。
 * IIFE + 自带样式注入(类名全带 adm- 前缀),挂在样式内联的旧页上也不打架。
 * 两层导航:模块行(各模块保持原路径)+ 当前模块的子页行。
 * 后面新增一块 = 往 MODULES 注册表加一行。 */
(function () {
  "use strict";

  const MODULES = [
    { key: "admin", title: "后台首页", path: "/admin", subs: [] },
    { key: "kb", title: "知识库", path: "/kb", subs: [
      { title: "① 手工录入", href: "#ingest" },
      { title: "② 建库材料", href: "#sources" },
      { title: "③ 对话挖知识", href: "#mining" },
      { title: "④ 向量化与双写", href: "#dualwrite" },
      { title: "⑤ 检索自测", href: "#search" },
      { title: "⑥ 最近入库", href: "#recent" },
    ] },
    { key: "chat", title: "聊天页", path: "/", subs: [] },
  ];

  const CSS = `
  .adm-nav { background:#fff; border-bottom:1px solid #e6e8eb; font-size:13px;
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }
  .adm-nav .adm-row { max-width:1180px; margin:0 auto; padding:0 20px;
    display:flex; align-items:center; gap:4px; flex-wrap:wrap; }
  .adm-nav .adm-brand { font-weight:700; color:#1f2329; padding:10px 12px 10px 0; font-size:14px; }
  .adm-nav a.adm-item { color:#8a919f; text-decoration:none; padding:10px 12px;
    border-bottom:2px solid transparent; }
  .adm-nav a.adm-item:hover { color:#ff7a45; }
  .adm-nav a.adm-item.adm-active { color:#ff7a45; border-bottom-color:#ff7a45; font-weight:600; }
  .adm-nav .adm-subrow { border-top:1px dashed #e6e8eb; }
  .adm-nav .adm-subrow a.adm-item { padding:6px 10px; font-size:12px; border-bottom:none; }
  .adm-nav .adm-subrow a.adm-item:hover { color:#ff7a45; }
  .adm-nav .adm-spacer { flex:1; }
  `;

  function mountAdminNav(active) {
    if (document.querySelector(".adm-nav")) return;
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const nav = document.createElement("nav");
    nav.className = "adm-nav";
    const row = document.createElement("div");
    row.className = "adm-row";
    row.innerHTML = '<span class="adm-brand">🐱 喵喵优选 · 后台</span>'
      + MODULES.map(m =>
        '<a class="adm-item' + (m.key === active ? " adm-active" : "") + '" href="' + m.path + '">'
        + m.title + "</a>").join("")
      + '<span class="adm-spacer"></span>';
    nav.appendChild(row);

    const current = MODULES.find(m => m.key === active);
    if (current && current.subs.length) {
      const sub = document.createElement("div");
      sub.className = "adm-row adm-subrow";
      sub.innerHTML = current.subs.map(s =>
        '<a class="adm-item" href="' + s.href + '">' + s.title + "</a>").join("");
      nav.appendChild(sub);
    }
    document.body.insertBefore(nav, document.body.firstChild);
  }

  window.mountAdminNav = mountAdminNav;
})();
