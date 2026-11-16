/* 后台共用工具:API 封装 / toast / 作业按钮与日志窗(日志尾按作业名缓存,重建不丢结论)。
 * 包在 IIFE 里,只暴露 window.Acceptance,不与宿主页的 $ / el 撞名。 */
(function () {
  "use strict";

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  async function api(path, opts) {
    const resp = await fetch(path, opts);
    let data = null;
    try { data = await resp.json(); } catch (_) { /* 非 JSON 响应 */ }
    if (!resp.ok) {
      const detail = data && data.detail;
      throw new Error(typeof detail === "string" ? detail
        : detail ? JSON.stringify(detail) : "请求失败(" + resp.status + ")");
    }
    return data;
  }
  const post = (path, body) => api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  function toast(msg, kind) {
    let box = document.getElementById("toast-box");
    if (!box) {
      box = document.createElement("div");
      box.id = "toast-box";
      document.body.appendChild(box);
    }
    const t = document.createElement("div");
    t.className = "toast" + (kind ? " " + kind : "");
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => t.remove(), kind === "err" ? 8000 : 4500);
  }

  function pill(text, kind) {
    return '<span class="pill ' + (kind || "mute") + '">' + esc(text) + "</span>";
  }

  function fmtTime(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    const p = n => String(n).padStart(2, "0");
    return p(d.getMonth() + 1) + "-" + p(d.getDate()) + " " + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  // ---------- 作业运行器 ----------
  // 页面只传作业名;命令在服务器白名单里写死。日志尾缓存:作业跑完触发整页刷新时,
  // 按钮与日志窗即便重建,最该读的结论也从缓存贴回来,不会被刷掉。
  const JOB_LABELS = {
    "kb-preview": "重跑 材料清单与切块预览",
    "kb-build": "重跑 离线建库",
    "kb-vectorize": "重跑 向量化作业",
    "kb-mine": "对话挖知识",
    "kb-reset": "重跑 清库重建",
    "seed-conv": "灌历史会话种子",
  };
  const logCache = {};   // name -> 最近一次日志尾
  const state = {};      // name -> {running, exit_code, started}

  function jobLogEl(name) { return document.querySelector('pre.log[data-job="' + name + '"]'); }

  function paintJob(name) {
    const st = state[name] || {};
    const btn = document.querySelector('button[data-job-start="' + name + '"]');
    const stopBtn = document.querySelector('button[data-job-stop="' + name + '"]');
    const dot = document.querySelector('span[data-job-dot="' + name + '"]');
    const log = jobLogEl(name);
    if (btn) btn.disabled = !!st.running;
    if (stopBtn) stopBtn.style.display = st.running ? "" : "none";
    if (dot) {
      dot.textContent = st.running ? "运行中…"
        : st.started ? (st.exit_code === 0 ? "已完成(exit 0)" : "已结束(exit " + st.exit_code + ")") : "";
      dot.style.color = st.running ? "" : (st.exit_code === 0 ? "var(--ok)" : "var(--err)");
    }
    if (log && logCache[name] != null) {
      log.textContent = logCache[name];
      log.scrollTop = log.scrollHeight;
    }
  }

  async function poll(name, onFinish) {
    let st;
    try { st = await api("/api/jobs/" + name); } catch (_) { return; }
    state[name] = st;
    if (st.log_tail) logCache[name] = st.log_tail;
    paintJob(name);
    if (st.running) {
      setTimeout(() => poll(name, onFinish), 1500);
    } else if (onFinish) {
      onFinish(name, st);
    }
  }

  async function start(name, heavy, onFinish) {
    const label = JOB_LABELS[name] || name;
    if (heavy) {
      let msg = "即将执行「" + label + "」(" + name + "),该作业较重,确认执行?";
      if (name === "kb-reset") {
        msg += "\n\n将清空知识库(MySQL 两表 + Milvus 集合)。若 Milvus 清理一时失败,双写可能短暂显示「对不上」,重跑一次清库或补向量即可恢复。";
      }
      if (name === "kb-mine") msg += "\n\n将调用聊天上游 LLM 抽取问答对。";
      if (!confirm(msg)) return;
    }
    try {
      state[name] = await post("/api/jobs/" + name);
      paintJob(name);
      poll(name, onFinish);
    } catch (e) {
      toast(name + ":" + e.message, "err");
    }
  }

  async function stop(name) {
    try { await post("/api/jobs/" + name + "/stop"); } catch (e) { toast(e.message, "err"); }
  }

  // 在容器里为每个作业名建「按钮 + 状态 + 日志窗」;页面刷新重建后日志从缓存贴回
  function mountJobs(container, names, opts) {
    opts = opts || {};
    const wrap = document.createElement("div");
    wrap.className = "jobs";
    wrap.innerHTML = names.map(name => {
      const heavy = name === "kb-mine" || name === "kb-reset";
      return '<div class="job">'
        + '<div class="job-head">'
        + '<button class="btn' + (heavy ? " danger" : "") + '" data-job-start="' + name + '">'
        + esc(JOB_LABELS[name] || name) + (heavy ? " ⚠" : "") + "</button>"
        + '<button class="btn mini danger" data-job-stop="' + name + '" style="display:none">停止</button>'
        + '<span class="running-dot" data-job-dot="' + name + '"></span>'
        + '<span class="job-desc">' + esc(name) + (heavy ? " · 重作业,需二次确认" : "") + "</span>"
        + "</div>"
        + '<pre class="log" data-job="' + name + '"></pre>'
        + "</div>";
    }).join("");
    container.appendChild(wrap);
    names.forEach(name => {
      wrap.querySelector('[data-job-start="' + name + '"]')
        .addEventListener("click", () => start(name, heavy(name), opts.onFinish));
      wrap.querySelector('[data-job-stop="' + name + '"]')
        .addEventListener("click", () => stop(name));
      if (logCache[name] != null) paintJob(name);
    });
    // 页面刷新/重建后恢复运行态:服务器说还在跑的就接着轮询
    api("/api/jobs").then(d => {
      (d.jobs || []).forEach(st => {
        if (!names.includes(st.name)) return;
        state[st.name] = st;
        paintJob(st.name);
        if (st.running) poll(st.name, opts.onFinish);
      });
    }).catch(() => {});
    return wrap;
  }
  function heavy(name) { return name === "kb-mine" || name === "kb-reset"; }

  window.Acceptance = { esc, api, post, toast, pill, fmtTime, mountJobs, logCache };
})();
