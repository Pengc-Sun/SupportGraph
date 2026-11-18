# 知行客服 — 电商智能客服系统

一个按十天课程逐章演进的电商客服项目:从纯对话,到工具链,到知识库语义检索,再到混合检索 + 重排 + 可量化的 RAG 评估闭环。FastAPI + LangChain,模型上游直连(聊天走 DeepSeek 官方,嵌入/重排走硅基流动),前端零依赖原生 HTML/CSS/JS。

> 仓库名 `SupportGraph`,产品名 **知行客服**。聊天助手昵称「知行」。

## 功能概览(逐章)

| 章 | 主题 | 能力 |
|----|------|------|
| Ch01 | 纯对话 | SSE 流式对话、多轮上下文 + token 预算裁剪、售后工单结构化提取 |
| Ch02 | 工具链 | Agent 编排(query_order / query_product / query_logistics / query_faq / create_ticket)、MySQL 持久化、工具徽章 |
| Ch03 | 知识库 | Markdown 切块(标题/递归/句末重叠/表格按行)、双写 MySQL + Milvus、对话挖知识、语义检索、录入后台 |
| Ch04 | 混合检索 + 评估 | Milvus Standalone dense + BM25 双向量、四策略检索(向量/BM25/混合 RRF/混合+重排)、query 理解改写、生成前自评闸、编号引用、300 题五桶评估、编造个案台账、裁判校准 |

## 技术栈

- **后端**:FastAPI、LangChain、SQLAlchemy(async)+ asyncmy、pymilvus
- **存储**:MySQL 8(业务/知识库/会话)、Milvus Standalone 2.5.4(向量 + BM25)
- **上游**:DeepSeek(聊天/生成/裁判,OpenAI 兼容)、硅基流动 `BAAI/bge-m3`(嵌入)、`BAAI/bge-reranker-v2-m3`(重排)
- **前端**:原生 HTML/CSS/JS,零依赖 markdown 渲染、纯 CSS/SVG 图表、SSE 流式
- **工程**:uv 管理依赖、pytest(282 单测)、docker compose、Makefile 作业白名单

## 快速开始

```bash
cp .env.example .env        # 填入 CHAT_API_KEY(DeepSeek 官方 platform.deepseek.com)
docker compose up -d        # MySQL 8 + Milvus 三件套(etcd/minio/standalone),首启自动跑 sql/ch02-ddl.sql
make seed                   # 已有容器时手动灌业务种子数据(幂等)
make dev                    # 启动应用 :8000
```

浏览器打开 `http://localhost:8000`(自动跳聊天页)。

| 路由 | 页面 |
|------|------|
| `/` | 聊天页(流式回答 + 工具徽章 + 可点引用角标 + 👍/👎 反馈) |
| `/admin` | 后台管理外壳(各模块健康卡) |
| `/kb` | 知识库录入(预览切块/重复标记/暂存/向量化) |
| `/rag-eval` | RAG 评估报告(四策略对照 + 编造个案台账 + 就地重跑) |

> 知识库语义检索与 RAG 评估需在 `.env` 补 `EMBED_API_KEY`、`RERANK_API_KEY`(硅基流动),并先建库(见下)。仅聊天/工具链只需 `CHAT_API_KEY`。

## 分章演示与验收

### Ch01 纯对话 / 结构化提取

```bash
curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"user_id": "u1", "message": "你们卖猫粮吗?"}'          # 流式对话
./scripts/demo_chat.sh                                        # 两轮上下文
curl -s http://localhost:8000/api/extract -H 'Content-Type: application/json' \
  -d '{"text": "订单 MH20260701123 的猫爬架散架了,我要退款"}'   # 结构化提取
make eval                                                     # 提取标注样例评估
```

### Ch02 工具链

```bash
make eval-agent           # 标注样例:期望工具 vs 实际选中工具
./scripts/demo_agent.sh   # 三条验收标准演示(/api/agent 非流式 JSON)
```

聊天页问「订单 1001 的物流到哪了」,可见工具徽章 + 流式回答。

### Ch03 知识库(需 EMBED_API_KEY)

```bash
make kb-build       # Markdown 切块入 MySQL(pending)
make kb-vectorize   # 嵌入 BAAI/bge-m3 → Milvus(需 key)
make kb-preview     # 盘点切块/特性
make seed-conv      # 灌合成对话
make kb-mine        # 对话挖知识(需 CHAT key)
make eval-retrieval # 语义召回标注样例(需 key + 建库)
make eval-mining    # 挖知识标注样例
```

### Ch04 混合检索 + RAG 评估(需 EMBED_API_KEY + RERANK_API_KEY)

```bash
make smoke-rag      # Milvus BM25 冒烟 + rerank 冒烟(rerank 需 key)
make eval-check     # 评估集自检:300 题五桶、要点逐字、id 唯一(0 错)
make eval-rag       # 四策略 × 五桶评估 → data/ch04/reports/rag_eval.{json,txt}(分钟级)
make judge-check    # 裁判校准:台账已处置个案原样重放,量与人工一致率
```

评估报告在 `/rag-eval` 页可视化:检索 Recall@5 / MRR、证据覆盖度、答案覆盖度、忠实度、拒答率、编造个案台账(跨轮累计 + 处置状态 + 证据快照)。

## 测试

```bash
make test    # uv run pytest,282 passed(需 Docker MySQL 运行中)
```

DB 测试连 `mewhelp_test` 库;Milvus 测试用 Standalone 唯一临时集合 + teardown;嵌入/重排/LLM 全 mock,真实链路留冒烟与评估。确定性代码走 TDD,纯 Prompt/数据任务走标注样例验证,前端走浏览器实测。

## 项目结构

```
app/
  api/        chat / agent / extract / kb / jobs / admin / rageval 路由
  core/       llm 工厂 / prompts / retrieval / rerank / query_understanding / selfcheck / jobs / chunking
  db/         models / repository / base(async engine)
  kb/         milvus_client / dualwrite / dedup / documents / sources / mining
  static/     index / admin / kb / rageval 页 + acceptance.css/js + admin.js
  tools/      业务工具 + 基础设施
data/kb/      知识库 Markdown(含易混型号族规格手册)
sql/          ch02/ch03/ch04 权威 DDL + 种子
scripts/      建库/向量化/挖知识/各章评估/冒烟/演示
tests/        282 单测 + 评估集 data/eval_ch04.jsonl(300 题)
dev-notes/    逐章开发留痕(ch01–ch04)
```

## 配置说明

`.env`(gitignored,勿提交;仓库 Public)关键项见 `.env.example`:

- `CHAT_BASE_URL` / `CHAT_MODEL` / `CHAT_API_KEY`:DeepSeek 官方,模型名写上游真名(`deepseek-flash`)
- `EMBED_API_KEY` / `RERANK_API_KEY`:硅基流动,嵌入/重排;BASE_URL 有内置默认
- `DATABASE_URL` / `TEST_DATABASE_URL`:MySQL(`mewhelp` / `mewhelp_test`)
- `TOKEN_BUDGET`:多轮上下文 token 预算

> 内部基础设施标识(数据库名 `mewhelp`、容器/卷 `mewhelp-*`、localStorage 键)为环境绑定标识,与产品品牌名「知行客服」分离,改名不影响运行。

## 开发留痕

逐章设计决策、Ruling、验收进展见 [`dev-notes/`](dev-notes/)(ch01–ch04)。ch03/ch04 的真实上游建库与评估验收待嵌入/重排 key 配齐后跑通。
