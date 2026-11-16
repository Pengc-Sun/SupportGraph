# MewHelp — 电商智能客服(Ch01 纯对话 + Ch02 工具链)

FastAPI + LangChain,模型上游直连。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY(硅基流动)
    make dev               # :8000 应用

## 验收

    # 1. 流式对话
    curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"user_id": "u1", "message": "你们卖猫粮吗?"}'

    # 2. 两轮上下文
    ./scripts/demo_chat.sh

    # 3. 结构化提取
    curl -s http://localhost:8000/api/extract -H 'Content-Type: application/json' \
      -d '{"text": "订单 MH20260701123 的猫爬架散架了,我要退款"}'

## Ch02 工具链

    # 启动
    docker compose up -d   # MySQL 8,首启自动跑 sql/ch02-ddl.sql + ch02-seed.sql
    make seed              # 已有容器时手动灌数(幂等)
    make dev               # :8000 应用

浏览器打开 `http://localhost:8000`(自动跳聊天页),问「订单 1001 的物流到哪了」,
可看到工具徽章 + 流式回答。

    # 评估/演示(需服务运行中 + 真实 key)
    make eval-agent        # 标注样例评估:期望工具 vs 实际选中工具
    ./scripts/demo_agent.sh  # 三条验收标准演示(走 /api/agent 非流式 JSON)

## 测试

    make test   # 单测(不打真实模型);现含 DB 测试,需 Docker MySQL 运行中
    make eval   # 标注样例评估(需服务运行中)
