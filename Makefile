.PHONY: dev test eval eval-agent seed kb-build kb-vectorize kb-mine eval-retrieval seed-conv eval-mining kb-preview kb-reset milvus-up milvus-down smoke-rag eval-rag

dev:
	./scripts/dev.sh

test:
	uv run pytest -v

eval:
	uv run python scripts/eval_extract.py

seed:
	docker compose exec -T mysql mysql -uroot -proot mewhelp < sql/ch02-seed.sql

eval-agent:
	uv run python scripts/eval_agent.py

kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py

seed-conv:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch03-seed.sql

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py

kb-preview:
	PYTHONPATH=. uv run python scripts/show_kb.py

kb-reset:
	PYTHONPATH=. uv run python scripts/reset_kb.py

milvus-up:
	docker-compose up -d etcd minio milvus-standalone
	@echo "等待 Milvus 就绪(healthz)..."; \
	for i in $$(seq 1 60); do \
	  curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && echo "Milvus OK" && exit 0; \
	  sleep 3; done; echo "Milvus 未就绪" && exit 1

milvus-down:
	docker-compose stop etcd minio milvus-standalone

smoke-rag:
	PYTHONPATH=. uv run python scripts/smoke_milvus_bm25.py
	PYTHONPATH=. uv run python scripts/smoke_rerank.py

eval-rag:
	PYTHONPATH=. uv run python scripts/eval_ch04.py
