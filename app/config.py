from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 这三项故意不给默认值,缺了在启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000
    database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp_test"
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = ""                 # 真实 key 在 .env 填 EMBED_API_KEY;空值时冒烟/嵌入调用会失败
    embed_model: str = "BAAI/bge-m3"        # 上游真实名,没有别名这一层
    milvus_uri: str = "data/milvus_knowledge.db"
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.4

settings = Settings()
