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

settings = Settings()
