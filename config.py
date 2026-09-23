from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    auth_key: str

    gemini_api: str
    gemini_model: str = "gemini-flash-lite-latest"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api: str | None = None
    collection_name: str = "book"

    embed_model: str = "BAAI/bge-base-en-v1.5"
    dims: int = 768

    chunk_size: int = 250   
    chunk_overlap: int = 50

    top_k: int = 3

settings = Settings()
