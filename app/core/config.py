from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Streaming Memory Backend"
    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL"
    )
    default_model: str = Field(default="deepseek-chat", validation_alias="DEFAULT_MODEL")
    redis_url: str = Field(
        default="redis://localhost:6379/0", validation_alias="REDIS_URL"
    )
    app_api_keys: str = Field(default="", validation_alias="APP_API_KEYS")
    dev_bypass_auth: bool = Field(default=False, validation_alias="DEV_BYPASS_AUTH")
    memory_ttl_seconds: int = Field(
        default=3600, validation_alias="MEMORY_TTL_SECONDS"
    )
    memory_max_turns: int = Field(default=10, validation_alias="MEMORY_MAX_TURNS")
    rate_limit_per_minute: int = Field(
        default=30, validation_alias="RATE_LIMIT_PER_MINUTE"
    )
    max_concurrent_per_key: int = Field(
        default=2, validation_alias="MAX_CONCURRENT_PER_KEY"
    )
    lock_ttl_seconds: int = Field(default=30, validation_alias="LOCK_TTL_SECONDS")
    rag_docs_dir: str = Field(default="knowledge/dogs", validation_alias="RAG_DOCS_DIR")
    rag_index_path: str = Field(
        default="knowledge/index/dog_basic_index.json",
        validation_alias="RAG_INDEX_PATH",
    )
    rag_default_top_k: int = Field(default=3, validation_alias="RAG_DEFAULT_TOP_K")
    rag_chunk_size: int = Field(default=500, validation_alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(
        default=80, validation_alias="RAG_CHUNK_OVERLAP"
    )
    rag_score_threshold: float = Field(
        default=0.3, validation_alias="RAG_SCORE_THRESHOLD"
    )
    rag_answer_score_threshold: float = Field(
        default=0.42, validation_alias="RAG_ANSWER_SCORE_THRESHOLD"
    )
    rag_source_score_threshold: float = Field(
        default=0.54, validation_alias="RAG_SOURCE_SCORE_THRESHOLD"
    )
    rag_symptom_answer_score_threshold: float = Field(
        default=0.52, validation_alias="RAG_SYMPTOM_ANSWER_SCORE_THRESHOLD"
    )
    rag_symptom_source_score_threshold: float = Field(
        default=0.66, validation_alias="RAG_SYMPTOM_SOURCE_SCORE_THRESHOLD"
    )
    rag_embedding_mode: str = Field(
        default="local", validation_alias="RAG_EMBEDDING_MODE"
    )
    rag_embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        validation_alias="RAG_EMBEDDING_MODEL",
    )

    @property
    def api_keys_set(self) -> set[str]:
        return {part.strip() for part in self.app_api_keys.split(",") if part.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
