# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    DB_URL: str = "postgresql+psycopg://ai:ai@localhost:5532/ai"
    EMBEDDER_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    COLLECTION_NAME: str = "science_docs"
    TABLE_NAME: str = "science_assistant"
    
    class Config:
        env_file = ".env"

settings = Settings()