"""
PDF Knowledge Assistant with Local Embeddings
"""

# core.py
import logging
import sys
import os
from typing import Optional, List
from dotenv import load_dotenv
import typer
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder import Embedder
from sentence_transformers import SentenceTransformer
from pydantic import Field, BaseModel
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("knowledge_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AppConfig(BaseModel):
    """Application configuration model"""
    db_url: str = Field(
        default="postgresql+psycopg://ai:ai@localhost:5532/ai",
        description="PostgreSQL connection URL"
    )
    collection_name: str = Field(
        default="science_docs",
        description="Vector database collection name"
    )
    embedder_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Hugging Face sentence transformer model name"
    )
    embedder_dim: int = Field(
        default=384,
        description="Dimension of embedding vectors"
    )
    table_name: str = Field(
        default="science_assistant",
        description="Database table name for assistant storage"
    )

config = AppConfig()

class HfEmbedder(Embedder):
    """Custom Hugging Face embedding model"""
    model_name: str = Field(default=config.embedder_model, frozen=True)
    _model: SentenceTransformer = None

    @property
    def model(self):
        """Lazy load the sentence transformer model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embedding_and_usage(self, text: str):
        """Generate embeddings with local model"""
        try:
            embedding = self.model.encode(text).tolist()
            if len(embedding) != config.embedder_dim:
                logger.error(f"Embedding dimension mismatch: Expected {config.embedder_dim}, got {len(embedding)}")
                raise ValueError("Embedding dimension mismatch")
            return embedding, {"model": self.model_name, "usage": {"total_tokens": 0}}
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

def setup_database():
    """Initialize PostgreSQL database with vector extension"""
    try:
        logger.info("Initializing database connection")
        engine = sa.create_engine(config.db_url)
        with engine.connect() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        logger.info("Database extension setup completed")
    except SQLAlchemyError as e:
        logger.error(f"Database setup failed: {str(e)}")
        sys.exit(1)

def create_vector_db():
    """Create PGVector instance with proper configuration"""
    return PgVector2(
        collection=config.collection_name,
        db_url=config.db_url,
        embedder=HfEmbedder(),
        dim=config.embedder_dim  # Critical dimension setting
    )

def create_knowledge_base(vector_db: PgVector2):
    """Create and load PDF knowledge base"""
    try:
        logger.info("Creating knowledge base")
        knowledge_base = PDFUrlKnowledgeBase(
            urls=["https://www.nobelprize.org/uploads/2024/10/advanced-chemistryprize2024.pdf"],
            vector_db=vector_db
        )
        logger.info("Loading documents into vector database")
        knowledge_base.load(recreate=False)
        return knowledge_base
    except Exception as e:
        logger.error(f"Knowledge base creation failed: {str(e)}")
        sys.exit(1)

def main(new_session: bool = False, user_id: str = "default_user"):
    """Main application workflow"""
    setup_database()
    
    # Initialize components
    vector_db = create_vector_db()
    knowledge_base = create_knowledge_base(vector_db)
    storage = PgAssistantStorage(
        table_name=config.table_name,
        db_url=config.db_url
    )

    # Manage session state
    run_id: Optional[str] = None
    if not new_session:
        existing_runs = storage.get_all_run_ids(user_id)
        if existing_runs:
            run_id = existing_runs[0]
            logger.info(f"Resuming existing session: {run_id}")

    # Initialize assistant
    assistant = Assistant(
        run_id=run_id,
        user_id=user_id,
        knowledge_base=knowledge_base,
        storage=storage,
        tools=[],
        show_tool_calls=True,
        read_chat_history=True,
    )

    # Start CLI
    logger.info(f"Starting assistant session: {assistant.run_id}")
    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(main)