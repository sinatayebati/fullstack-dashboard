import os
import json
from dataclasses import dataclass
from typing import Optional

# Load environment variables
MONGODB_URI = os.getenv("MONGO_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class Config:
    DB_NAME: str = "langchain_db"
    COLLECTION_VECTOR: str = "invoice_db"
    COLLECTION_IMAGE: str = "image_bytes"
    VECTOR_INDEX_NAME: str = "invoice_vector_index"
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    MONGODB_TIMEOUT_MS: int = 5000
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    RETRIEVER_K: int = 10

def validate_environment():
    required_vars = {
        "MONGO_URL": MONGODB_URI,
        "OPENAI_API_KEY": OPENAI_API_KEY
    }
    missing = [var for var, val in required_vars.items() if not val]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Load data index
with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'index.json')) as f:
    DATA_INDEX = json.load(f) 