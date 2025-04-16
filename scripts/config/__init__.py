import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "200"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
DB_DIR = os.getenv("DB_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pytorch_docs")
INDEXED_FILE = "./data/indexed_chunks.json"