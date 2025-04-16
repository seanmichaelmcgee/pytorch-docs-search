import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config")

# Load environment variables (but prioritize existing environment variables)
load_dotenv()

# Get OpenAI API key directly from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-3-large"  # Upgraded from text-embedding-ada-002
EMBEDDING_DIMENSIONS = 3072  # Dimensions for text-embedding-3-large (was 1536 for ada-002)
EMBEDDING_BATCH_SIZE = 10  # Reduced from 20 due to larger embedding size

# Embedding cache configuration
EMBEDDING_CACHE_DIR = "./data/embedding_cache"
EMBEDDING_CACHE_MAX_SIZE_GB = 1.0  # Maximum cache size in GB

# Document processing configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "200"))

# Search configuration
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))

# Database configuration
DB_DIR = os.getenv("DB_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pytorch_docs")
INDEXED_FILE = "./data/indexed_chunks.json"