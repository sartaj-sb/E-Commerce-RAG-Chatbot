import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Project Paths
PDF_FOLDER = "data"
CHROMA_DB = "chroma_db"

# Embedding Model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
TOP_K = 3

# Groq Model
LLM_MODEL = "llama-3.3-70b-versatile"

# Generation Settings
TEMPERATURE = 0