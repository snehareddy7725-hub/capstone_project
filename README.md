# Support Assistant Module

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

## Architecture
The RAG pipeline consists of four stages:

### 1. Ingestion
- Location: `load_documents()` function
- Loads 8 policy documents from `/docs` directory
- Each document is a separate text file with Zepto policies

### 2. Embedding
- Location: `create_embeddings()` function
- Uses `all-MiniLM-L6-v2` model from sentence-transformers
- Embeds each document chunk and stores in ChromaDB

### 3. Retrieval
- Location: `retrieve_and_answer()` node in LangGraph
- Uses ChromaDB similarity search with cosine distance
- Retrieves top-3 most relevant chunks for policy queries

### 4. Generation
- Location: `retrieve_and_answer()` and `direct_answer()` nodes
- MOCK_LLM=1 (default): Uses deterministic template responses
- MOCK_LLM=0: Uses Groq API for real LLM responses

## Example Queries

### Policy Query (triggers retrieval)