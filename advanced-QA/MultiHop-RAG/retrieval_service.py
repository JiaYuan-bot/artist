from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
from contextlib import asynccontextmanager

from faiss_service import SimpleRetrievalServiceFaiss

# Global retrieval service instance
retrieval_service: Optional[SimpleRetrievalServiceFaiss] = None


# Request/Response models
class RetrievalRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: Optional[int] = Field(None,
                                 description="Number of results to return")


class BatchRetrievalRequest(BaseModel):
    queries: List[str] = Field(..., description="List of search queries")
    top_k: Optional[int] = Field(None,
                                 description="Number of results per query")


class RetrievalResult(BaseModel):
    text: str
    score: float
    metadata: Dict


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
    count: int


class BatchRetrievalResponse(BaseModel):
    results: List[List[RetrievalResult]]
    count: int


class HealthResponse(BaseModel):
    status: str
    message: str


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the retrieval service
    global retrieval_service

    print("Initializing retrieval service...")

    PROJECT_ID = "dbgroup"
    CORPUS_PATH = "/home/jiayuan/nl2sql/MultiHop-RAG/dataset/corpus.json"

    retrieval_service = SimpleRetrievalServiceFaiss(
        corpus_path=CORPUS_PATH,
        project_id=PROJECT_ID,
        location="us-central1",
        model_name="gemini-embedding-001",
        chunk_size=512,
        top_k=5,
        persist_dir="./li_store",
        faiss_index_path="./faiss.index",
        distance="ip",
    )

    print("Retrieval service ready!")

    yield

    # Shutdown: Cleanup if needed
    print("Shutting down retrieval service...")


# Initialize FastAPI app
app = FastAPI(
    title="FAISS Retrieval Service",
    description=
    "High-performance retrieval API using FAISS and Vertex AI embeddings",
    version="1.0.0",
    lifespan=lifespan)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "FAISS Retrieval Service is running"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {"status": "healthy", "message": "Retrieval service is operational"}


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    Retrieve relevant documents for a single query
    
    - **query**: Search query text
    - **top_k**: Number of results to return (optional, defaults to service default)
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        results = retrieval_service.retrieve(query=request.query,
                                             top_k=request.top_k)

        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Retrieval failed: {str(e)}")


@app.post("/batch_retrieve", response_model=BatchRetrievalResponse)
async def batch_retrieve(request: BatchRetrievalRequest):
    """
    Retrieve relevant documents for multiple queries in batch
    
    - **queries**: List of search query texts
    - **top_k**: Number of results per query (optional)
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not request.queries:
        raise HTTPException(status_code=400,
                            detail="Queries list cannot be empty")

    try:
        results = retrieval_service.batch_retrieve(queries=request.queries,
                                                   top_k=request.top_k)

        return {"results": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Batch retrieval failed: {str(e)}")


@app.post("/rebuild")
async def rebuild_index():
    """
    Manually rebuild the FAISS index from corpus
    Warning: This will overwrite the existing index
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        retrieval_service.rebuild()
        return {"status": "success", "message": "Index rebuilt successfully"}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Rebuild failed: {str(e)}")


if __name__ == "__main__":
    # Run with uvicorn
    # For high concurrency, increase workers in production
    uvicorn.run(
        "retrieval_service:app",
        host="0.0.0.0",
        port=8008,
        workers=50,  # Use multiple workers for production
        log_level="info")
