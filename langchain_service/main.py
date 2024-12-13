from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import logging
from datetime import datetime
from config.config import Config
from rag.initializer import initialize_rag_pipeline
from rag.rag_chain import query_rag
from rag.vector_store import vector_search


load_dotenv()
config = Config()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store the RAG pipeline and MongoDB client
rag_chain = None
retriever = None
mongo_client = None


class QuestionRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    confidence: str
    source_documents: list[str]

class VectorSearchRequest(BaseModel):
    query: str
    filters: dict | None = None

class VectorSearchResponse(BaseModel):
    results: list


@app.get("/")
async def root():
    return {"message": "Welcome to the Langchain AI Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    
@app.post("/query", response_model=QueryResponse)
async def query_invoice(request: QuestionRequest):
    global rag_chain, retriever, config
    if rag_chain is None or retriever is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    try:
        result = query_rag(rag_chain, retriever, request.question, config)
        return {
            "question": request.question,
            "answer": result['answer'],
            "sources": result['sources'],
            "confidence": result['confidence'],
            "source_documents": [doc.page_content for doc in result['source_documents']]
        }
    except Exception as e:
        logger.error(f"Error querying RAG pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

async def periodic_health_check():
    while True:
        logger.info(f"Health Check: Service is healthy at {datetime.now().isoformat()}")
        await asyncio.sleep(300)  # Sleep for 300 seconds (5 minutes)


@app.post("/vector-search")
async def vector_search_endpoint(request: VectorSearchRequest):
    global mongo_client, config
    if mongo_client is None:
        raise HTTPException(status_code=503, detail="MongoDB client not initialized")
    try:
        # Get the vector collection
        db = mongo_client[config.DB_NAME]
        collection = db[config.COLLECTION_VECTOR]
        
        # Perform vector search
        results = await vector_search(
            collection=collection,
            query_text=request.query,
            config=config,
            filters=request.filters
        )
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    global rag_chain, retriever, mongo_client
    rag_chain, retriever, mongo_client = initialize_rag_pipeline(config)
    asyncio.create_task(periodic_health_check())

@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client:
        mongo_client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
