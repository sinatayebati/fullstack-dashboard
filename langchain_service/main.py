from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import asyncio
import logging
from datetime import datetime
from rag.invoice_rag import initialize_rag_pipeline, query_rag

load_dotenv()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store the RAG pipeline and MongoDB client
rag_chain = None
retriever = None
mongo_client = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Langchain AI Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/query")
async def query_invoice(question: str):
    global rag_chain, retriever
    if rag_chain is None or retriever is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    try:
        answer, source_documents = query_rag(rag_chain, retriever, question)
        return {
            "question": question,
            "answer": answer,
            "source_documents": [doc.page_content for doc in source_documents]
        }
    except Exception as e:
        logger.error(f"Error querying RAG pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

async def periodic_health_check():
    while True:
        logger.info(f"Health Check: Service is healthy at {datetime.now().isoformat()}")
        await asyncio.sleep(300)  # Sleep for 300 seconds (5 minutes)

@app.on_event("startup")
async def startup_event():
    global rag_chain, retriever, mongo_client
    rag_chain, retriever, mongo_client = initialize_rag_pipeline()
    asyncio.create_task(periodic_health_check())

@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client:
        mongo_client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
