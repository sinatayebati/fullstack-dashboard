from fastapi import FastAPI
from dotenv import load_dotenv
import asyncio
import logging
from datetime import datetime

load_dotenv()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"message": "Welcome to the Langchain AI Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

async def periodic_health_check():
    while True:
        logger.info(f"Health Check: Service is healthy at {datetime.now().isoformat()}")
        await asyncio.sleep(300)  # Sleep for 300 seconds (5 minutes)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_health_check())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
