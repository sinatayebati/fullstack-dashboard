import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Langchain AI Service"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "timestamp" in response.json()

@pytest.mark.asyncio
async def test_query_invoice():
    response = client.post("/query", json={"question": "test question"})
    assert response.status_code == 200
    assert "question" in response.json()
    assert "answer" in response.json()
    assert "source_documents" in response.json()