import pytest
import os
from unittest.mock import Mock, patch
from invoice_rag import (
    Config, 
    connect_to_mongodb, 
    check_or_create_vector_index,
    load_data_to_mongodb,
    query_rag
)

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def mock_mongodb_client():
    with patch('pymongo.MongoClient') as mock_client:
        mock_db = Mock()
        mock_collection = Mock()
        mock_client.return_value = Mock(
            server_info=Mock(return_value=True),
            __getitem__=Mock(return_value=mock_db)
        )
        mock_db.__getitem__.return_value = mock_collection
        yield mock_client

def test_connect_to_mongodb(mock_mongodb_client, config):
    client, db, collection = connect_to_mongodb(config)
    assert client is not None
    assert db is not None
    assert collection is not None
    mock_mongodb_client.assert_called_once_with(
        os.getenv("MONGO_URL"), 
        serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS
    )

def test_check_or_create_vector_index(mock_mongodb_client, config):
    collection = mock_mongodb_client()[config.DB_NAME][config.COLLECTION_NAME]
    collection.list_search_indexes.return_value = []
    check_or_create_vector_index(collection, config)
    collection.create_search_index.assert_called_once()

@pytest.mark.asyncio
async def test_query_rag():
    mock_rag_chain = Mock()
    mock_retriever = Mock()
    mock_rag_chain.invoke.return_value = "Test answer"
    mock_retriever.get_relevant_documents.return_value = []
    
    result = query_rag(
        rag_chain=mock_rag_chain,
        retriever=mock_retriever,
        question="test question",
        config=Config()
    )
    
    assert result['answer'] == "Test answer"
    assert result['confidence'] == "low"