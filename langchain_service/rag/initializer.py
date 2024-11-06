import os
from typing import Optional

from config.config import Config, DATA_INDEX
from mongodb.mongodb_client import connect_to_mongodb, load_data_to_mongodb, load_image_bytes_to_mongodb
from .vector_store import check_or_create_vector_index, create_or_load_vector_store
from .rag_chain import setup_rag_pipeline

def initialize_rag_pipeline(config: Optional[Config] = None):
    if config is None:
        config = Config()
        
    # 1. Connect to MongoDB
    client, db, collection = connect_to_mongodb(config)
    
    # 2. Check if collection already has documents
    doc_count = collection.count_documents({})
    documents = []
    
    if doc_count == 0:
        print("Collection is empty. Loading documents...")
        # Only load documents if collection is empty
        data_file = os.path.join(os.path.dirname(__file__), '..', DATA_INDEX['invoice_data']['path'])
        documents = load_data_to_mongodb(collection, data_file, config)
        
        # Load image data separately
        total_image_documents = load_image_bytes_to_mongodb(db, data_file, config)
    else:
        print(f"Collection already contains {doc_count} documents. Skipping document loading.")
    
    # 3. Create or load vector store
    vector_store = create_or_load_vector_store(collection, documents, config)
    
    # 4. Ensure vector index exists
    check_or_create_vector_index(collection, config)
    
    # 5. Setup RAG pipeline
    rag_chain, retriever = setup_rag_pipeline(vector_store, config)
    
    return rag_chain, retriever, client