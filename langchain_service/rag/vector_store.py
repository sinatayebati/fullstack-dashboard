import pymongo
from typing import List
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config.config import Config

def check_or_create_vector_index(collection, config: Config):
    """Enhanced index creation with better error handling"""
    try:
        if collection.count_documents({}) == 0:
            print("Collection is empty. Load data before creating index.")
            return
            
        # Get all search indexes
        existing_indexes = list(collection.list_search_indexes())
        existing_index_names = [index.get('name') for index in existing_indexes]
        print(f"Existing index names: {existing_index_names}")
        
        if config.VECTOR_INDEX_NAME in existing_index_names:
            print(f"Vector index '{config.VECTOR_INDEX_NAME}' already exists. Skipping creation.")
            return
            
        print(f"Creating new vector index '{config.VECTOR_INDEX_NAME}'...")
        
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1536,
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "metadata.id",
                    },
                    {
                        "type": "filter",
                        "path": "metadata.invoice_no",
                    },
                    {
                        "type": "filter",
                        "path": "metadata.seller",
                    },
                    {
                        "type": "filter",
                        "path": "metadata.client",
                    }
                ]
            },
            name=config.VECTOR_INDEX_NAME,
            type="vectorSearch"
        )
        
        collection.create_search_index(search_index_model)
        print(f"Vector index '{config.VECTOR_INDEX_NAME}' created successfully.")
        
    except pymongo.errors.OperationFailure as e:
        if "already exists" in str(e):
            print(f"Index '{config.VECTOR_INDEX_NAME}' already exists.")
        else:
            print(f"Error creating index: {str(e)}")
            raise



def create_or_load_vector_store(collection, documents: List[Document], config: Config):
    """Modified vector store creation with proper text key handling"""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
            chunk_size=8000
        )
        
        # Check if collection exists and has documents
        doc_count = collection.count_documents({})
        
        if doc_count > 0:
            print(f"Found existing collection with {doc_count} documents. Loading vector store...")
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name=config.VECTOR_INDEX_NAME,
                embedding_key="embedding",
                text_key="text",
                metadata_key="metadata"
            )
            return vector_store
            
        else:
            print("Creating new vector store from documents...")
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=embeddings,
                collection=collection,
                index_name=config.VECTOR_INDEX_NAME
            )
            print(f"Vector store created with index '{config.VECTOR_INDEX_NAME}' and {len(documents)} documents.")
            return vector_store
        
    except Exception as e:
        print(f"Error in vector store operation: {str(e)}")
        raise