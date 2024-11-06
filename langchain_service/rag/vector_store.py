import time
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
                        "similarity": "euclidean"
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
                        "path": "metadata.date",
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
        
        result = collection.create_search_index(search_index_model)
        print("New search index named " + result + " is building.")

        # Wait for initial sync to complete
        print("Polling to check if the index is ready. This may take up to a minute.")
        predicate=None
        if predicate is None:
            predicate = lambda index: index.get("queryable") is True

        while True:
            indices = list(collection.list_search_indexes(config.VECTOR_INDEX_NAME))
            if len(indices) and predicate(indices[0]):
                break
            time.sleep(5)
        print(result + " is ready for querying.")
        
    except pymongo.errors.OperationFailure as e:
        if "already exists" in str(e):
            print(f"Index '{config.VECTOR_INDEX_NAME}' already exists.")
        else:
            print(f"Error creating index: {str(e)}")
            raise