import time
import pymongo
import numpy as np
from typing import List
from bson.binary import Binary, BinaryVectorDtype
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.config import Config
from .ingest import generate_bson_vector
    

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
                        "path": "embedding_float",
                        "numDimensions": 1536,
                        "similarity": "euclidean"
                    },
                    {
                        "type": "vector",
                        "path": "embedding_int8",
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



async def vector_search(collection, query_text: str, config: Config, use_int8: bool = False):
    """Perform vector search using BSON vectors"""
    try:
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query_text)
        query_embedding_np = np.array(query_embedding, dtype=np.int8 if use_int8 else np.float32)
        
        # Convert to BSON
        bson_query_vector = generate_bson_vector(
            query_embedding_np,
            BinaryVectorDtype.INT8 if use_int8 else BinaryVectorDtype.FLOAT32
        )
        
        # Create search pipeline
        pipeline = [
            {
                '$vectorSearch': {
                    'index': config.VECTOR_INDEX_NAME,
                    'path': 'embedding_int8' if use_int8 else 'embedding_float',
                    'queryVector': bson_query_vector,
                    'numCandidates': 100,
                    'limit': 5
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'page_content': 1,
                    'metadata': 1,
                    'score': { '$meta': 'vectorSearchScore' }
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return results
        
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        raise