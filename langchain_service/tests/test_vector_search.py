import sys
import os
from pymongo import MongoClient
from config.config import Config
from rag.vector_store import vector_search

async def test_vector_search():
    # Initialize configuration
    config = Config()
    
    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client[config.DB_NAME]
    collection = db[config.COLLECTION_VECTOR]
    
    # Test queries
    test_queries = [
        "Invoice Number: 39280409 | Date: 07/06/2014 | Seller: Davis PLC 72057 Castillo Via Deniseshire, KY 95233",
    ]
    
    print("\nRunning Vector Search Tests...")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query #{i}")
        print("-" * 60)
        print(f"Query: {query}")
        
        try:
            # Perform vector search
            results = await vector_search(collection, query, config)
            
            # Process and display results
            print("\nResults:")
            if results:
                for doc in results:
                    print("\nDocument:")
                    print(f"page_content: {doc['page_content'][:200]}...")
                    print(f"metadata: {doc['metadata']}")
                    print(f"score: {doc['score']}")
                    print("-" * 40)
            else:
                print("No matching documents found.")
                
        except Exception as e:
            print(f"Error during search: {str(e)}")
        
        print("-" * 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_vector_search()) 