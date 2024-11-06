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
    
    # Test queries with filters
    test_cases = [
        {
            "query": "Seller: Davis PLC 72057 Castillo Via Deniseshire, KY 95233",
            "filters": {
                "invoice_no": "39280409",
                "date": "07/06/2014",
            }
        },
        {
            "query": "Items: Description: BUYPOWER Gaming Computer AMD Ryzen 3 3100",
            "filters": {
            }
        },
    ]
    
    print("\nRunning Vector Search Tests...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Query #{i}")
        print("-" * 60)
        print(f"Query: {test_case['query']}")
        print(f"Filters: {test_case['filters']}")
        
        try:
            # Perform vector search with filters
            results = await vector_search(
                collection=collection,
                query_text=test_case['query'],
                config=config,
                filters=test_case['filters']
            )
            
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