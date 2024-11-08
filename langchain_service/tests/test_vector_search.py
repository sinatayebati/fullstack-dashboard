import sys
import os
import requests
import json
import time
from typing import Dict

class VectorSearchTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self) -> Dict:
        """Check if the service is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def vector_search(self, query: str, filters: dict = None, use_int8: bool = False) -> Dict:
        """Send a vector search request to the API"""
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "filters": filters,
            "use_int8": use_int8
        }
        response = self.session.post(
            f"{self.base_url}/vector-search",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            print(f"Error Status Code: {response.status_code}")
            print(f"Error Response: {response.text}")
            raise Exception(f"Vector search failed with status code {response.status_code}")
            
        return response.json()

    def run_test_suite(self, test_cases: list):
        """Run a series of vector search tests"""
        print("Starting Vector Search Test Suite")
        print("=" * 80)
        
        # Check health
        print("\nChecking Service Health...")
        health = self.check_health()
        print(f"Service Status: {health['status']}")
        print(f"Timestamp: {health['timestamp']}")
        
        # Run test cases
        print("\nRunning Test Cases...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case #{i}")
            print("-" * 60)
            print(f"Query: {test_case['query']}")
            print(f"Filters: {test_case.get('filters', {})}")
            print(f"Use Int8: {test_case.get('use_int8', False)}")
            
            try:
                start_time = time.time()
                result = self.vector_search(
                    query=test_case['query'],
                    filters=test_case.get('filters'),
                    use_int8=test_case.get('use_int8', False)
                )
                end_time = time.time()
                
                print("\nResults:")
                if result['results']:
                    for doc in result['results']:
                        print("\nDocument:")
                        print(f"page_content: {doc['page_content'][:200]}...")
                        print("metadata:")
                        print(json.dumps(doc['metadata'], indent=4))
                        print(f"score: {doc['score']}")
                        print("-" * 40)
                else:
                    print("No matching documents found.")
                
                print(f"\nResponse Time: {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error testing vector search: {str(e)}")
            
            print("-" * 60)
            time.sleep(1)  # Small delay between queries

if __name__ == "__main__":
    # Test cases with use_int8 parameter
    test_cases = [
        {
            "query": "Seller: Davis PLC 72057 Castillo Via Deniseshire, KY 95233",
            "filters": {
                "invoice_no": "39280409",
                "date": "07/06/2014",
            },
            "use_int8": False
        },
        {
            "query": "Items: Description: BUYPOWER Gaming Computer AMD Ryzen 3 3100",
            "filters": {},
            "use_int8": False
        },
    ]
    
    # Create tester and run tests
    tester = VectorSearchTester()
    tester.run_test_suite(test_cases)