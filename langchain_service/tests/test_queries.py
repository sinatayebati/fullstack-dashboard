import requests
import json
import time
from typing import List, Dict

class RAGTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self) -> Dict:
        """Check if the service is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def query_rag(self, question: str) -> Dict:
        """Send a query to the RAG pipeline"""
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "question": question
        }
        response = self.session.post(
            f"{self.base_url}/query",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            print(f"Error Status Code: {response.status_code}")
            print(f"Error Response: {response.text}")
            raise Exception(f"Query failed with status code {response.status_code}")
            
        return response.json()
    
    def run_test_suite(self, questions: List[str]):
        """Run a series of test questions"""
        print("Starting RAG Pipeline Test Suite")
        print("=" * 80)
        
        # Check health
        print("\nChecking Service Health...")
        health = self.check_health()
        print(f"Service Status: {health['status']}")
        print(f"Timestamp: {health['timestamp']}")
        
        # Run queries
        print("\nRunning Test Queries...")
        for i, question in enumerate(questions, 1):
            print(f"\nTest Query #{i}")
            print("-" * 80)
            print(f"Question: {question}")
            
            try:
                start_time = time.time()
                result = self.query_rag(question)
                end_time = time.time()
                
                print("\nResults:")
                print(f"Answer: {result.get('answer', 'No answer provided')}")
                print(f"Confidence: {result.get('confidence', 'N/A')}")
                print(f"Response Time: {end_time - start_time:.2f} seconds")
                
                if 'sources' in result:
                    print("\nSources:")
                    for source in result['sources']:
                        print(f"- {source}")
                
                if 'source_documents' in result:
                    print("\nSource Documents:")
                    for doc in result['source_documents']:
                        print(f"- {doc[:200]}...")  # Print first 200 chars of each document
                
            except Exception as e:
                print(f"Error testing query: {str(e)}")
            
            print("-" * 80)
            time.sleep(1)  # Small delay between queries

if __name__ == "__main__":
    test_questions = [
        "what is the invoice number for the seller that has the following description: 'Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228'",
        "who is the seller of invoice no: 40378170",
    ]
    
    # Create tester and run tests
    tester = RAGTester()
    tester.run_test_suite(test_questions)