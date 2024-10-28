import os
import json
import pymongo
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (you'll need to set these)
MONGODB_URI = os.getenv("MONGO_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class Config:
    DB_NAME: str = "langchain_db"
    COLLECTION_NAME: str = "invoice_db"
    VECTOR_INDEX_NAME: str = "invoice_vector_index"
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    MONGODB_TIMEOUT_MS: int = 5000
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    RETRIEVER_K: int = 3

# DB_NAME = "langchain_db"
# COLLECTION_NAME = "invoice_db"
# VECTOR_INDEX_NAME = "invoice_vector_index"

def validate_environment():
    required_vars = {
        "MONGO_URL": MONGODB_URI,
        "OPENAI_API_KEY": OPENAI_API_KEY
    }
    missing = [var for var, val in required_vars.items() if not val]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Load data index
with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'index.json')) as f:
    DATA_INDEX = json.load(f)

def connect_to_mongodb(config: Config):
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS)
        client.server_info()
        db = client[config.DB_NAME]
        collection = client[config.DB_NAME][config.COLLECTION_NAME]
        return client, db, collection
    except pymongo.errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB server")
        raise
    except pymongo.errors.OperationFailure as e:
        print(f"Authentication failed: {str(e)}")
        raise

def check_or_create_vector_index(collection, config: Config):
    try:
        # Get all search indexes
        existing_indexes = list(collection.list_search_indexes())
        # print(f"Raw existing indexes: {existing_indexes}")  # Debug print
        
        # Extract just the names for easier checking
        existing_index_names = [index.get('name') for index in existing_indexes]
        print(f"Existing index names: {existing_index_names}")  # Debug print
        
        if config.VECTOR_INDEX_NAME in existing_index_names:
            print(f"Vector index '{config.VECTOR_INDEX_NAME}' already exists. Skipping creation.")
            return
            
        print(f"No index matching '{config.VECTOR_INDEX_NAME}' found. Creating new index...")
        
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
                        "path": "id",
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
            print(f"NOTE: Despite our checks, MongoDB reports that index '{config.VECTOR_INDEX_NAME}' already exists.")
            print(f"Error details: {str(e)}")  # Debug print
        else:
            print(f"An error occurred: {str(e)}")  # Debug print
            raise

def load_data_to_mongodb(collection, data_file, config: Config):
    # Check if data already exists
    if collection.count_documents({}) > 0:
        print("Data already exists in the collection. Skipping data loading.")
        return []

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
        
    documents = []
    total_documents = 0
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for row in data['rows']:
            parsed_data = json.loads(row['row']['parsed_data'])
            content = (
                f"Invoice ID: {row['row']['id']}\n"
                f"Parsed Data: {parsed_data['json']}\n"
                f"Raw Data: {row['row']['raw_data']}"
            )
            
            doc = Document(
                page_content=content,
                metadata={
                    "id": row['row']['id'],
                    "image_url": row['row']['image']['src'],
                    "timestamp": datetime.datetime.utcnow()
                }
            )
            documents.append(doc)
            
            if len(documents) >= config.BATCH_SIZE:
                collection.insert_many([doc.dict() for doc in documents])
                total_documents += len(documents)
                documents = []
                
        if documents:  # Insert remaining documents
            collection.insert_many([doc.dict() for doc in documents])
            total_documents += len(documents)
            
        print(f"Loaded {total_documents} documents into MongoDB.")
        return total_documents
        
    except json.JSONDecodeError:
        print("Error parsing JSON data")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def create_or_load_vector_store(collection, config: Config):
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        index_name=config.VECTOR_INDEX_NAME
    )
    print(f"Vector store loaded or created with index '{config.VECTOR_INDEX_NAME}'.")
    return vector_store

def setup_rag_pipeline(vector_store, config: Config):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K}
    )

    template = """
    You are an AI assistant analyzing invoice data. Use the following context to answer the question.
    If the answer cannot be determined from the context, say "I cannot answer this based on the available information."
    
    Context:
    {context}
    
    Question: {question}
    
    Remember to:
    - Only use information from the provided context
    - Be specific and cite invoice IDs when relevant
    - Indicate if any information is unclear or missing
    
    Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(
        model=config.LLM_MODEL, 
        temperature=config.LLM_TEMPERATURE,
        request_timeout=config.REQUEST_TIMEOUT,
        max_retries=config.MAX_RETRIES
    )

    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            formatted_doc = f"Invoice ID: {doc.metadata['id']}\n"
            formatted_doc += f"Content: {doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        return "\n\n".join(formatted_docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def query_rag_pipeline(rag_chain, retriever, question):
    answer = rag_chain.invoke(question)
    source_documents = retriever.get_relevant_documents(question)
    return answer, source_documents

def initialize_rag_pipeline(config: Optional[Config] = None):
    if config is None:
        config = Config()
        
    client, db, collection = connect_to_mongodb(config)
    check_or_create_vector_index(collection, config)
    data_file = os.path.join(os.path.dirname(__file__), '..', DATA_INDEX['invoice_data']['path'])
    total_documents = load_data_to_mongodb(collection, data_file, config)
    vector_store = create_or_load_vector_store(collection, config)
    rag_chain, retriever = setup_rag_pipeline(vector_store, config)
    return rag_chain, retriever, client

def query_rag(rag_chain, retriever, question, config: Optional[Config] = None):
    if config is None:
        config = Config()
        
    for attempt in range(config.MAX_RETRIES):
        try:
            answer, source_docs = query_rag_pipeline(rag_chain, retriever, question)
            return {
                'answer': answer,
                'sources': [doc.metadata['id'] for doc in source_docs],
                'confidence': 'high' if source_docs else 'low',
                'source_documents': source_docs  # Add this to match the API response
            }
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                raise
            print(f"Query attempt {attempt + 1} failed: {str(e)}")
            time.sleep(1)  # Add delay between retries
