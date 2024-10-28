import os
import json
import pymongo
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

# Constants
DB_NAME = "langchain_db"
COLLECTION_NAME = "invoice_db"
VECTOR_INDEX_NAME = "invoice_vector_index"

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

def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Verify connection
        client.server_info()
        db = client[DB_NAME]
        collection = client[DB_NAME][COLLECTION_NAME]
        return client, db, collection
    except pymongo.errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB server")
        raise
    except pymongo.errors.OperationFailure as e:
        print(f"Authentication failed: {str(e)}")
        raise

def check_or_create_vector_index(collection):
    try:
        # Get all search indexes
        existing_indexes = list(collection.list_search_indexes())
        # print(f"Raw existing indexes: {existing_indexes}")  # Debug print
        
        # Extract just the names for easier checking
        existing_index_names = [index.get('name') for index in existing_indexes]
        print(f"Existing index names: {existing_index_names}")  # Debug print
        
        if VECTOR_INDEX_NAME in existing_index_names:
            print(f"Vector index '{VECTOR_INDEX_NAME}' already exists. Skipping creation.")
            return
            
        print(f"No index matching '{VECTOR_INDEX_NAME}' found. Creating new index...")
        
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
            name=VECTOR_INDEX_NAME,
            type="vectorSearch"
        )
        
        collection.create_search_index(search_index_model)
        print(f"Vector index '{VECTOR_INDEX_NAME}' created successfully.")
        
    except pymongo.errors.OperationFailure as e:
        if "already exists" in str(e):
            print(f"NOTE: Despite our checks, MongoDB reports that index '{VECTOR_INDEX_NAME}' already exists.")
            print(f"Error details: {str(e)}")  # Debug print
        else:
            print(f"An error occurred: {str(e)}")  # Debug print
            raise

def load_data_to_mongodb(collection, data_file):
    # Check if data already exists
    if collection.count_documents({}) > 0:
        print("Data already exists in the collection. Skipping data loading.")
        return []

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
        
    # Batch processing for large datasets
    BATCH_SIZE = 100
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
            
            if len(documents) >= BATCH_SIZE:
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

def create_or_load_vector_store(collection):
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        index_name=VECTOR_INDEX_NAME
    )
    print(f"Vector store loaded or created with index '{VECTOR_INDEX_NAME}'.")
    return vector_store

def setup_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
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
        model="gpt-4o-mini", 
        temperature=0,
        request_timeout=30,
        max_retries=2
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

def initialize_rag_pipeline():
    client, db, collection = connect_to_mongodb()
    check_or_create_vector_index(collection)
    data_file = os.path.join(os.path.dirname(__file__), '..', DATA_INDEX['invoice_data']['path'])
    total_documents = load_data_to_mongodb(collection, data_file)
    vector_store = create_or_load_vector_store(collection)
    rag_chain, retriever = setup_rag_pipeline(vector_store)
    return rag_chain, retriever, client

def query_rag(rag_chain, retriever, question):
    return query_rag_pipeline(rag_chain, retriever, question)
