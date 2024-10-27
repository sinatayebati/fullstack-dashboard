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

# Load data index
with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'index.json')) as f:
    DATA_INDEX = json.load(f)

def connect_to_mongodb():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = client[DB_NAME][COLLECTION_NAME]
    return client, db, collection

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

    with open(data_file, 'r') as f:
        data = json.load(f)
    
    documents = []
    for row in data['rows']:
        parsed_data = json.loads(row['row']['parsed_data'])
        content = f"Invoice ID: {row['row']['id']}\n"
        content += f"Parsed Data: {parsed_data['json']}\n"
        content += f"Raw Data: {row['row']['raw_data']}"
        
        doc = Document(
            page_content=content,
            metadata={
                "id": row['row']['id'],
                "image_url": row['row']['image']['src']
            }
        )
        documents.append(doc)
    
    # Insert documents into MongoDB
    if documents:
        collection.insert_many([doc.dict() for doc in documents])
        print(f"Loaded {len(documents)} documents into MongoDB.")
    
    return documents

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
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    {context}
    Question: {question}
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    documents = load_data_to_mongodb(collection, data_file)
    vector_store = create_or_load_vector_store(collection)
    rag_chain, retriever = setup_rag_pipeline(vector_store)
    return rag_chain, retriever, client

def query_rag(rag_chain, retriever, question):
    return query_rag_pipeline(rag_chain, retriever, question)
