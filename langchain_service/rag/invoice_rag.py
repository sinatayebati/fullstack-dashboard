import os
import json
import pymongo
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Any, Dict
from pydantic import Field, BaseModel
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

# Load environment variables (you'll need to set these)
MONGODB_URI = os.getenv("MONGO_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Config:
    DB_NAME: str = "langchain_db"
    COLLECTION_VECTOR: str = "invoice_db"
    COLLECTION_IMAGE: str = "image_bytes"
    VECTOR_INDEX_NAME: str = "invoice_vector_index"
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    MONGODB_TIMEOUT_MS: int = 5000
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    RETRIEVER_K: int = 10

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
    """Enhanced connection function that ensures collection exists"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS)
        client.server_info()
        db = client[config.DB_NAME]
        
        # Explicitly create collection if it doesn't exist
        if config.COLLECTION_VECTOR not in db.list_collection_names():
            db.create_collection(config.COLLECTION_VECTOR)
            print(f"Created collection: {config.COLLECTION_VECTOR}")
        
        collection = db[config.COLLECTION_VECTOR]
        return client, db, collection
        
    except pymongo.errors.ServerSelectionTimeoutError:
        print("Failed to connect to MongoDB server")
        raise
    except pymongo.errors.OperationFailure as e:
        print(f"Authentication failed: {str(e)}")
        raise

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

@dataclass
class InvoiceDocument:
    id: str
    header: Dict[str, Any]
    items: list[Dict[str, Any]]
    summary: Dict[str, Any]
    raw_text: str
    
    def to_searchable_text(self) -> str:
        """Convert invoice data to a structured text format optimized for embedding"""
        header_text = (
            f"Invoice Number: {self.header.get('invoice_no', '')} | "
            f"Date: {self.header.get('invoice_date', '')} | "
            f"Seller: {self.header.get('seller', '')} | "
            f"Seller Tax ID: {self.header.get('seller_tax_id', '')} | "
            f"Client: {self.header.get('client', '')} | "
            f"Client Tax ID: {self.header.get('client_tax_id', '')}"
        )
        
        items_text = "Items: " + " | ".join(
            f"Description: {item.get('item_desc', '')}, "
            f"Quantity: {item.get('item_qty', '')}, "
            f"Price: {item.get('item_net_price', '')}"
            for item in self.items
        )
        
        summary_text = (
            f"Summary - Net Worth: {self.summary.get('total_net_worth', '')} | "
            f"VAT: {self.summary.get('total_vat', '')} | "
            f"Gross Worth: {self.summary.get('total_gross_worth', '')}"
        )
        
        return f"{header_text}\n{items_text}\n{summary_text}"

def parse_invoice_data(row: Dict[str, Any]) -> InvoiceDocument:
    """
    Parse raw invoice data into structured format with improved JSON handling
    """
    try:
        # First parse the outer JSON structure
        parsed_data = json.loads(row['row']['parsed_data'])
        
        # The 'json' field contains a string that looks like a Python dict
        # Need to clean it properly before parsing
        json_str = parsed_data['json']
        
        # Clean up the string to make it valid JSON
        # Replace Python-style single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        # Handle potential None/null values
        json_str = json_str.replace('None', 'null')
        
        # Parse the cleaned JSON string
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing inner JSON: {str(e)}")
            print(f"Problematic JSON string: {json_str}")
            raise
            
        # Create InvoiceDocument with proper error handling for missing fields
        return InvoiceDocument(
            id=row['row']['id'],
            header=json_data.get('header', {}),
            items=json_data.get('items', []),
            summary=json_data.get('summary', {}),
            raw_text=row['row'].get('raw_data', '')
        )
        
    except Exception as e:
        print(f"Error in parse_invoice_data: {str(e)}")
        print(f"Input row structure: {json.dumps(row, indent=2)}")
        raise


def load_data_to_mongodb(collection, data_file, config: Config) -> List[Document]:
    """Modified to handle document processing without directly inserting"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        documents = []
        print(f"Processing {len(data['rows'])} rows from data file...")
        
        for row_index, row in enumerate(data['rows']):
            try:
                invoice = parse_invoice_data(row)
                
                # Create main document with searchable text
                main_doc = Document(
                    page_content=invoice.to_searchable_text(),
                    metadata={
                        "id": invoice.id,
                        "invoice_no": invoice.header.get('invoice_no', ''),
                        "seller": invoice.header.get('seller', ''),
                        "client": invoice.header.get('client', '')
                    }
                )
                
                documents.append(main_doc)
                
                if (row_index + 1) % 100 == 0:
                    print(f"Processed {row_index + 1} documents...")
                
            except Exception as e:
                print(f"Error processing row {row_index}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(documents)} documents.")
        return documents
        
    except Exception as e:
        print(f"Error in load_data_to_mongodb: {str(e)}")
        raise


def load_image_bytes_to_mongodb(db, data_file, config: Config):
    """
    Load image bytes data to a separate MongoDB collection.
    """
    image_collection = db[config.COLLECTION_IMAGE]

    # Check if data already exists
    if image_collection.count_documents({}) > 0:
        print("Image bytes data already exists in the collection. Skipping data loading.")
        return []

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    image_documents = []
    total_documents = 0
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Create index for invoice_id if it doesn't exist
        image_collection.create_index("invoice_id", unique=True)
        
        for row in data['rows']:
            image_doc = {
                "invoice_id": row['row']['id'],
                "image_height": row['row']['image']['height'],
                "image_width": row['row']['image']['width'],
                "image_bytes": row['row']['image']['bytes']
            }
            image_documents.append(image_doc)
            
            if len(image_documents) >= config.BATCH_SIZE:
                try:
                    image_collection.insert_many(image_documents, ordered=False)
                except pymongo.errors.BulkWriteError as e:
                    print(f"Some documents were not inserted: {str(e)}")
                total_documents += len(image_documents)
                image_documents = []
                
        if image_documents:  # Insert remaining documents
            try:
                image_collection.insert_many(image_documents, ordered=False)
                total_documents += len(image_documents)
            except pymongo.errors.BulkWriteError as e:
                print(f"Some documents were not inserted: {str(e)}")
            
        print(f"Loaded {total_documents} image documents into MongoDB.")
        return total_documents
        
    except json.JSONDecodeError:
        print("Error parsing JSON data")
        raise
    except Exception as e:
        print(f"Error loading image data: {str(e)}")
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


def setup_rag_pipeline(vector_store, config: Config):
    """Modified RAG pipeline with better query handling"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": config.RETRIEVER_K,
            "score_threshold": 0.7,
            "pre_filter": {}
        }
    )

    template = """
    You are an AI assistant specialized in retrieving invoice information. The context contains structured invoice data with specific fields like invoice numbers, seller information, and addresses.

    Context:
    {context}

    Question: {question}

    Instructions:
    1. For invoice number queries: Look for exact matches in the invoice_no field
    2. For seller queries: Compare complete seller addresses
    3. For client queries: Match client information exactly
    4. If multiple matches are found, list all relevant matches
    5. If no exact match is found, indicate this clearly and suggest similar matches if available

    Provide your answer in this format:
    - Exact Match: [Yes/No]
    - Found Information: [Details]
    - Confidence: [High/Medium/Low]
    - Supporting Details: [Any relevant context]

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
        return "\n\n".join(
            f"Invoice ID: {doc.metadata['id']}\n"
            f"Content: {doc.page_content}\n"
            for doc in docs
        )

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
        
    # 1. Connect to MongoDB
    client, db, collection = connect_to_mongodb(config)
    
    # 2. Check if collection already has documents
    doc_count = collection.count_documents({})
    documents = []
    
    if doc_count == 0:
        print("Collection is empty. Loading documents...")
        # Only load documents if collection is empty
        data_file = os.path.join(os.path.dirname(__file__), '..', DATA_INDEX['invoice_data']['path'])
        documents = load_data_to_mongodb(collection, data_file, config)
        
        # Load image data separately
        total_image_documents = load_image_bytes_to_mongodb(db, data_file, config)
    else:
        print(f"Collection already contains {doc_count} documents. Skipping document loading.")
    
    # 3. Create or load vector store
    vector_store = create_or_load_vector_store(collection, documents, config)
    
    # 4. Ensure vector index exists
    check_or_create_vector_index(collection, config)
    
    # 5. Setup RAG pipeline
    rag_chain, retriever = setup_rag_pipeline(vector_store, config)
    
    return rag_chain, retriever, client


def query_rag(rag_chain, retriever, question, config: Optional[Config] = None):
    if config is None:
        config = Config()
    
    def process_query(query_input):
        if isinstance(query_input, dict):
            base_query = query_input.get('query', '')
            filter_dict = query_input.get('filter', {})
            
            # Process filter dict to match MongoDB format
            processed_filter = {}
            for key, value in filter_dict.items():
                clean_key = key.replace('metadata.', '')
                processed_filter[clean_key] = value
            
            # Update retriever search kwargs with filter
            retriever.search_kwargs.update({'pre_filter': processed_filter})
            return base_query
        return query_input
    
    for attempt in range(config.MAX_RETRIES):
        try:
            processed_question = process_query(question)
            answer, source_docs = query_rag_pipeline(rag_chain, retriever, processed_question)
            
            # Reset search kwargs to default
            retriever.search_kwargs = {
                "k": config.RETRIEVER_K,
                "score_threshold": 0.7
            }
            
            return {
                'answer': answer,
                'sources': [doc.metadata['id'] for doc in source_docs],
                'confidence': 'high' if source_docs else 'low',
                'source_documents': source_docs
            }
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                raise
            print(f"Query attempt {attempt + 1} failed: {str(e)}")
            time.sleep(1)

