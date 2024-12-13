import os
import json
import pymongo
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
from bson.binary import Binary, BinaryVectorDtype
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from config.config import Config



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
            f"Client Tax ID: {self.header.get('client_tax_id', '')} |"
            f"IBAN: {self.header.get('iban', '')}"
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
                        "date": invoice.header.get('invoice_date', ''),
                        "seller": invoice.header.get('seller', ''),
                        "client": invoice.header.get('client', ''),
                        "seller_tax_id": invoice.header.get('seller_tax_id', ''),
                        "client_tax_id": invoice.header.get('client_tax_id', ''),
                        "iban": invoice.header.get('iban', '')
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


def generate_bson_vector(vector, vector_dtype):
    """Convert numpy array to BSON vector format"""
    return Binary.from_vector(vector, vector_dtype)


def create_embeddings_for_documents(documents: List[Document], embeddings_model: OpenAIEmbeddings):
    """Generate embeddings for documents and convert to BSON format"""
    texts = [doc.page_content for doc in documents]
    raw_embeddings = embeddings_model.embed_documents(texts)
    
    # Debug: Print first embedding before conversion
    print("Raw embedding (first 5 values):", raw_embeddings[0][:5])
    
    # Convert to numpy arrays
    float32_arrays = [np.array(emb, dtype=np.float32) for emb in raw_embeddings]
    
    # Debug: Print arrays before BSON conversion
    print("Float32 array (first 5 values):", float32_arrays[0][:5])
    
    # Convert to BSON format
    bson_float32_embeddings = [
        generate_bson_vector(f32_arr, BinaryVectorDtype.FLOAT32)
        for f32_arr in float32_arrays
    ]
    
    return bson_float32_embeddings


def create_or_load_vector_store(collection, documents: List[Document], config: Config):
    """Modified to use BSON vector approach and return vector store interface"""
    try:
        # Initialize embeddings model
        embeddings_model = OpenAIEmbeddings(
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
                embedding=embeddings_model,
                index_name=config.VECTOR_INDEX_NAME,
                embedding_key="embedding_float", 
                text_key="page_content",
                metadata_key="metadata"
            )
            return vector_store
            
        print("Creating new vector store from documents...")
        
        # Generate BSON embeddings
        bson_float32_embeddings = create_embeddings_for_documents(
            documents, embeddings_model
        )
        
        # Create documents with embeddings
        docs_to_insert = []
        for i, (doc, f32_emb) in enumerate(zip(
            documents, bson_float32_embeddings
        )):
            mongo_doc = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "embedding_float": f32_emb,
            }
            docs_to_insert.append(mongo_doc)
        
        # Insert documents
        collection.insert_many(docs_to_insert)
        print(f"Inserted {len(docs_to_insert)} documents into MongoDB.")
        
        # Create and return vector store interface
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings_model,
            index_name=config.VECTOR_INDEX_NAME,
            embedding_key="embedding_float",  # Using float32 as primary embedding
            text_key="page_content",
            metadata_key="metadata"
        )
        
        print(f"Vector store interface created for {len(documents)} documents.")
        return vector_store
        
    except Exception as e:
        print(f"Error in vector store operation: {str(e)}")
        raise