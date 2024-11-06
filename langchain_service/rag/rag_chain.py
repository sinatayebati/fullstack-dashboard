import time
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config.config import Config

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