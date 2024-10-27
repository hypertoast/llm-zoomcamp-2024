import os
import psycopg2
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import time

# Load model for embedding generation
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# OpenAI API client initialization
client = OpenAI(api_key='YOUR-API-KEY')

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")

### 1. Database Initialization ###
def init_db_connection():
    """Initialize and return the PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="feedback_db",
            user="your_user",
            password="your_password"
        )
        return conn
    except Exception as e:
        print("Database connection error:", e)
        return None

### 2. Generate Query Embedding ###
def generate_query_embedding(query):
    """Generate an embedding vector for the given query using Sentence Transformers."""
    return embedding_model.encode(query).tolist()

### 3. Retrieve Context from Elasticsearch ###
def retrieve_context(query, k=1):
    """Retrieve the top-k context documents from Elasticsearch using semantic search."""
    query_vector = generate_query_embedding(query)
    
    # Updated search query for Elasticsearch 8.x
    search_query = {
        "knn": {
            "field": "question_text_vector_knn",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": k * 2
        },
        "fields": ["text"],
        "_source": False
    }
    
    try:
        response = es.search(
            index="qa_index",
            body=search_query,
            size=k
        )
        return [hit["fields"]["text"][0] for hit in response["hits"]["hits"]]
    except Exception as e:
        print("Error retrieving context:", e)
        return []

### 4. Generate Response with Context (RAG) ###
def generate_response_with_context(query, context):
    """Generate a response using the query and retrieved context with RAG."""
    context_text = " ".join([f"Context {i+1}: {item}" for i, item in enumerate(context)])
    prompt = f"""
    Using the context provided, answer the question accurately.

    Context:
    {context_text}

    Question: {query}

    Answer:
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return completion.choices[0].message.content.strip(), completion.usage.total_tokens
    except Exception as e:
        print("Error generating RAG response:", e)
        return "Error: Unable to generate RAG response.", 0

### 5. Generate Standalone Response ###
def generate_standalone_response(query):
    """Generate a response using only the query, without additional context."""
    prompt = f"Answer the following question as accurately as possible:\n\nQuestion: {query}\n\nAnswer:"
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return completion.choices[0].message.content.strip(), completion.usage.total_tokens
    except Exception as e:
        print("Error generating standalone response:", e)
        return "Error: Unable to generate standalone response.", 0

### 6. Calculate Token Usage ###
def calculate_token_usage(tokens_used):
    """Calculate the cost based on token usage."""
    token_cost = 0.02  # Estimated cost per 1,000 tokens (update as needed)
    return tokens_used * token_cost / 1000
