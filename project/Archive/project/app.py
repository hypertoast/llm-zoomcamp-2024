import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import json
import random
import datetime
from openai import OpenAI
from rank_bm25 import BM25Okapi  # Import BM25 for keyword-based retrieval


# Step 1: OpenAI API key (replace 'your-openai-api-key' with your actual API key)
client = OpenAI(api_key='your-openai-api-key')

# Step 2: Load the Kaggle dataset (intents.json)
with open('intents.json') as f:
    data = json.load(f)

# Step 3: Flatten the JSON structure to get patterns and responses
patterns = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])  # List of responses for each pattern

# Step 4: BM25 keyword-based retrieval setup
tokenized_patterns = [pattern.split() for pattern in patterns]  # Tokenize the patterns for BM25
bm25 = BM25Okapi(tokenized_patterns)  # Initialize BM25 with tokenized patterns

# Step 5: Load Hugging Face Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Step 6: Embed text function using Hugging Face model
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Step 7: Generate embeddings for all patterns (queries)
response_embeddings = torch.cat([embed_text(pattern) for pattern in patterns])

# Step 8: Define BM25 retrieval function
def retrieve_response_bm25(query, patterns, responses):
    tokenized_query = query.split()  # Tokenize the query
    bm25_scores = bm25.get_scores(tokenized_query)  # Get BM25 scores for the query
    closest_index = np.argmax(bm25_scores)  # Find the closest matching pattern
    return random.choice(responses[closest_index]), "BM25 Keyword-based"

# Step 9: Define the Hugging Face retrieval function (vector-based)
def retrieve_response_huggingface(query, response_embeddings, patterns, responses):
    query_embedding = embed_text(query).numpy()
    similarities = cosine_similarity(query_embedding, response_embeddings.numpy())
    closest_index = np.argmax(similarities)

    # Randomly select a response from the associated responses
    return random.choice(responses[closest_index]), "Vector-based"

# Step 10: Define the OpenAI retrieval function with dataset context
def retrieve_response_openai(query, context_responses):
    # Create a prompt using the Kaggle dataset responses as context
    prompt = f"""
    You are a mental health assistant. Here are some previous responses from our knowledge base:

    {context_responses}

    Based on the above knowledge, answer the following question:
    User: {query}
    """

    # Use the new ChatCompletion.create API
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful mental health assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150)
        return response.choices[0].message.content.strip(), prompt
    except Exception as e:
        return f"Error: {str(e)}", prompt

# Step 11: Define logging function to log feedback, LLM, retrieval method, prompt, and response
def log_feedback(query, response, feedback, llm_choice, retrieval_method, prompt):
    with open('feedback_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now(), query, llm_choice, retrieval_method, prompt, response, feedback])

# Step 12: Streamlit UI with LLM selection, retrieval method, and feedback system
st.title("Mental Health Q&A System")

# LLM selection (Hugging Face or OpenAI)
llm_choice = st.selectbox("Select the LLM to use:", ("Hugging Face", "OpenAI"))

# Retrieval method selection (BM25 Keyword-based or Vector-based search)
retrieval_method_choice = st.selectbox("Select the retrieval method:", ("BM25 Keyword-based", "Vector-based"))

# User query input
user_query = st.text_input("Enter your query:")

if user_query:
    retrieval_method = None
    prompt = None

    # Determine which LLM to use for the response
    if llm_choice == "Hugging Face":
        if retrieval_method_choice == "BM25 Keyword-based":
            response, retrieval_method = retrieve_response_bm25(user_query, patterns, responses)
        else:
            response, retrieval_method = retrieve_response_huggingface(user_query, response_embeddings, patterns, responses)
        prompt = "N/A"  # Hugging Face doesn't use a prompt-based generation
    else:
        # Create a context using the responses from the dataset
        context_responses = "\n".join([random.choice(resp_list) for resp_list in responses])
        response, prompt = retrieve_response_openai(user_query, context_responses)
        retrieval_method = "N/A"  # Since OpenAI generates directly from the prompt

    st.write("Retrieved Response:")
    st.write(response)

    # Feedback buttons
    feedback = st.radio("Was this response helpful?", ('Yes', 'No'))

    if st.button("Submit Feedback"):
        log_feedback(user_query, response, feedback, llm_choice, retrieval_method, prompt)
        st.write("Thank you for your feedback!")
