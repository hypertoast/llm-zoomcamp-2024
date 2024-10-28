# llm-zoomcamp-2024
# Project 3
# Mental Health Q&A System with RAG Flow Using OpenAI

This project is created by me for the [LLM Zoomcamp 2024](https://github.com/DataTalksClub/llm-zoomcamp).

## Contents
- [Problem statement and project description](#problem-statement-and-project-description)
- [Technologies, tools and data sources used](#technologies-tools-and-data-sources-used)
- [Project flow diagram](#project-flow-diagram)
- [Project flow explanation](#project-flow-explanation)
- [How to replicate](#how-to-replicate)
- [Application demo and features](#application-demo-and-features)
- [Scope for improvement](#scope-for-improvement)
- [Reviewing criteria](#reviewing-criteria)

## Problem statement and project description

Mental health is an important topic in today’s world, and there are numerous queries that individuals may have regarding their mental well-being. However, having access to the right knowledge base and tools to answer these questions is often difficult. This project builds a Retrieval-Augmented Generation (RAG) system to provide mental health-related answers based on a conversational dataset from Kaggle.

Using OpenAI's GPT models, this project allows users to ask mental health-related questions and receive responses.

The goal of the project is to demonstrate the application of LLMs and retrieval-based techniques to create a reliable Q&A system for mental health-related questions using a blend of knowledge base retrieval and language model generation.

## Technologies, tools and data sources used

- Kaggle Dataset: [Mental Health Conversational Data](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data/) for knowledge base creation.
- Python: Used for building the application pipeline.
- Streamlit: For creating the UI to interact with the system.
- Transformers: For creating vector embeddings using the sentence-transformers/all-mpnet-base-v2 model.
- Elastic Search: Used as a vector data store to save embeddings from the dataset.
- Postgres: To perform RAG Monitoring, Feedback and Analytics.
- OpenAI API: For generating responses using the gpt-4 model.
- Docker: For containerization of the pipeline and the application.

## Project flow diagram

[A live Mermaid based Flow Diagram - Click Here](https://mermaid.ink/img/pako:eNqNkk1vwjAMhv-KlTNcJu3SwySgtFQD7QPYpeUQEtNWtEmVJkOI9r8v_Rh0EwdysJL4eW3HzoUwyZE45JDJE0uo0rBxIwF2TcJtiQo-DKozBKIwegfj8QtML2vMkGlYLld1h3Z22rirhYnjVMTgUYYVzH7hT9QqxW-awQp1Ink91LwVKCZBBW7oo0BFNVq-LKQoEU6pTqADwH_f7IYZZ636y8aXarynJXJYI1UsqWAe9hkRtmVTzx1qN4wyXT09wyueT1Ldonj_o9yh_lQ0b1vkNzKjxO0VWkLTzZ71HqLch6jO-i27CGcya9vdjs5D5HvKjj25aJkgXMr46gIqeAcHQtvOM51KYXkyIjmqnKbc_o1Lo4-ITjDHiDh2y6k6RiQSteWo0XJ9Fow4WhkcESVNnBDnQLPSnkzB7TjdlMaK5v1t_QMmXMwU?type=png)

## Project flow explanation

The project is divided into two main parts:

- Part 1: Data Ingestion, Cleansing, Mapping, vector Indexing:
  1. Kaggle Dataset: Load the conversational dataset (intents.json) from Kaggle, which includes user input patterns and associated responses.
  2. Cleaning: Clean the dataset for extra whitespaces, special characters, new lines, URLs etc.
  3. Create Index:  Create an enhanced index with improved settings, custom analyzers, and dense_vector mappings.
  4. Create Mappings
     - Analyzers: rag_analyzer & ngram_analyzer
     - Synonym filters
     - Mappings: 
        - question_vector_knn
        - text_vector_knn
        - question_text_vector_knn
        - vector_combined_knn
  5. Patterns:
      - variations: 
        pattern,
        Q: {pattern}
        Question: {pattern}
        pattern.lower()
        Can you tell me {pattern.lower()}?
        I want to know {pattern.lower()}
        Please explain {pattern.lower()}
      - Promot Engineering:
        - Question context: {text}
        - Background information: {text}
        - Complete context: {text}
        - {text}
  7. Evaluate Retrievals
     - Methods
       - semantic_search_question
       - semantic_search_text
       - semantic_search_combined
       - text_search
       - combined_search
    - Metrics used
      - Hit Rate
      - MRR
      - NDCG
      - F1
      - Precision
      - Recall   
  9. Evaluate RAG
      - Methods:
        - Traditional Metrics
          - ROUGE
          - BLEU
        - LLM As a Judge
          - GPT 3.5
          - GPT 4o
        - A→Q→A' Evaluation
  10. Streamlit UI:
    - Core
      - Users can select between OpenAI (GPT 3x/ GPT 4x) for querying.
      - Querying: Users input a query, which is passed through the chosen retrieval method.
      - Feedback Collection: After receiving the response, users can rate the helpfulness of the response, which is logged for future analytics.
    - Analytics
      - Basic Usage Statistics
      - Query vs Response Length
      - Feedback Analysis by Model GPT3.5
      - Feedback Analysis by Model GPT4o
      - Cost Analysis
      - Total Estimated Cost
      - Download Analytics Data

## Evaluations

### Retrieval
```
Overall Results:

                     Method       Hit Rate            MRR           NDCG             F1      Precision         Recall
0  semantic_search_question  0.500 ± 0.500  0.500 ± 0.500  0.964 ± 0.048  0.167 ± 0.167  0.100 ± 0.100  0.500 ± 0.500
1      semantic_search_text  0.500 ± 0.500  0.417 ± 0.449  0.870 ± 0.120  0.167 ± 0.167  0.100 ± 0.100  0.500 ± 0.500
2  semantic_search_combined  0.500 ± 0.500  0.500 ± 0.500  0.934 ± 0.098  0.167 ± 0.167  0.100 ± 0.100  0.500 ± 0.500
3               text_search  0.667 ± 0.471  0.667 ± 0.471  0.953 ± 0.093  0.222 ± 0.157  0.133 ± 0.094  0.667 ± 0.471
4           combined_search  0.667 ± 0.471  0.667 ± 0.471  0.964 ± 0.059  0.222 ± 0.157  0.133 ± 0.094  0.667 ± 0.471

```

Looking at the metrics:

1.  **Hit Rate** (percentage of queries where a relevant result was found):
-   `text_search` and `combined_search` perform best (0.667 ± 0.471)
-   Both methods find relevant results about 67% of the time
-   The high standard deviation (±0.471) suggests some variability in performance
2.  **MRR** (Mean Reciprocal Rank - how high the first relevant result appears):
-   Again, `text_search` and `combined_search` lead (0.667 ± 0.471)
-   This means when they find a relevant result, it tends to be ranked higher
3.  **NDCG** (Normalized Discounted Cumulative Gain - overall ranking quality):
-   All methods perform well (>0.87)
-   `semantic_search_question` and `combined_search` slightly lead (0.964)
-   Low standard deviations indicate consistent ranking quality
4.  **F1 Score** (harmonic mean of precision and recall):
-   `text_search` and `combined_search` perform best (0.222 ± 0.157)
-   However, the overall F1 scores are relatively low, suggesting room for improvement
5.  **Precision** (accuracy of relevant results):
-   Generally low across all methods (0.1-0.133)
-   `text_search` and `combined_search` slightly better
6.  **Recall** (ability to find all relevant results):
-   `text_search` and `combined_search` lead (0.667 ± 0.471)
-   Matches the hit rate, suggesting good coverage

**Recommendation:** Based on these results, I decided using the `combined_search` method for the RAG system because:
1.  It achieves the highest scores across most metrics
2.  It combines the benefits of both semantic and text-based search
3.  It shows the most consistent NDCG (0.964 ± 0.059)
4.  It ties for best hit rate and MRR


Let me break down what each method compared was actually doing:

1.  `semantic_search_question`
-   Uses the `question_vector_knn` field
-   Optimized for question-type embeddings
-   Pure vector similarity search using the question-focused embeddings
2.  `semantic_search_text`
-   Uses the `text_vector_knn` field
-   Optimized for general text embeddings
-   Pure vector similarity search using text-focused embeddings
3.  `semantic_search_combined`
-   Uses the `question_text_vector_knn` field
-   Combines both question and text embeddings
-   Pure vector search but with hybrid embeddings
4.  `text_search`
-   Traditional Elasticsearch text search
-   Uses the text fields directly
-   No vector/embedding involvement
-   Uses text analysis, synonyms, and fuzzy matching
5.  `combined_search`
-   Hybrid approach
-   Combines both vector search (`vector_combined_knn`) and text search
-   Weights between vector similarity (0.7) and text matching (0.3)

From our results, interestingly, the traditional `text_search` and the hybrid `combined_search` performed better than pure vector-based approaches. This suggests that for my specific use case, lexical matching (actual text matching) might be as important as semantic understanding.


### RAG

1.  **Traditional Metrics**
-   Very similar performance in basic metrics (ROUGE, BLEU)
-   Both models show comparable semantic\_similarity (0.589 vs 0.571)
-   Identical context\_relevance (0.554), suggesting similar retrieval performance
2.  **LLM Judge Evaluation**
-   GPT-4 scored significantly higher (4.07 vs 0.66)
-   Qualitative feedback shows key differences:
    -   GPT-4's responses are more comprehensive and detailed
    -   GPT-3.5's feedback consistently mentions "could be improved"
    -   GPT-4's feedback is more positive and indicates completeness
3.  **A→Q→A' Evaluation**
-   GPT-4 shows better performance across all metrics:
    -   Higher factual\_consistency (0.87 vs 0.61)
    -   Better information\_coverage (0.81 vs 0.58)
    -   Slightly lower semantic\_similarity (0.36 vs 0.45)

Key Insights:

1.  **Consistency vs Variation**
    -   GPT-4 shows higher consistency in maintaining factual accuracy
    -   GPT-3.5 might be more literal in semantic similarity but less comprehensive
2.  **Quality of Responses**
    -   GPT-4 provides more complete and detailed responses
    -   GPT-3.5 gives adequate but less detailed answers
3.  **Retrieval Effectiveness**
    -   Both models show similar context\_relevance
    -   The difference is in how they use the retrieved information


## How to replicate

### Step 0 - Prerequisites
- Create an OpenAI API key from here.
- Install Python and Docker.

### Step 1 - Clone the Repo
- [Clone Repo](https://github.com/hypertoast/llm-zoomcamp-2024/)
```bash
git clone https://github.com/hypertoast/llm-zoomcamp-2024/
cd llm-zoomcamp-2024/project
```

### Step 2 - Set up Python environment
- Install the required Python libraries:
```bash
pip install -r requirements.txt
```
### Step 3 - Add OpenAI key
- Edit app.py and update your OpenAI API Key

### Step 4 - Run the application
```python
streamlit run app.py
```

### Step 5 - Containerization with Docker
- Build the Docker image:
```bash
docker build -t mental-health-rag-app .
```
- Run the Docker container:
```bash
docker run -it -p 8501:8501 mental-health-rag-app
```

## Scope for improvement
- Integrating Kaggle API to download the data-set. Currently, the raw source is embedded within this project (intents.json)
- Integrate Offline LLM to save costs
- Modularize code for reusability
- Data Ingestion Pipeline to continuously update indexes in Elastic Search
- Slightly better polished UI with user management
- 


## Reviewing criteria

[Click Here](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md#evaluation-criteria) for the reviewing criteria
