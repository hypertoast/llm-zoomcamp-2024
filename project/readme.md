# Mental Health Q&A System with RAG Flow Using OpenAI

This project is created for the [LLM Zoomcamp 2024](https://github.com/DataTalksClub/llm-zoomcamp).

## Contents
- [Problem Statement and Project Description](#problem-statement-and-project-description)
- [Technologies and Tools](#technologies-and-tools)
- [Deep-dive Technical Implementation](#deep-dive)
- [Evaluation Framework](#3-evaluation-framework)
- [How to Replicate](#how-to-replicate)
- [Scope for Improvement](#scope-for-improvement)

## Problem Statement and Project Description

Mental health is an increasingly important topic in today's world, with many individuals seeking answers to their mental health-related queries. However, accessing reliable information and immediate support can be challenging. This project addresses this need by building a Retrieval-Augmented Generation (RAG) system that provides mental health-related answers based on a curated conversational dataset.

The system combines:
- Knowledge base retrieval from a verified dataset
- Advanced language model capabilities (OpenAI's GPT models)
- Real-time feedback collection and analytics
- Comprehensive evaluation metrics

The goal is to demonstrate how LLMs and retrieval-based techniques can create a reliable Q&A system for mental health-related questions while maintaining transparency and measurable quality.

[App Screenshots are available here](https://github.com/hypertoast/llm-zoomcamp-2024/blob/main/project3/assets/readme.md)
[App has also been hosted on my personal VPS](https://stream.hypertoast.com)
--> PS: This link may not be accessible in a few days (to prevent abuse). I may take it down or bring it up with some additional checks for OpenAI key integration
Feel free to create an issue if it is not accessible during your evaluation and I will ensure its up

## Technologies and Tools

- **Data Source**: [Mental Health Conversational Data](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data/) from Kaggle
- **Core Technologies**:
  - Python 3.11
  - OpenAI API (GPT-3.5-turbo and GPT-4)
  - Elasticsearch for vector storage
  - PostgreSQL for analytics and feedback
  - Streamlit for UI
- **Key Libraries**:
  - sentence-transformers (all-mpnet-base-v2 model)
  - elasticsearch-py
  - psycopg2
  - pandas & numpy
- **Infrastructure**: Docker for containerization

## Deep-dive

### 1. Data Pipeline
- **Data Loading & Cleaning**:
  - Load conversational dataset (intents.json)
  - Clean text: remove whitespace, special characters, URLs
  - Normalize format for consistent processing

- **Index Creation**:
  - Custom analyzers: rag_analyzer & ngram_analyzer
  - Synonym filters for mental health terminology
  - Dense vector mappings for embeddings

- **Vector Generation**:
  - Multiple embedding types:
    - question_vector_knn
    - text_vector_knn
    - question_text_vector_knn
    - vector_combined_knn

### 2. Query Processing
- **Pattern Variations**:
  ```python
  variations = [
      pattern,
      f"Q: {pattern}",
      f"Question: {pattern}",
      pattern.lower(),
      f"Can you tell me {pattern.lower()}?",
      f"I want to know {pattern.lower()}"
  ]
  ```

- **Context Enhancement**:
  ```python
  contexts = [
      f"Question context: {text}",
      f"Background information: {text}",
      f"Complete context: {text}"
  ]
  ```

### 3. Evaluation Framework

#### Retrieval Evaluation
- Methods compared:
  - semantic_search_question
  - semantic_search_text
  - semantic_search_combined
  - text_search
  - combined_search

- Metrics:
  - Hit Rate
  - MRR (Mean Reciprocal Rank)
  - NDCG (Normalized Discounted Cumulative Gain)
  - F1 Score
  - Precision & Recall

#### RAG Evaluation
- Traditional Metrics:
  - ROUGE (1, 2, L)
  - BLEU Score
  - Semantic Similarity

- LLM as Judge:
  - GPT-3.5 vs GPT-4 comparison
  - Response completeness
  - Factual accuracy
  - Clarity and coherence

- A→Q→A' Evaluation:
  - Semantic consistency
  - Information preservation
  - Response quality

### 4. Application Features

#### Core Functionality:
- Model selection (GPT-3.5/GPT-4)
- Real-time query processing
- Context-aware responses
- Feedback collection

#### Analytics Dashboard:
- Usage statistics
- Response quality metrics
- Model performance comparison
- Cost analysis
- Exportable analytics data


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


## How to Replicate

### Prerequisites
1. OpenAI API key
2. Docker and Python 3.11 installed
3. At least 8GB RAM recommended

### Quick Start
```bash
# Clone repository
git clone https://github.com/hypertoast/llm-zoomcamp-2024/
cd llm-zoomcamp-2024/project

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY='your-key-here'

# Run application
streamlit run app.py
```

### Docker Deployment
```bash
# Build image
docker build -t mental-health-rag-app .

# Run container
docker run -it -p 8501:8501 \
  -e OPENAI_API_KEY='your-key-here' \
  mental-health-rag-app
```

### Docker Compose
```
docker compose up -d --build
```

PS: Current docker compose setup also installs Jupyter environment to play with notebooks and interact with the whole system.
My experience with Docker has been a little flaky where in a couple systems, inter container communication was sporadic. But, rest asasured, it works (already up on my demo)


Caution:
- Just realized that you will need to run the evaluation/retrieval_evaluation.ipynb for cleaning the data set, create index and index the data into Elastic Search. So, just one step before you jump in.

## Scope for Improvement

1. **Technical Enhancements**:
   - Integrate Kaggle API for automated dataset updates
   - Add offline LLM support for reduced costs
   - Implement data ingestion pipeline for continuous index updates

2. **Feature Additions**:
   - User authentication and session management
   - Response caching for common queries
   - Multi-language support

3. **Code Quality**:
   - Enhanced modularity for better reusability
   - Comprehensive test coverage
   - CI/CD pipeline integration

## License
MIT License

## Acknowledgments
- LLM Zoomcamp 2024 team
- Kaggle dataset contributors
- OpenAI API team
