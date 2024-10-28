import streamlit as st
from elasticsearch import Elasticsearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Tuple
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'mental_health_rag'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres'),
            host=os.getenv('DB_HOST', 'localhost')
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def log_message(self, role: str, content: str, model_type: str,
                   tokens_used: int = 0, response_time: int = None) -> int:
        """Log chat message"""
        try:
            self.cursor.execute(
                """
                INSERT INTO chat_messages 
                (role, content, model_type, tokens_used, response_time_ms)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (role, content, model_type, tokens_used, response_time)
            )
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except Exception as e:
            logging.error(f"Error logging message: {str(e)}")
            return None

    def log_feedback(self, message_id: int, rating: bool):
        """Log user feedback"""
        try:
            self.cursor.execute(
                "INSERT INTO message_feedback (message_id, rating) VALUES (%s, %s)",
                (message_id, rating)
            )
            self.conn.commit()
        except Exception as e:
            logging.error(f"Error logging feedback: {str(e)}")

    def log_context(self, message_id: int, context_text: str, relevance_score: float = 0):
        """Log retrieved context for a message"""
        try:
            self.cursor.execute(
                """
                INSERT INTO message_contexts 
                (message_id, context_text, relevance_score)
                VALUES (%s, %s, %s)
                """,
                (message_id, context_text, relevance_score)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging context: {str(e)}")
            self.conn.rollback()


class SimpleAnalyticsDashboard:
    def __init__(self, db_connection):
        self.conn = db_connection

    def execute_query_with_debug(self, query: str, params: dict = None) -> Tuple[pd.DataFrame, str]:
        """Execute query and return both results and debug info"""
        debug_query = query
        if params:
            for key, value in params.items():
                debug_query = debug_query.replace(f":{key}", str(value))
        
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            return df, debug_query
        except Exception as e:
            return pd.DataFrame(), f"Query failed: {str(e)}\n\nQuery:\n{debug_query}"

    def render_dashboard(self):
        st.title("Analytics Dashboard 📊")
        
        # Debug mode toggle
        debug_mode = st.sidebar.checkbox("Show Debug Info")

        # 1. Basic Usage Statistics
        st.subheader("Basic Usage Statistics")
        daily_usage_query = """
        SELECT 
            DATE(timestamp) as date,
            model_type,
            COUNT(*) as request_count
        FROM chat_messages 
        WHERE role = 'assistant'
        AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY DATE(timestamp), model_type
        ORDER BY date;
        """
        
        usage_df, debug_query = self.execute_query_with_debug(daily_usage_query)
        if debug_mode:
            with st.expander("Daily Usage Query"):
                st.code(debug_query, language='sql')
        
        if not usage_df.empty:
            st.bar_chart(usage_df.pivot(index='date', columns='model_type', values='request_count'))

        # 2. Query vs Response Length Analysis
        st.subheader("Query vs Response Length")
        length_query = """
        WITH conversation_pairs AS (
            SELECT 
                m1.content as query,
                m2.content as response,
                m2.model_type
            FROM chat_messages m1
            JOIN chat_messages m2 
            ON m1.id = (m2.id - 1)
            WHERE m1.role = 'user' 
            AND m2.role = 'assistant'
            AND m2.timestamp >= CURRENT_DATE - INTERVAL '7 days'
        )
        SELECT 
            model_type,
            AVG(LENGTH(query)) as avg_query_length,
            AVG(LENGTH(response)) as avg_response_length,
            DATE(NOW()) as date
        FROM conversation_pairs
        GROUP BY model_type;
        """
        
        length_df, length_debug = self.execute_query_with_debug(length_query)
        if debug_mode:
            with st.expander("Length Analysis Query"):
                st.code(length_debug, language='sql')
        
        if not length_df.empty:
            # Create a bar chart for query vs response lengths
            length_comparison = pd.melt(
                length_df, 
                id_vars=['model_type'],
                value_vars=['avg_query_length', 'avg_response_length'],
                var_name='metric',
                value_name='length'
            )
            st.bar_chart(length_comparison.set_index('metric')['length'])

        # 3. Feedback Analysis by Model
        st.subheader("Feedback Analysis by Model")
        feedback_query = """
        SELECT 
            cm.model_type,
            COUNT(CASE WHEN mf.rating = true THEN 1 END) as positive_feedback,
            COUNT(CASE WHEN mf.rating = false THEN 1 END) as negative_feedback
        FROM chat_messages cm
        LEFT JOIN message_feedback mf ON cm.id = mf.message_id
        WHERE cm.role = 'assistant'
        AND cm.timestamp >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY cm.model_type;
        """
        
        feedback_df, feedback_debug = self.execute_query_with_debug(feedback_query)
        if debug_mode:
            with st.expander("Feedback Analysis Query"):
                st.code(feedback_debug, language='sql')
        
        if not feedback_df.empty:
            # Create separate charts for each model
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("GPT-3.5 Feedback")
                gpt35_feedback = feedback_df[feedback_df['model_type'] == 'gpt-3.5-turbo'].iloc[0]
                st.bar_chart({
                    'Positive': [gpt35_feedback['positive_feedback']],
                    'Negative': [gpt35_feedback['negative_feedback']]
                })
            
            with col2:
                st.subheader("GPT-4 Feedback")
                gpt4_feedback = feedback_df[feedback_df['model_type'] == 'gpt-4'].iloc[0]
                st.bar_chart({
                    'Positive': [gpt4_feedback['positive_feedback']],
                    'Negative': [gpt4_feedback['negative_feedback']]
                })

        # 4. Cost Analysis
        st.subheader("Cost Analysis")
        cost_query = """
        SELECT 
            DATE(timestamp) as date,
            model_type,
            COUNT(*) * CASE 
                WHEN model_type = 'gpt-4' THEN 0.03
                ELSE 0.002
            END as estimated_cost,
            COUNT(*) as request_count
        FROM chat_messages
        WHERE role = 'assistant'
        AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY DATE(timestamp), model_type
        ORDER BY date;
        """
        
        cost_df, cost_debug = self.execute_query_with_debug(cost_query)
        if debug_mode:
            with st.expander("Cost Analysis Query"):
                st.code(cost_debug, language='sql')
        
        if not cost_df.empty:
            # Daily cost trend
            st.line_chart(cost_df.pivot(index='date', columns='model_type', values='estimated_cost'))
            
            # Total cost summary
            total_cost = cost_df['estimated_cost'].sum()
            st.metric("Total Estimated Cost", f"${total_cost:.2f}")
            
            # Cost breakdown by model
            cost_breakdown = cost_df.groupby('model_type').agg({
                'estimated_cost': 'sum',
                'request_count': 'sum'
            }).round(2)
            cost_breakdown['cost_per_request'] = (
                cost_breakdown['estimated_cost'] / cost_breakdown['request_count']
            ).round(3)
            st.dataframe(cost_breakdown)

        # Download data option
        if st.button("Download Analytics Data"):
            # Combine all relevant data
            download_data = {
                'usage': usage_df.to_dict() if not usage_df.empty else {},
                'length_analysis': length_df.to_dict() if not length_df.empty else {},
                'feedback': feedback_df.to_dict() if not feedback_df.empty else {},
                'cost': cost_df.to_dict() if not cost_df.empty else {}
            }
            
            # Convert to JSON
            json_str = json.dumps(download_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="analytics_data.json",
                mime="application/json"
            )

class AnalyticsDashboard:
    def __init__(self, db_connection):
        self.conn = db_connection

    def get_detailed_metrics(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Get comprehensive metrics with proper token and cost calculations"""
        
        # Usage and cost metrics with proper grouping
        usage_query = """
        WITH daily_stats AS (
            SELECT 
                DATE(timestamp) as date,
                model_type,
                COUNT(*) as request_count,
                SUM(tokens_used) as total_tokens,
                AVG(tokens_used) as avg_tokens,
                AVG(response_time_ms) as avg_response_time,
                SUM(CASE 
                    WHEN model_type = 'gpt-4' THEN tokens_used * 0.03 / 1000.0
                    ELSE tokens_used * 0.002 / 1000.0
                END) as estimated_cost
            FROM chat_messages
            WHERE role = 'assistant'
            AND timestamp >= CURRENT_DATE - INTERVAL '{} days'
            GROUP BY DATE(timestamp), model_type
        )
        SELECT 
            date,
            model_type,
            request_count,
            total_tokens,
            avg_tokens,
            avg_response_time,
            estimated_cost,
            SUM(request_count) OVER (PARTITION BY model_type ORDER BY date) as cumulative_requests,
            SUM(estimated_cost) OVER (PARTITION BY model_type ORDER BY date) as cumulative_cost
        FROM daily_stats
        ORDER BY date, model_type
        """.format(days)  # Use string formatting instead of parameter
        
        usage_df = pd.read_sql_query(usage_query, self.conn)
        
        # Response time distribution
        time_query = """
        SELECT 
            model_type,
            response_time_ms,
            tokens_used,
            timestamp::date as date
        FROM chat_messages
        WHERE role = 'assistant'
        AND timestamp >= CURRENT_DATE - INTERVAL '{} days'
        AND response_time_ms IS NOT NULL
        """.format(days)
        
        time_df = pd.read_sql_query(time_query, self.conn)

        # Feedback analysis with model type
        feedback_query = """
        SELECT 
            cm.model_type,
            DATE(mf.timestamp) as date,
            COUNT(CASE WHEN rating = true THEN 1 END) as positive,
            COUNT(CASE WHEN rating = false THEN 1 END) as negative
        FROM message_feedback mf
        JOIN chat_messages cm ON mf.message_id = cm.id
        WHERE mf.timestamp >= CURRENT_DATE - INTERVAL '{} days'
        GROUP BY cm.model_type, DATE(mf.timestamp)
        ORDER BY date
        """.format(days)
        
        feedback_df = pd.read_sql_query(feedback_query, self.conn)

        return {
            'usage': usage_df,
            'time': time_df,
            'feedback': feedback_df
        }

    def format_number(self, num: float) -> str:
        """Format numbers for display"""
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        if num >= 1000:
            return f"{num/1000:.1f}K"
        return f"{num:.0f}"

    def render_dashboard(self):
        st.title("Analytics Dashboard 📊")
        
        # Date range selector
        days = st.selectbox(
            "Select time range",
            options=[7, 30, 90],
            format_func=lambda x: f"Last {x} days"
        )

        try:
            # Get all metrics
            metrics = self.get_detailed_metrics(days)
            usage_df = metrics['usage']

            if usage_df.empty:
                st.warning("No usage data available for the selected time range.")
                return

            # Top-level metrics
            st.subheader("Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_requests = usage_df['request_count'].sum()
                st.metric("Total Requests", self.format_number(total_requests))
            
            with col2:
                total_cost = usage_df['estimated_cost'].sum()
                st.metric("Total Cost", f"${total_cost:.2f}")
            
            with col3:
                avg_tokens = usage_df['avg_tokens'].mean()
                st.metric("Avg Tokens/Request", self.format_number(avg_tokens))
            
            with col4:
                avg_response_time = usage_df['avg_response_time'].mean()
                st.metric("Avg Response Time", f"{self.format_number(avg_response_time)}ms")

            # Token Usage Patterns
            st.subheader("Token Usage Patterns")
            
            fig_tokens = px.line(
                usage_df,
                x='date',
                y='avg_tokens',
                color='model_type',
                title='Average Token Usage by Model',
                labels={'avg_tokens': 'Average Tokens', 'date': 'Date', 'model_type': 'Model'}
            )
            fig_tokens.update_layout(yaxis_title="Tokens per Request")
            st.plotly_chart(fig_tokens, use_container_width=True)

            # Cost Analysis
            st.subheader("Cost Analysis")
            
            # Daily cost by model
            fig_cost = px.line(
                usage_df,
                x='date',
                y='estimated_cost',
                color='model_type',
                title='Daily Cost by Model',
                labels={'estimated_cost': 'Cost (USD)', 'date': 'Date', 'model_type': 'Model'}
            )
            fig_cost.update_layout(yaxis_title="Cost (USD)")
            st.plotly_chart(fig_cost, use_container_width=True)

            # Model Usage Comparison
            st.subheader("Model Usage Comparison")
            model_comparison = usage_df.groupby('model_type').agg({
                'request_count': 'sum',
                'estimated_cost': 'sum',
                'avg_tokens': 'mean',
                'avg_response_time': 'mean'
            }).round(2)
            
            # Rename columns for better display
            model_comparison.columns = [
                'Total Requests',
                'Total Cost ($)',
                'Avg Tokens',
                'Avg Response Time (ms)'
            ]
            st.dataframe(model_comparison)

            # Download data option
            if st.button("Download Analytics Data"):
                csv = usage_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="analytics_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
            st.exception(e)

class MentalHealthRAG:
    def __init__(self):
        """Initialize RAG components"""
        # Initialize OpenAI with API key from environment
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Elasticsearch
        self.es = Elasticsearch(['http://localhost:9200'])
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Initialize database manager
        self.db = DatabaseManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant context with similarity scores"""
        try:
            query_embedding = self.embedding_model.encode(query)
            
            search_query = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 0.3
                                    }
                                }
                            }
                        ]
                    }
                },
                "knn": {
                    "field": "vector_combined_knn",
                    "query_vector": query_embedding.tolist(),
                    "k": k,
                    "num_candidates": 100,
                    "boost": 0.7
                }
            }
            
            response = self.es.search(index="qa_index_2", body=search_query)
            
            # Enhanced context information with scores
            contexts = []
            for hit in response['hits']['hits']:
                contexts.append({
                    'text': hit['_source'].get('text', ''),
                    'score': hit['_score'],
                    'tag': hit['_source'].get('tag', 'unknown')
                })
            
            return contexts
            
        except Exception as e:
            self.logger.error(f"Error in context retrieval: {str(e)}")
            return []

    def generate_response(self, query: str, contexts: List[Dict], model: str = "gpt-3.5-turbo") -> tuple:
        """Generate response using OpenAI"""
        try:
            start_time = time.time()
            
            context_text = "\n".join([c.get('text', '') for c in contexts])
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a compassionate AI mental health assistant. 
                    Use this context to provide support: {context_text}
                    
                    Guidelines:
                    - Be empathetic and understanding
                    - Provide practical, actionable advice
                    - Encourage professional help when appropriate
                    - Keep responses concise but helpful"""
                },
                {"role": "user", "content": query}
            ]
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            response_time = int((time.time() - start_time) * 1000)
            tokens_used = response.usage.total_tokens
            
            return (response.choices[0].message.content, response_time, tokens_used)
            
        except Exception as e:
            self.logger.error(f"Error in response generation: {str(e)}")
            return ("I apologize, but I'm having trouble generating a response. Please try again.", 0, 0)

def render_warnings_and_info():
    """Render comprehensive warnings and information"""
    st.markdown("""
    ## Welcome to the Mental Health Support Assistant 🤗
    
    This AI assistant is designed to provide supportive listening and general mental health information. 
    While it aims to be helpful, please remember:
    """)
    
    # Critical Warnings
    st.error("""
    ⚠️ **IMPORTANT SAFETY INFORMATION:**
    - This is NOT a replacement for professional mental health care
    - In case of emergency or crisis, contact emergency services immediately
    - If you're having thoughts of self-harm or suicide, please call 988 (US) immediately
    """)
    
    # Main Information
    with st.expander("ℹ️ About This Service"):
        st.markdown("""
        This assistant can:
        - Provide empathetic listening and support
        - Offer general mental health information
        - Suggest coping strategies and self-help resources
        - Help you understand when to seek professional help
        
        This assistant cannot:
        - Diagnose mental health conditions
        - Provide medical advice or treatment
        - Replace professional mental health care
        - Handle emergency situations
        """)
    
    # Resources
    with st.expander("🆘 Emergency Resources"):
        st.markdown("""
        **24/7 Crisis Support:**
        - 988 Suicide & Crisis Lifeline (US): Call or text 988
        - Crisis Text Line: Text HOME to 741741
        - Emergency Services: 911 (US)
        
        **Find Professional Help:**
        - [Psychology Today Therapist Finder](https://www.psychologytoday.com/us/therapists)
        - [SAMHSA Treatment Locator](https://findtreatment.gov/)
        - [National Alliance on Mental Illness (NAMI)](https://www.nami.org/help)
        """)
    
    # Privacy Notice
    with st.expander("🔒 Privacy Information"):
        st.markdown("""
        **What we collect:**
        - Chat messages and responses
        - Usage statistics for service improvement
        - Anonymous feedback data
        
        **What we DON'T collect:**
        - Personal identification information
        - Medical or health records
        - Location data
        
        All data is anonymized and used only for service improvement.
        """)

def render_debug_info(response_data: Dict):
    """Enhanced debug information display without nested expanders"""
    with st.expander("🔍 Debug Information", expanded=True):
        # Response metrics
        st.markdown("### Response Metrics")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Tokens Used", response_data.get('tokens', 0))
        with cols[1]:
            st.metric("Response Time", f"{response_data.get('response_time', 0)}ms")
        with cols[2]:
            st.metric("Model Used", response_data.get('model', 'unknown'))
        
        # Context display
        st.markdown("### Retrieved Contexts")
        if response_data.get('contexts'):
            # Create a clean table-like display for contexts
            for idx, context in enumerate(response_data['contexts'], 1):
                st.markdown(f"""
                ---
                **Context {idx}**
                - **Relevance Score:** {context.get('score', 0):.3f}
                - **Tag:** {context.get('tag', 'unknown')}
                - **Text:** {context.get('text', 'No text available')}
                """)
        else:
            st.warning("No contexts were retrieved")

def main():
    st.set_page_config(page_title="Mental Health Support Assistant", layout="wide")

    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MentalHealthRAG()

    # Initialize chat history with proper message IDs
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0

    # Create tabs
    tab1, tab2 = st.tabs(["Chat", "Analytics"])

    with tab1:
        st.title("Mental Health Support Assistant 🤗")
        render_warnings_and_info()
        
        # Model selection in sidebar
        with st.sidebar:
            model = st.selectbox(
                "Select Model",
                ["gpt-3.5-turbo", "gpt-4"]
            )

        # Chat interface with proper message handling
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                # Only show feedback buttons for assistant messages
                if msg["role"] == "assistant":
                    # Create unique keys using both index and message ID
                    message_id = msg.get('id', f'temp_{idx}')
                    cols = st.columns([10, 1, 1])
                    with cols[1]:
                        if st.button("👍", key=f"up_{message_id}_{idx}"):
                            st.session_state.rag_system.db.log_feedback(message_id, True)
                    with cols[2]:
                        if st.button("👎", key=f"down_{message_id}_{idx}"):
                            st.session_state.rag_system.db.log_feedback(message_id, False)

        if prompt := st.chat_input():
            # Increment message counter for temporary IDs
            st.session_state.message_counter += 1
            temp_id = f'temp_{st.session_state.message_counter}'
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "id": temp_id
            })
            
            with st.chat_message("user"):
                st.write(prompt)

            # Log user message
            user_msg_id = st.session_state.rag_system.db.log_message(
                "user", prompt, model
            )

            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                debug_placeholder = st.empty()
                
                # Retrieve contexts
                contexts = st.session_state.rag_system.retrieve_context(prompt)
                
                # Generate response
                response, response_time, tokens = st.session_state.rag_system.generate_response(
                    prompt, contexts, model
                )
                
                message_placeholder.write(response)

                # Log assistant message
                assistant_msg_id = st.session_state.rag_system.db.log_message(
                    "assistant", response, model, tokens, response_time
                )
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "id": assistant_msg_id
                })

                # Enhanced debug information
                with st.expander("🔍 Debug Information", expanded=False):
                    # Response metrics
                    st.markdown("### Response Metrics")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Response Time", f"{response_time}ms")
                    with cols[1]:
                        st.metric("Tokens Used", tokens)
                    with cols[2]:
                        st.metric("Model", model)
                    with cols[3]:
                        st.metric("Message ID", assistant_msg_id)
                    
                    # Context information
                    st.markdown("### Retrieved Contexts")
                    if contexts:
                        for idx, context in enumerate(contexts, 1):
                            with st.container():
                                st.markdown(f"**Context {idx}**")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"Text: {context.get('text', 'No text available')}")
                                with col2:
                                    if 'score' in context:
                                        st.markdown(f"Score: {context.get('score', 0):.3f}")
                                    if 'tag' in context:
                                        st.markdown(f"Tag: {context.get('tag', 'unknown')}")
                                st.markdown("---")
                    else:
                        st.warning("No contexts were retrieved")
                    
                    # Prompt information
                    st.markdown("### Prompt Details")
                    st.markdown(f"**User Query:** {prompt}")
                    
                    # Token usage breakdown (if available)
                    if hasattr(response, 'usage'):
                        st.markdown("### Token Usage Breakdown")
                        token_cols = st.columns(3)
                        with token_cols[0]:
                            st.metric("Prompt Tokens", response.usage.prompt_tokens)
                        with token_cols[1]:
                            st.metric("Completion Tokens", response.usage.completion_tokens)
                        with token_cols[2]:
                            st.metric("Total Tokens", response.usage.total_tokens)

                # You might also want to log these contexts
                if assistant_msg_id:
                    # If you have a table for storing contexts
                    for context in contexts:
                        try:
                            st.session_state.rag_system.db.log_context(
                                message_id=assistant_msg_id,
                                context_text=context.get('text', ''),
                                relevance_score=context.get('score', 0)
                            )
                        except Exception as e:
                            st.error(f"Failed to log context: {str(e)}")




                    
#    with tab2:
#        analytics = AnalyticsDashboard(st.session_state.rag_system.db.conn)
#        analytics.render_dashboard()

    with tab2:  # Analytics tab
        try:
            analytics = SimpleAnalyticsDashboard(st.session_state.rag_system.db.conn)
            analytics.render_dashboard()
        except Exception as e:
            st.error(f"Error initializing analytics: {str(e)}")
            st.exception(e)        

if __name__ == "__main__":
    main()
