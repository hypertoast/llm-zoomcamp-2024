import streamlit as st
import psycopg2
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from utils import init_db_connection, generate_query_embedding, retrieve_context
from utils import generate_response_with_context, generate_standalone_response, calculate_token_usage

# Initialize database connection
conn = init_db_connection()


def plot_avg_response_time(conn):
    query = """
    SELECT 
        DATE(timestamp) AS date,
        AVG(response_time) AS avg_response_time
    FROM user_feedback
    GROUP BY date
    ORDER BY date;
    """
    data = pd.read_sql(query, conn)
    st.line_chart(data.set_index("date")["avg_response_time"], width=0, height=0, use_container_width=True)


def plot_avg_response_length(conn):
    query = """
    SELECT 
        DATE(timestamp) AS date,
        AVG(LENGTH(rag_response)) AS avg_rag_response_length,
        AVG(LENGTH(standalone_response)) AS avg_standalone_response_length
    FROM user_feedback
    GROUP BY date
    ORDER BY date;
    """
    data = pd.read_sql(query, conn)
    st.line_chart(data.set_index("date")[["avg_rag_response_length", "avg_standalone_response_length"]], width=0, height=0, use_container_width=True)


def plot_avg_query_length(conn):
    query = """
    SELECT 
        DATE(timestamp) AS date,
        AVG(LENGTH(query)) AS avg_query_length
    FROM user_feedback
    GROUP BY date
    ORDER BY date;
    """
    data = pd.read_sql(query, conn)
    st.line_chart(data.set_index("date")["avg_query_length"], width=0, height=0, use_container_width=True)


def plot_avg_tokens_per_day(conn):
    query = """
    SELECT 
        DATE(timestamp) AS date,
        AVG(total_cost / 0.02 * 1000) AS avg_tokens_used
    FROM user_feedback
    GROUP BY date
    ORDER BY date;
    """
    data = pd.read_sql(query, conn)
    st.line_chart(data.set_index("date")["avg_tokens_used"], width=0, height=0, use_container_width=True)


def fetch_daily_query_volume():
    """Retrieve the daily query volume data from the database."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DATE(timestamp) AS date, COUNT(*) AS query_count
                FROM user_feedback
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=["Date", "Query Volume"])
    except Exception as e:
        st.error("Error fetching daily query volume data: " + str(e))
        return pd.DataFrame()  # Return an empty DataFrame if error

def fetch_feedback_type_over_time():
    """Retrieve feedback type distribution data over time from the database."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DATE(timestamp) AS date,
                       SUM(CASE WHEN rag_feedback = '👍 Yes' THEN 1 ELSE 0 END) AS rag_positive,
                       SUM(CASE WHEN rag_feedback = '👎 No' THEN 1 ELSE 0 END) AS rag_negative,
                       SUM(CASE WHEN standalone_feedback = '👍 Yes' THEN 1 ELSE 0 END) AS standalone_positive,
                       SUM(CASE WHEN standalone_feedback = '👎 No' THEN 1 ELSE 0 END) AS standalone_negative
                FROM user_feedback
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=["Date", "RAG Positive", "RAG Negative", "Standalone Positive", "Standalone Negative"])
    except Exception as e:
        st.error("Error fetching feedback type over time data: " + str(e))
        return pd.DataFrame()  # Return an empty DataFrame if error

def plot_daily_query_volume(data):
    """Plot daily query volume as a line chart."""
    if data.empty:
        st.write("No data available for daily query volume.")
        return

    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["Query Volume"], marker="o", color="#1f77b4")
    ax.set_xlabel("Date")
    ax.set_ylabel("Query Volume")
    ax.set_title("Daily Query Volume")
    st.pyplot(fig)

def plot_feedback_type_over_time(data):
    """Plot feedback type distribution over time as stacked area chart."""
    if data.empty:
        st.write("No data available for feedback type distribution over time.")
        return

    fig, ax = plt.subplots()
    ax.stackplot(data["Date"], data["RAG Positive"], data["RAG Negative"], data["Standalone Positive"], data["Standalone Negative"], 
                 labels=["RAG Positive", "RAG Negative", "Standalone Positive", "Standalone Negative"], 
                 colors=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Feedback Count")
    ax.set_title("Feedback Type Distribution Over Time")
    ax.legend(loc="upper left")
    st.pyplot(fig)



def fetch_response_time_data():
    """Retrieve average response time by feedback type."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT rag_feedback, AVG(response_time) AS avg_response_time
                FROM user_feedback
                GROUP BY rag_feedback
            """)
            data = cursor.fetchall()
            # Convert the results into a DataFrame
            return pd.DataFrame(data, columns=["Feedback Type", "Average Response Time"])
    except Exception as e:
        st.error("Error fetching response time data: " + str(e))
        return pd.DataFrame()  # Return an empty DataFrame if error

def fetch_cost_comparison_data():
    """Retrieve cost comparison data for RAG and standalone responses over time."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DATE(timestamp) AS date,
                       SUM(CASE WHEN rag_feedback = '👍 Yes' THEN total_cost ELSE 0 END) AS rag_cost,
                       SUM(CASE WHEN standalone_feedback = '👍 Yes' THEN total_cost ELSE 0 END) AS standalone_cost
                FROM user_feedback
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            data = cursor.fetchall()
            # Convert the results into a DataFrame
            return pd.DataFrame(data, columns=["Date", "RAG Cost", "Standalone Cost"])
    except Exception as e:
        st.error("Error fetching cost comparison data: " + str(e))
        return pd.DataFrame()  # Return an empty DataFrame if error

def plot_response_time_by_feedback(data):
    """Plot average response time by feedback type as a bar chart."""
    if data.empty:
        st.write("No data available for response time by feedback type.")
        return

    fig, ax = plt.subplots()
    ax.bar(data["Feedback Type"], data["Average Response Time"], color=["#1f77b4", "#ff7f0e"])
    ax.set_xlabel("Feedback Type")
    ax.set_ylabel("Average Response Time (s)")
    ax.set_title("Average Response Time by Feedback Type")
    st.pyplot(fig)

def plot_cost_comparison(data):
    """Plot cost comparison over time as a line chart."""
    if data.empty:
        st.write("No data available for cost comparison.")
        return

    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["RAG Cost"], label="RAG Cost", marker="o")
    ax.plot(data["Date"], data["Standalone Cost"], label="Standalone Cost", marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Token Cost")
    ax.set_title("Cost Comparison Between RAG and Standalone Over Time")
    ax.legend()
    st.pyplot(fig)


def fetch_feedback_data():
    """Retrieve feedback counts for RAG and standalone responses."""
    query = """
        SELECT 
            COUNT(*) FILTER (WHERE rag_feedback = '👍 Yes') AS rag_positive,
            COUNT(*) FILTER (WHERE rag_feedback = '👎 No') AS rag_negative,
            COUNT(*) FILTER (WHERE standalone_feedback = '👍 Yes') AS standalone_positive,
            COUNT(*) FILTER (WHERE standalone_feedback = '👎 No') AS standalone_negative
        FROM user_feedback
    """
    return pd.read_sql(query, conn)

def fetch_cost_data():
    """Retrieve token costs per day for RAG and standalone responses."""
    query = """
        SELECT 
            DATE(timestamp) AS date,
            SUM(CASE WHEN rag_feedback = '👍 Yes' THEN total_cost ELSE 0 END) AS rag_cost,
            SUM(CASE WHEN standalone_feedback = '👍 Yes' THEN total_cost ELSE 0 END) AS standalone_cost
        FROM user_feedback
        GROUP BY DATE(timestamp)
        ORDER BY date
    """
    return pd.read_sql(query, conn)

def plot_feedback_distribution(data):
    """Plot feedback distribution as a bar chart."""
    labels = ['RAG Positive', 'RAG Negative', 'Standalone Positive', 'Standalone Negative']
    counts = [data['rag_positive'][0], data['rag_negative'][0], data['standalone_positive'][0], data['standalone_negative'][0]]

    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=['#4CAF50', '#FF5252', '#4CAF50', '#FF5252'])
    ax.set_ylabel("Feedback Count")
    ax.set_title("Feedback Distribution (RAG vs. Standalone)")
    st.pyplot(fig)

def plot_cost_analysis(data):
    """Plot token costs over time as a line chart."""
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['rag_cost'], label='RAG Cost', marker='o')
    ax.plot(data['date'], data['standalone_cost'], label='Standalone Cost', marker='x')
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Token Cost")
    ax.set_title("Token Cost Over Time (RAG vs. Standalone)")
    ax.legend()
    st.pyplot(fig)

# Function to display detailed debug info
def display_debug_info(query, rag_context, rag_cost, standalone_cost):
    st.write("**Query:**", query)
    st.write("**Retrieved Context (RAG):**", rag_context)
    st.write("**Token Costs - RAG:**", rag_cost)
    st.write("**Token Costs - Standalone:**", standalone_cost)

# Helper functions for saving feedback and displaying analytics
def save_user_feedback(conn, query, rag_response, standalone_response, rag_feedback, standalone_feedback, cost):
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_feedback (query, rag_response, standalone_response, rag_feedback, standalone_feedback, total_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (query, rag_response, standalone_response, rag_feedback, standalone_feedback, cost, datetime.now())
            )
            conn.commit()
            st.success("Feedback saved!")
    except Exception as e:
        st.error("Error saving feedback: " + str(e))

def display_analytics(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    AVG(total_cost) AS avg_cost,
                    SUM(CASE WHEN rag_feedback = '👍 Yes' THEN 1 ELSE 0 END) AS rag_positive,
                    SUM(CASE WHEN rag_feedback = '👎 No' THEN 1 ELSE 0 END) AS rag_negative
                FROM user_feedback
            """)
            result = cursor.fetchone()
            st.subheader("Analytics Overview")
            st.write("**Average Token Cost per Query:**", result[0])
            st.write("**RAG Positive Feedback:**", result[1])
            st.write("**RAG Negative Feedback:**", result[2])

            # Additional Feedback Insights
            cursor.execute("""
                SELECT COUNT(*) as total_feedback 
                FROM user_feedback
            """)
            feedback = cursor.fetchone()
            st.write("**Total Feedback Entries:**", feedback[0])

    except Exception as e:
        st.error("Error displaying analytics: " + str(e))

# Main application function with sidebar navigation
# Main application function with sidebar navigation
def main():
    st.set_page_config(page_title="Mental Health Q&A App", layout="wide")
    st.sidebar.title("Navigation")
    
    # Sidebar Menu
    page = st.sidebar.radio("Go to", ["Home", "Analytics", "Settings"])

    # Initialize session states
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    if "enable_llm_response" not in st.session_state:
        st.session_state.enable_llm_response = False  # Default to enabled

    # Home page
    if page == "Home":
        st.title("Mental Health RAG-Based Q&A")

        # User query input
        query = st.text_input("Enter your question about mental health:")

        # Retrieve or initialize session state
        if "rag_response" not in st.session_state:
            st.session_state.rag_response = ""
            st.session_state.standalone_response = ""
            st.session_state.rag_cost = 0
            st.session_state.standalone_cost = 0
            st.session_state.rag_context = []

        # Generate responses if a new query is entered
        if query and (st.session_state.get("last_query") != query):
            st.session_state.last_query = query

            # Step 1: Retrieve context and generate RAG response
            st.session_state.rag_context = retrieve_context(query)
            st.session_state.rag_response, rag_tokens = generate_response_with_context(query, st.session_state.rag_context)
            st.session_state.rag_cost = calculate_token_usage(rag_tokens)

            # Step 2: Conditionally generate standalone response
            if st.session_state.enable_llm_response:
                st.session_state.standalone_response, standalone_tokens = generate_standalone_response(query)
                st.session_state.standalone_cost = calculate_token_usage(standalone_tokens)
            else:
                st.session_state.standalone_response = "LLM response generation is disabled. Please enable it in the Settings page on the left."
                st.session_state.standalone_cost = 0

        # Display responses in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RAG Response")
            st.write(st.session_state.rag_response)
            # Display feedback buttons only if a response exists
            if st.session_state.rag_response and st.button("👍 Yes, the RAG response is helpful"):
                save_user_feedback(
                    conn, query, st.session_state.rag_response, st.session_state.standalone_response, "👍 Yes", "👎 No",
                    st.session_state.rag_cost + st.session_state.standalone_cost
                )

            if st.session_state.rag_response and st.button("👎 No, the RAG response is not helpful"):
                save_user_feedback(
                    conn, query, st.session_state.rag_response, st.session_state.standalone_response, "👎 No", "👎 No",
                    st.session_state.rag_cost + st.session_state.standalone_cost
                )

        with col2:
            st.subheader("Standalone Response")
            st.write(st.session_state.standalone_response)
            # Display feedback buttons only if standalone response exists
            if st.session_state.enable_llm_response and st.session_state.standalone_response:
                if st.button("👍 Yes, the Standalone response is helpful"):
                    save_user_feedback(
                        conn, query, st.session_state.rag_response, st.session_state.standalone_response, "👎 No", "👍 Yes",
                        st.session_state.rag_cost + st.session_state.standalone_cost
                    )

                if st.button("👎 No, the Standalone response is not helpful"):
                    save_user_feedback(
                        conn, query, st.session_state.rag_response, st.session_state.standalone_response, "👎 No", "👎 No",
                        st.session_state.rag_cost + st.session_state.standalone_cost
                    )

        # Debug Information
        if st.session_state.debug_mode:
            with st.expander("Debug Information", expanded=True):
                display_debug_info(query, st.session_state.rag_context, st.session_state.rag_response, st.session_state.standalone_response, st.session_state.rag_cost, st.session_state.standalone_cost)
    

    elif page == "Analytics":
        st.title("Analytics Dashboard")
        display_analytics(conn)
        """Display Analytics Dashboard in the Analytics Tab."""

        st.subheader("Feedback Distribution")
        feedback_data = fetch_feedback_data()
        plot_feedback_distribution(feedback_data)
        
        st.subheader("Cost Analysis Over Time")
        cost_data = fetch_cost_data()
        plot_cost_analysis(cost_data)        

        # Average Response Time by Feedback Type
        st.subheader("Average Response Time by Feedback Type")
        response_time_data = fetch_response_time_data()
        plot_response_time_by_feedback(response_time_data)

        # Cost Comparison Between RAG and Standalone
        st.subheader("Cost Comparison Between RAG and Standalone Over Time")
        cost_comparison_data = fetch_cost_comparison_data()
        plot_cost_comparison(cost_comparison_data)

        # Daily Query Volume
        st.subheader("Daily Query Volume")
        daily_query_volume_data = fetch_daily_query_volume()
        plot_daily_query_volume(daily_query_volume_data)

        # Feedback Type Over Time
        st.subheader("Feedback Type Distribution Over Time")
        feedback_type_data = fetch_feedback_type_over_time()
        plot_feedback_type_over_time(feedback_type_data)

        # Average tokens used per day
        st.subheader("Average Tokens Used Per Day")
        plot_avg_tokens_per_day(conn)

        # Average characters used per question
        st.subheader("Average Characters Per Question")
        plot_avg_query_length(conn)

        # Average characters per response
        st.subheader("Average Characters Per Response")
        plot_avg_response_length(conn)

        # Average response time
        st.subheader("Average Response Times")
        plot_avg_response_time(conn)


    elif page == "Settings":
        st.title("Settings")
        st.write("Settings options such as API keys, thresholds, and configurations can be added here.")

        # Debug mode setting toggle
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
        st.write("Debug mode will show additional information on the Home page when enabled.")

        # In Settings Page
        st.subheader("LLM Settings")
        enable_llm_response = st.checkbox("Enable LLM Response Generation", value=True)


# Run the main function
if __name__ == "__main__":
    main()
