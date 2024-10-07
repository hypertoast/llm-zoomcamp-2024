# Step 1: Use an official Python runtime as a parent image
FROM python:3.8-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the local files into the container
COPY . /app

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose port 8501 for Streamlit
EXPOSE 8501

# Step 6: Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
