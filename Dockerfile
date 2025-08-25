# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Streamlit uses 8501, FastAPI uses 8000)
EXPOSE 8501

# Default command (change to uvicorn for FastAPI)
CMD ["streamlit", "run", "dashboard.py"]
