# ===========================================
# FastAPI RAG Chatbot Dockerfile
# ===========================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI on port 6000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
