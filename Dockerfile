# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables for the application
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Install system dependencies (optional, but good practice for building tools)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# NOTE: The 'chroma_db' and 'data' directories are excluded from copying here 
# as they will be created at runtime.
COPY . $APP_HOME

# Expose the port FastAPI runs on (default for uvicorn)
EXPOSE 6000

# The command to run the application using Uvicorn
# 'main' is the file (main.py), 'app' is the FastAPI instance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6000"]