# Use official Python image with slim version for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install tf-keras for transformers compatibility
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir tf-keras

# Install Python dependencies with extended timeout
RUN pip install --no-cache-dir -r requirements.txt --timeout 600

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Create models directory
RUN mkdir -p /app/models

# Copy the models directory first (to leverage Docker layer caching for large files)
COPY models/ /app/models/

# Install tf-keras for compatibility
RUN pip install --no-cache-dir tf-keras

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run using Gunicorn with a reasonable timeout
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "300", "app:app"]
