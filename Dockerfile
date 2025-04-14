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

# Install Python dependencies (split into parts to avoid timeout)
COPY requirements.txt requirements-minimal.txt requirements-rest.txt ./

# Install minimal requirements first
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Install torch separately with extended timeout
RUN pip install --no-cache-dir torch --timeout 600

# Install the remaining requirements
RUN pip install --no-cache-dir -r requirements-rest.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy all application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# NOTE: Better to train models outside Docker and copy them instead of training during build
# COPY pre-trained models (do this only if they already exist)
COPY models/* /app/models/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run using Gunicorn with a reasonable timeout (e.g., 300s)
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "300", "app:app"]
