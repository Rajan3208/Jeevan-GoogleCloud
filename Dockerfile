# Use official Python image with slim version for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2
ENV PYTHONMALLOC=malloc
ENV PORT=8080
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/models/transformers_cache

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

# Install Python dependencies with extended timeout
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir -r requirements.txt --timeout 600

# Install tf-keras for transformers compatibility
RUN pip install --no-cache-dir tf-keras

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Create directories for models
RUN mkdir -p /app/models/transformers_cache

# Copy the models directory first (to leverage Docker layer caching for large files)
COPY models/ /app/models/

# Copy utility modules
COPY utils/ /app/utils/

# Copy the startup script
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Copy the rest of the application code
COPY *.py /app/

# Expose port
EXPOSE 8080

# Use the startup script
CMD ["./startup.sh"]
