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
ENV SKIP_MODELS_ON_STARTUP=true

# Add build caching to speed up rebuilds
ARG BUILDKIT_INLINE_CACHE=1

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

# Install essential packages first for faster startup
COPY requirements-minimal.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for models
RUN mkdir -p /app/models/transformers_cache /app/utils

# Copy utility modules first
COPY utils/__init__.py /app/utils/
COPY utils/pdf_processor.py /app/utils/
COPY utils/text_extraction.py /app/utils/
COPY utils/insight_generator.py /app/utils/

# Copy the startup script
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Copy application code
COPY app.py /app/

# Create placeholder models directory
RUN mkdir -p /app/models
RUN touch /app/models/.keep

# Copy full requirements and install remaining dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt --timeout 600

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the models if they exist
COPY models/ /app/models/

# Expose port
EXPOSE 8080

# Use the startup script
CMD ["./startup.sh"]
