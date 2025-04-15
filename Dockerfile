# Use official Python image with slim version for smaller size
FROM python:3.10-slim

# Set working directory - no app subdirectory
WORKDIR /

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2
ENV PYTHONMALLOC=malloc
ENV PORT=8080
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/models/transformers_cache
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

# Create directories for models
RUN mkdir -p /models/transformers_cache /utils

# Install essential packages
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt --timeout 600

# Copy utility modules
COPY utils/__init__.py /utils/
COPY utils/pdf_processor.py /utils/
COPY utils/text_extraction.py /utils/
COPY utils/insight_generator.py /utils/

# Copy the startup script to root directory
COPY startup.sh /
RUN chmod +x /startup.sh

# Copy application code
COPY app.py /

# Create placeholder models directory
RUN mkdir -p /models
RUN touch /models/.keep

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the models if they exist
COPY models/ /models/

# Expose port
EXPOSE 8080

# Use the startup script with absolute path
CMD ["/startup.sh"]
