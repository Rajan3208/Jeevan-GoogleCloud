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

# Install system dependencies (optimized to reduce image size)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create directories for models and utils
RUN mkdir -p /app/models/transformers_cache /app/utils

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install CPU-only PyTorch first to avoid NVIDIA dependency issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install packages in batches to avoid dependency conflicts
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    Werkzeug

RUN pip install --no-cache-dir \
    pypdfium2 \
    PyPDF2 \
    pytesseract \
    Pillow \
    numpy

RUN pip install --no-cache-dir \
    scikit-learn \
    spacy

RUN pip install --no-cache-dir \
    tensorflow \
    keras \
    transformers

RUN pip install --no-cache-dir \
    easyocr \
    langchain \
    langchain-community \
    unstructured \
    python-multipart

# Copy utility modules
COPY utils/ /app/utils/

# Copy the startup script and make it executable
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Copy application code
COPY app.py /app/
COPY train.py /app/
COPY README.md /app/

# Create models directory and copy any existing models
RUN mkdir -p /app/models
COPY models/ /app/models/

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Expose port
EXPOSE 8080

# Command to run the app
CMD ["/app/startup.sh"]
