#!/bin/bash

# Print startup information
echo "---------------------------------------------"
echo "Starting PDF Analysis API service..."
echo "Current environment:"
echo "PORT: $PORT"
echo "SKIP_MODELS_ON_STARTUP: $SKIP_MODELS_ON_STARTUP"
echo "PYTHONPATH: $PYTHONPATH"
echo "---------------------------------------------"

# Ensure the PORT is properly set
if [ -z "$PORT" ]; then
  echo "PORT environment variable not set, defaulting to 8080"
  export PORT=8080
fi

# Create and check necessary directories
echo "Checking required directories..."
mkdir -p /app/models/transformers_cache
echo "Directory structure confirmed."

# Check if the application should load models during startup
if [ "$SKIP_MODELS_ON_STARTUP" = "true" ]; then
  echo "Model preloading is disabled. Models will be loaded on demand."
else
  echo "Attempting to preload lightweight models (with timeout protection)..."
  # Use timeout to prevent hanging during startup - 60 seconds max
  timeout 60 python -c "from utils.insight_generator import load_models; load_models(preload_all=False)" || {
    echo "WARNING: Model preloading failed or timed out, but continuing startup process."
    echo "Models will be loaded on demand when endpoints are accessed."
  }
fi

# Print final startup message
echo "---------------------------------------------"
echo "Starting Gunicorn server on port $PORT"
echo "$(date)"
echo "---------------------------------------------"

# Start with carefully tuned Gunicorn settings
# - Single worker to prevent memory issues
# - 8 threads for concurrent requests
# - 5-minute timeout for long-running operations
# - Check interval of 10s to quickly detect failures
exec gunicorn \
  --bind :$PORT \
  --workers 1 \
  --threads 8 \
  --timeout 300 \
  --graceful-timeout 120 \
  --keep-alive 5 \
  --check-interval 10 \
  --log-level info \
  app:app
