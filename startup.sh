#!/bin/bash
echo "Starting up service..."
echo "Creating necessary directories..."

# Set up any required environment checks
echo "Environment: PORT=$PORT"
echo "Environment: SKIP_MODELS_ON_STARTUP=$SKIP_MODELS_ON_STARTUP"

# Ensure port is set
if [ -z "$PORT" ]; then
  echo "PORT is not set, defaulting to 8080"
  export PORT=8080
fi

# Check if we should skip model loading on startup
if [ "$SKIP_MODELS_ON_STARTUP" = "true" ]; then
  echo "Skipping model loading on startup (will load on-demand)"
else
  echo "Attempting to load lightweight models..."
  # Use timeout to prevent hanging during startup
  timeout 60 python -c "from utils.insight_generator import load_models; load_models(preload_all=False)" || echo "Model loading failed or timed out, but continuing startup"
fi

# Start the application with more generous timeouts
echo "Starting server with Gunicorn on port $PORT..."
exec gunicorn --bind :$PORT \
  --workers 1 \
  --threads 8 \
  --timeout 300 \
  --graceful-timeout 120 \
  --keep-alive 5 \
  --log-level info \
  --preload \
  app:app
