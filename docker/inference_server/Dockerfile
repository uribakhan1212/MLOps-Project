# docker/inference_server/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy inference-specific requirements
COPY docker/inference_server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only inference server code and required assets
RUN mkdir -p src
COPY src/inference_server.py ./src/inference_server.py
COPY models/ ./models/

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1

# Run server
CMD ["python", "./src/inference_server.py"]