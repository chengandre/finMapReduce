# Multi-stage Dockerfile for MapReduce QA Webapp
# Stage 1: Base system with Python and system dependencies
FROM python:3.10.18-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for PDF processing and marker CLI
RUN apt-get update && apt-get install -y \
    # Basic build tools
    build-essential \
    # PDF processing dependencies
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    # Marker CLI dependencies
    pandoc \
    # Network tools for downloading
    wget \
    curl \
    # Git for potential package installs
    git \
    # Image processing
    libmagic1 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements files
COPY webapp/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install marker CLI if available (optional dependency)
RUN pip install marker-pdf || echo "Warning: marker-pdf not available, will fallback to PyPDF"

# Stage 3: Application
FROM dependencies as application

# Copy the entire project (needed for parent directory imports)
COPY . .

# Create necessary directories
RUN mkdir -p /app/cache/marker \
    /app/cache/pdf_cache \
    /app/cache/prompts_log \
    /tmp/webapp_uploads

# Set proper permissions
RUN chown -R app:app /app /tmp/webapp_uploads

# Copy entrypoint script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown app:app /entrypoint.sh

# Switch to app user
USER app

# Set working directory to webapp backend for running the app
WORKDIR /app/webapp/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "main.py"]