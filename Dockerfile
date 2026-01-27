# FROM python:3.10-slim

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# # Copy application
# COPY . .

# # Expose port
# EXPOSE 8000

# # Run application
# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies (Cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the application
COPY . .

# Use a default port but allow override via ENV (for GCP/AWS compatibility)
ENV PORT=8000
EXPOSE 8000

# Using "exec" form for better signal handling (Ctrl+C works better)
CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT