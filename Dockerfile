# EDAgent Production Dockerfile for Render / Cloud Hosting
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

# Install system dependencies needed for compiling packages and image rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and sample data
COPY edagent/ ./edagent/
COPY sample_data/ ./sample_data/
COPY README.md .

# Create non-root user for security sandbox compliance
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Run FastAPI service
CMD ["uvicorn", "edagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
