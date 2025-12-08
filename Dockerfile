# Use Python base image
FROM python:3.11-slim

# Install system dependencies in one layer for better caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Upgrade pip (use cache for faster builds)
RUN pip install --upgrade pip wheel setuptools

# Copy Python requirements first (better layer caching)
COPY requirements.txt ./

# Install Python dependencies (use pip cache, install in order of size)
# Install smaller packages first, then OpenCV (largest)
# python-Levenshtein is optional - if it fails, thefuzz will work without it (just slower)
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    python-multipart==0.0.12 \
    pytesseract==0.3.10 \
    Pillow==10.4.0 \
    numpy==1.26.4 \
    thefuzz==0.19.0 \
    && (pip install --no-cache-dir python-Levenshtein==0.21.1 || echo "python-Levenshtein failed, continuing without it") \
    && pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Copy package files for Node.js
COPY package.json package-lock.json* ./

# Install Node.js dependencies
# Use npm install (more forgiving than npm ci if lock file has issues)
RUN npm install --prefer-offline --no-audit

# Copy application code
COPY . .

# Build frontend (this creates the dist/ directory)
RUN npm run build

# Quick verification (combined into one RUN)
RUN test -f server/api.py && test -d dist && test -f dist/index.html || exit 1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# CRITICAL: Start FastAPI server using uvicorn
# This MUST be the only command that runs
# DO NOT use start.sh or any other script
# Render will use this CMD, not start.sh
CMD ["/bin/sh", "-c", "echo '=== Starting TTB Verifier with FastAPI ===' && echo 'PORT: '${PORT:-8000} && echo 'Working dir: '$(pwd) && echo 'Python: '$(python --version) && echo 'Node: '$(node --version) && uvicorn server.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
