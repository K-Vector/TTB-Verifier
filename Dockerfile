FROM python:3.11-slim

# Install system deps - tesseract, node, etc.
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

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python deps - python-Levenshtein is optional, skip if it fails
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

# Install Node deps
COPY package.json package-lock.json* ./
RUN npm install --prefer-offline --no-audit

# Copy everything and build frontend
COPY . .
RUN npm run build

# Make sure everything built correctly
RUN test -f server/api.py && test -d dist && test -f dist/index.html || exit 1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Start the server - Render uses this CMD
CMD ["/bin/sh", "-c", "uvicorn server.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
