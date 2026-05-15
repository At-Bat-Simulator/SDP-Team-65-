FROM python:3.12-slim

# Install Node.js and git-lfs
RUN apt-get update && apt-get install -y curl git git-lfs && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    git lfs install && \
    apt-get clean

WORKDIR /app

# Build frontend
COPY frontend/package*.json frontend/
RUN cd frontend && npm install

COPY frontend/ frontend/
RUN cd frontend && npm run build

# Install Python deps
COPY api/requirements.txt api/
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy rest of project
COPY . .

# Download serving table (too large for git)
RUN mkdir -p artifacts/serving && \
    curl -L "https://github.com/At-Bat-Simulator/SDP-Team-65-/releases/download/v1.0/serving_table.parquet" \
    -o artifacts/serving/serving_table.parquet

EXPOSE 8080

WORKDIR /app/api
CMD ["gunicorn", "app:app", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:8080"]
