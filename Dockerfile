FROM python:3.12-slim

# Install Node.js
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
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

EXPOSE 8080

WORKDIR /app/api
CMD ["gunicorn", "app:app", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:8080"]
