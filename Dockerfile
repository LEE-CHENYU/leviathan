FROM python:3.10-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY api/ api/
COPY kernel/ kernel/
COPY MetaIsland/ MetaIsland/
COPY Leviathan/ Leviathan/
COPY utils/ utils/
COPY config/ config/
COPY scripts/ scripts/

# Install package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "scripts/run_server.py"]
