FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Create directories
RUN mkdir -p data_cache models logs

# Default: paper trading with momentum
ENV TRADING_MODE=paper
ENV ACTIVE_STRATEGY=momentum

ENTRYPOINT ["python", "main.py"]
CMD ["--interval", "60"]
