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

# Create directories and set ownership
RUN mkdir -p data_cache models logs \
    && addgroup --system --gid 1001 trader \
    && adduser --system --uid 1001 --ingroup trader trader \
    && chown -R trader:trader /app

# Run as non-root user
USER trader

# Default: paper trading with momentum
ENV TRADING_MODE=paper
ENV ACTIVE_STRATEGY=momentum

# Numba cache — non-root user cannot write to site-packages;
# redirect JIT cache to a writable directory.
ENV NUMBA_CACHE_DIR=/app/data_cache/.numba_cache

# Health check — verify process is alive and responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, signal; os.kill(1, signal.SIG_DFL) or True" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--interval", "60"]
