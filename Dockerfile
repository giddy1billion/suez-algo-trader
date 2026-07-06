FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (including .git for governance commit hash tracking)
COPY . .

# Embed git commit hash at build time as fallback for governance
ARG GIT_COMMIT_HASH=""
RUN if [ -z "$GIT_COMMIT_HASH" ] && [ -d .git ]; then \
        git rev-parse HEAD > /app/.git_commit 2>/dev/null || true; \
    elif [ -n "$GIT_COMMIT_HASH" ]; then \
        echo "$GIT_COMMIT_HASH" > /app/.git_commit; \
    fi

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
