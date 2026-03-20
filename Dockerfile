FROM python:3.11.8-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY data ./data
COPY md ./md
COPY .env.example ./.env.example

RUN uv sync --frozen


FROM python:3.11.8-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY --from=builder /app /app

EXPOSE 8000

CMD ["uv", "run", "rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
