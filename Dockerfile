FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Project metadata and sources; pyproject declares readme = "README.md",
# so it must be present for the build.
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples

RUN pip install .

EXPOSE 8000

# serve creates the database schema before starting uvicorn.
CMD ["agentverdict", "serve", "--host", "0.0.0.0", "--port", "8000"]
