FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md setup.py /app/
COPY src /app/src
COPY configs /app/configs

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "simcore.api:app", "--host", "0.0.0.0", "--port", "8000"]
