FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy env requirements
COPY env/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY . .

EXPOSE 7860

CMD ["uvicorn", "env.app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "300", "--workers", "1"]
