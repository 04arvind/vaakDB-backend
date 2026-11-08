FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
COPY app/ai/requirements.txt ./app/ai/requirements.txt
RUN pip install -r requirements.txt -r app/ai/requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 vaak \
    && mkdir -p /app/var/uploads \
    && chown -R vaak:vaak /app/var
USER vaak

EXPOSE 8000
# Run migrations first:  docker run --rm --env-file .env <image> alembic upgrade head
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
