FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PRED_LOG=/app/logs/predictions.jsonl

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY serving/ serving/
COPY models/model.onnx models/classes.json models/

RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
