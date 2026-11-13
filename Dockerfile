FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
RUN mkdir -p /app/artifacts
EXPOSE 8000
# Bind to localhost at the host/port level; do not expose publicly without authentication/TLS.
CMD ["uvicorn", "carsel.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
