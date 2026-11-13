# Builds a self-contained API image: data is downloaded and the model is
# trained during the build, so the container starts serving straight away.
#
#   docker build -t turbofan-rul .
#   docker run -p 8000:8000 turbofan-rul
#
# To use Ollama running on your own machine for the explanations:
#   docker run -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 turbofan-rul

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt -e .

COPY scripts ./scripts
RUN python scripts/download_data.py && python -m turbofan_rul.train

EXPOSE 8000
CMD ["uvicorn", "turbofan_rul.api:app", "--host", "0.0.0.0", "--port", "8000"]
