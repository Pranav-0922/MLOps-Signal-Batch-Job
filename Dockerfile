FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

CMD ["python", "src/run.py", \
     "--input", "data/data.csv", \
     "--config", "config/config.yaml", \
     "--output", "metrics.json", \
     "--log-file", "run.log"]
