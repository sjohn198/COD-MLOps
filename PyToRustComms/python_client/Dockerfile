FROM python:3.13-slim AS worker

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "training_worker.py"]
