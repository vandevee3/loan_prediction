FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
RUN pip install --no-cache-dir -r requirements.txt && \ 
    pip install --no-cache-dir "fastapi[standard]" uvicorn

COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY backend/ ./backend/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]