FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.39.0 requests==2.32.3

COPY frontend/ui.py ./ui.py

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]