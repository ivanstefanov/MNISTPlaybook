FROM python:3.10-slim

RUN pip install --no-cache-dir torch kserve

WORKDIR /app
COPY server.py /app/server.py

CMD ["python", "server.py"]
