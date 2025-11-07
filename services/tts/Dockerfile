FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi uvicorn[standard]
# Expect a piper binary available in the container, or mount from host and set PIPER_BIN
ENV PIPER_BIN=piper
WORKDIR /app
COPY tts_server.py /app/
CMD ["uvicorn", "tts_server:app", "--host", "0.0.0.0", "--port", "8003"]
