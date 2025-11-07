FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn[standard] faster-whisper==1.0.3
ENV ASR_DEVICE=cpu ASR_BEAM_SIZE=1
WORKDIR /app
COPY asr_server.py /app/
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "8001"]
