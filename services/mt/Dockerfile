FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi uvicorn[standard] transformers==4.44.2 sentencepiece==0.2.0 torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY mt_server.py /app/
CMD ["uvicorn", "mt_server:app", "--host", "0.0.0.0", "--port", "8002"]
