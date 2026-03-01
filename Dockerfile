FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY static/ static/
COPY data/ data/

ENV GAME_ID=26feb09arizku
ENV VLM_ENDPOINT=https://adityapdev13--qwen3-vl-inference-inference-serve.modal.run
ENV SCORE_ENDPOINT=https://adityapdev13--qwen3-vl-inference-scoreinference-serve.modal.run/v1/score
ENV MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
ENV PORT=8080

EXPOSE 8080

CMD ["python", "main.py"]
