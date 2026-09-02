FROM python:3.11-slim

# mediapipe/opencv need these system libs at runtime, including EGL/GL
# for mediapipe's graph runtime even when we never open a display window
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libegl1 libgles2 libsm6 libxext6 libxrender1 \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/templates

EXPOSE 8501
ENV SIGN_TEMPLATE_DIR=/app/templates

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
