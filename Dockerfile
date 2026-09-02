FROM python:3.11-slim

# mediapipe needs EGL/GL libs at runtime even though we never open a
# display window; opencv-python-headless doesn't need GTK/X11 libs, so
# we keep this list minimal to avoid apt package-name churn across
# Debian base image versions (e.g. the libglib2.0-0 -> libglib2.0-0t64
# rename in newer releases)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libgles2 ffmpeg curl \
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
