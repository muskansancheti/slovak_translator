FROM python:3.10-slim

# System dependencies for OpenCV + PyMuPDF
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["gunicorn", "server:app", "--workers", "1", "--timeout", "300", "--bind", "0.0.0.0:7860"]
