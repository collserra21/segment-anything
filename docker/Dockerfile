FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SAM_HOME=/models
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu && pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
COPY app.py .
COPY tests ./tests
ENTRYPOINT ["python", "app.py"]
