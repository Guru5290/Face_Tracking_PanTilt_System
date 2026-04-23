FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    python3-pip \
    v4l-utils \
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src/ ./src/
CMD ["python3", "src/main.py"]
