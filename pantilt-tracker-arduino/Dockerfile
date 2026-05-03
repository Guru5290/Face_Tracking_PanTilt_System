FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    cmake \
    v4l-utils \
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-report.txt /app/
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
# l4t-ml base includes NumPy; install tracker deps then matplotlib for benchmarks/charts.
RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install --no-cache-dir -r requirements-report.txt
COPY src/ ./src/
CMD ["python3", "src/main.py"]
