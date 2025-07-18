FROM nvidia/cuda:12.9.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git libgl1-mesa-glx libglib2.0-0\
    && python3.10 -m pip install --no-cache-dir --upgrade pip \
    && python3.10 -m pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app/src

# 
COPY dataset /app/dataset
COPY requirements.txt . 
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

COPY src/ .

# CMD ["python3.10", "trainer.py"]
CMD ["python3.10", "pt_to_engine.py"]

