FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    git \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    rm /tmp/get-pip.py


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

ADD . /app
WORKDIR /app

RUN ls -la /app && cat /app/requirements.txt


RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install -r requirements.txt --index-url https://pypi.org/simple
    
# This version is useful in case GPU are older with cuda-11.8
RUN pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    
EXPOSE 5000
# Auto serves this file once container is created
ENTRYPOINT ["python3","-u","/app/main.py"]
WORKDIR /app
