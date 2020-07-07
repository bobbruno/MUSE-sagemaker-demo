# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM nvidia/cuda:10.1-base-ubuntu18.04
#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.2.0-gpu-py37-cu102-ubuntu18.04

ENV NCCL_VERSION=2.4.7-1+cuda10.1
ENV CUDNN_VERSION=7.6.2.24-1+cuda10.1
ENV TF_TENSORRT_VERSION=6.0.1

RUN apt-get update && \
    apt-get -y install --no-install-recommends --allow-unauthenticated --allow-downgrades\
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    nginx \
    ca-certificates \
    cuda-command-line-tools-10-1 \
    cuda-cufft-10-1 \
    cuda-curand-10-1 \
    cuda-cusolver-10-1 \
    cuda-cusparse-10-1 \
    #cuda-cublas-dev not available with 10-1, install libcublas instead
    libcublas10=10.1.0.105-1 \
    libcublas-dev=10.1.0.105-1 \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libgomp1 \
    curl \
    wget \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN ln -s /usr/local/cuda-10.1 /usr/local/nvidia

# Here we get all python packages.

RUN pip3 --no-cache-dir install numpy \
                                pandas \
                                flask gevent gunicorn \
                                multi-model-server \
                                sagemaker-inference \
                                retrying

COPY requirements.txt /usr/local/bin/requirements.txt

RUN pip3 install -r /usr/local/bin/requirements.txt

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ADD . /opt/program/

RUN ls /opt/program/

# Set up the program in the image
#COPY transformscript.py /opt/program/transformscript.py

#COPY nginx.conf /opt/program/nginx.conf

#COPY predictor.py /opt/program/predictor.py

#COPY serve /opt/program/serve

#COPY train /opt/program/train

#COPY wsgi.py /opt/program/wsgi.py

RUN chmod +x /opt/program/train

RUN chmod +x /opt/program/serve

WORKDIR /opt/program

