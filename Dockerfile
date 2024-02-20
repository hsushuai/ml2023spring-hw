ARG RELEASE
ARG LAUNCHPAD_BUILD_ARCH

LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=22.04

ADD file:aa9b51e9f0067860cebbc9930374452d1384ec3c59badb5e4733130eedc90329 in /
CMD ["/bin/bash"]
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN /bin/sh -c apt update && apt install -y wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && apt clean && rm -rf /var/lib/apt/lists/* # buildkit

ARG PYTHON_VERSION

RUN |1 PYTHON_VERSION=3.10.11 /bin/sh -c cd /tmp && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && tar -xvf Python-${PYTHON_VERSION}.tgz && cd Python-${PYTHON_VERSION} && ./configure --enable-optimizations && make && make install && cd .. && rm Python-${PYTHON_VERSION}.tgz && rm -r Python-${PYTHON_VERSION} && ln -s /usr/local/bin/python3 /usr/local/bin/python && ln -s /usr/local/bin/pip3 /usr/local/bin/pip && python -m pip install --upgrade pip && rm -r /root/.cache/pip # buildkit

ARG PYTORCH_VERSION
ARG PYTORCH_VERSION_SUFFIX
ARG TORCHVISION_VERSION
ARG TORCHVISION_VERSION_SUFFIX
ARG TORCHAUDIO_VERSION
ARG TORCHAUDIO_VERSION_SUFFIX
ARG PYTORCH_DOWNLOAD_URL

RUN |8 PYTHON_VERSION=3.10.11 PYTORCH_VERSION=2.0.1 PYTORCH_VERSION_SUFFIX=+cu118 TORCHVISION_VERSION=0.15.2 TORCHVISION_VERSION_SUFFIX=+cu118 TORCHAUDIO_VERSION=2.0.2 TORCHAUDIO_VERSION_SUFFIX=+cu118 PYTORCH_DOWNLOAD_URL=https://download.pytorch.org/whl/cu118/torch_stable.html /bin/sh -c if [ ! $TORCHAUDIO_VERSION ]; then TORCHAUDIO=; else TORCHAUDIO=torchaudio==${TORCHAUDIO_VERSION}${TORCHAUDIO_VERSION_SUFFIX}; fi && if [ ! $PYTORCH_DOWNLOAD_URL ]; then pip install torch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}             torchvision==${TORCHVISION_VERSION}${TORCHVISION_VERSION_SUFFIX}             ${TORCHAUDIO};     else         pip install             torch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}             torchvision==${TORCHVISION_VERSION}${TORCHVISION_VERSION_SUFFIX}             ${TORCHAUDIO}             -f ${PYTORCH_DOWNLOAD_URL};     fi &&     rm -r /root/.cache/pip # buildkit

WORKDIR /workspace