FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
# Set Timezone and prevent interactive prompts during installation
ARG TZ
ENV TZ="$TZ"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  less git procps sudo fzf zsh man-db unzip gnupg2 curl wget \
  python3-pip python3-dev build-essential ca-certificates jq \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt --break-system-packages

RUN mkdir -p /workspace && chown -R 1000:1000 /workspace

WORKDIR /workspace
USER 1000:1000
