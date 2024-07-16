FROM nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt update
RUN apt upgrade -y

# Install apt packages
COPY apt.txt apt.txt
ARG DEBIAN_FRONTEND=noninteractive
RUN xargs -a apt.txt apt-get install -y --no-install-recommends && rm -rf /var/cache/*

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# For some reason needs to be installed separately
RUN pip3 install flash-attn

# copy above docker folder to /consistency and set working directory
COPY . /consistency
WORKDIR /consistency

CMD [ "sleep", "infinity" ]