# Fetch base image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Setup work directory & configuration files
WORKDIR /mobility/
COPY src/requirements.txt /mobility/requirements.txt
COPY lib/ lib/
COPY pre/ pre/

# Get & install necessary tools on image
RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx ffmpeg libsm6 libxext6
RUN apt-get install -y build-essential cmake pkg-config
RUN apt-get install -y libx11-dev libatlas-base-dev
RUN apt-get install -y libgtk-3-dev libboost-python-dev
RUN apt-get install -y python-dev python3-dev python3-pip
RUN pip install --upgrade pip

# After successful build
CMD pip install -r requirements.txt
