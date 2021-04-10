# Fetch base image
FROM tensorflow/tensorflow:latest-gpu

# Setup work directory & configuration files
WORKDIR /mobility/
COPY requirements.txt /mobility/requirements.txt

# Get & install necessary tools on image
RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get update
RUN apt-get install -y python3.8-dev python3-pip
RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python
RUN python -m pip --no-cache-dir install --upgrade pip setuptools

# When dependencies are finalized
RUN pip install -r requirements.txt

# After successful build
# Warning: Will be overridden with ANY parameters
CMD pip install -r requirements.txt
