FROM python:3.12-slim

# ENV PYTHONDONTWRITEBYTECODE=1
RUN useradd -ms /bin/bash user

# install openMPI and MPI's mpicxx binary
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl libopenmpi-dev mpi-default-dev

# set workdir for /opt/cmdstan-CSVER
WORKDIR /opt/

COPY requirements.txt /opt/requirements.txt
RUN pip install --upgrade --no-cache-dir --requirement /opt/requirements.txt

# go back to the main user directory
USER user
WORKDIR /home/user
