FROM python:3.12-slim

# ENV PYTHONDONTWRITEBYTECODE=1
RUN useradd -ms /bin/bash user

WORKDIR /opt/

COPY requirements.txt /opt/requirements.txt
RUN pip install --upgrade --no-cache-dir --requirement /opt/requirements.txt

USER user
WORKDIR /home/user
