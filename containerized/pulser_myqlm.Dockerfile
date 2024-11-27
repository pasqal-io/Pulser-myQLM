FROM python:3.10-slim

# Install system dependencies needed for many Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copu requirements and install dependencies
COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app
# Ensure /opt/app is in PYTHONPATH
ENV PYTHONPATH=/opt/app:$PYTHONPATH

RUN pip install -r requirements.txt
RUN pip install pytest

# Copy application files
COPY ./pulser_myqlm /opt/app/pulser_myqlm
COPY ./VERSION.txt /opt/app/VERSION.txt
COPY ./containerized/test_fresnel.py /opt/app/tests/test_fresnel.py
COPY ./tests/conftest.py /opt/app/tests/conftest.py
COPY ./tests/helpers /opt/app/tests/helpers
