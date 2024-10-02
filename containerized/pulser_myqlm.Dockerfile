FROM 9ygszqk0.gra7.container-registry.ovh.net/base/python:3.8-main

# Install system dependencies using apk
RUN apk update && \
    apk add --no-cache \
    libffi-dev \
    build-base

# Copu requirements and install dependencies
COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN pip install pytest

# Copy application files
COPY ./pulser_myqlm /opt/app/pulser_myqlm
COPY ./containerized/test_fresnel.py /opt/app/tests/test_fresnel.py

# Run tests
CMD ["pytest", "/opt/app/tests/test_fresnel.py"]
