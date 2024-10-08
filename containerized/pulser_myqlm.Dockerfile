FROM python3.10:slim

# Install system dependencies needed for many Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copu requirements and install dependencies
COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app
RUN pip install -r requirements.txt --use-deprecated=legacy-resolver
RUN pip install pytest

# Copy application files
COPY ./pulser_myqlm /opt/app/pulser_myqlm
COPY ./containerized/test_fresnel.py /opt/app/tests/test_fresnel.py

# Run tests
CMD ["pytest", "/opt/app/tests/test_fresnel.py"]
