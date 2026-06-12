# Dockerfile for reproducible circuit knitting environment
# Matches lenore (Ubuntu 24.04, Python 3.12, same package versions)
# Build: docker build -t circuit-knitting .
# Run: docker run -it --rm circuit-knitting

FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    gcc \
    g++ \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify GLIBC version matches lenore (2.39)
RUN ldd --version | grep -i glibc

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install required packages at exact versions matching lenore
RUN pip install --no-cache-dir --upgrade pip && \
    pip install ibm-platform-services==0.75.1 \
              cryptography==49.0.0 \
              qiskit==1.4.2 \
              qiskit-aer==0.15.0 \
              qiskit-ibm-runtime==0.32.0 \
              qiskit-ibm-provider==0.11.0 \
              numpy==2.1.3 \
              websockets==16.0 \
              scipy==1.17.1

# Verify installation
RUN python3 --version && \
    python3 -c "import qiskit; print('qiskit:', qiskit.__version__)" && \
    python3 -c "import numpy; print('numpy:', numpy.__version__)" && \
    python3 -c "import qiskit_aer; print('qiskit-aer:', qiskit_aer.__version__)" && \
    python3 -c "import ibm_platform_services; print('ibm-platform-services:', ibm_platform_services.__version__)" && \
    ldd --version | grep -i glibc

# Set working directory
WORKDIR /app

# Default command: run bash
CMD ["bash"]
