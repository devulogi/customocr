FROM python:3.9-slim

# Install system dependencies for OpenCV, OCR, and PDF processing
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgomp1 \
    # Image processing
    libopencv-dev \
    # PDF processing
    wget \
    curl \
    git \
    build-essential \
    # Compilation optimization
    ccache \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create virtual environment
RUN python -m venv venv

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in venv
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create directories for project structure
RUN mkdir -p pdfs outputs

# Set environment variables for venv and OCR performance
ENV PATH=/workspace/venv/bin:$PATH
ENV PYTHONPATH=/workspace
ENV OPENCV_LOG_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1

# Expose port for potential web interface
EXPOSE 8000

CMD ["bash"]
