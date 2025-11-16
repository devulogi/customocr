FROM python:3.9-slim

# Install system dependencies for OpenCV, OCR, and PDF processing
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    # OCR and image processing
    libopencv-dev \
    python3-opencv \
    # PDF processing
    libmupdf-dev \
    # General utilities
    wget \
    curl \
    git \
    build-essential \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for project structure
RUN mkdir -p pdfs outputs diagrams

# Set environment variables for better OCR performance
ENV PYTHONPATH=/workspace
ENV OPENCV_LOG_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1

# Expose port for potential web interface
EXPOSE 8000

CMD ["bash"]