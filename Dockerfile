FROM --platform=linux/amd64 python:3.13

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .

# First install CPU‐only torch from PyTorch’s CPU index
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1+cpu \
  && pip install --no-cache-dir -r requirements.txt

# Copy in your processing script
COPY process_pdfs.py .

COPY models /app/models

# When the container launches, run this:
CMD ["python", "process_pdfs.py"]
