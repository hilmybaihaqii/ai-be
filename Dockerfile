# Gunakan base image python slim + cpu-friendly
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . .

# Install dependency system (biar tensorflow nggak error)
RUN apt-get update && \
    apt-get install -y build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Jalankan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
