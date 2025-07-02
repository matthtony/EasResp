# 1. Use a slim Python base
FROM python:3.10-slim

# 2. Install system deps for faiss build
RUN apt-get update && \
    apt-get install -y build-essential swig cmake git && \
    rm -rf /var/lib/apt/lists/*

# 3. Copy & install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: faiss-cpu && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy your application code
COPY . .

# 5. Expose Flask’s default port
EXPOSE 5000

# 6. Run your Flask app directly
#    Assumes in your code you have:
#      if __name__ == "__main__":
#        app.run(host="0.0.0.0", port=5000)
CMD ["python", "interface.py"]
