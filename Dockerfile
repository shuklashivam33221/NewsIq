FROM python:3.10-slim

WORKDIR /app

# Install git and git-lfs for downloading models if needed
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
