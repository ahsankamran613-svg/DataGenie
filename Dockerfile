# 1. Start from official lightweight Python 3.12 Linux image
FROM python:3.12-slim

# 2. Set working folder inside the container
WORKDIR /app

# 3. Copy and install Python dependencies first (cached for fast rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project source code
COPY . .

# 5. Expose Streamlit's default port and run the app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
