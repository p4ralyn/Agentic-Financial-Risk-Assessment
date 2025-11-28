FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
