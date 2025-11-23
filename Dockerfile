FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY Simulations/main.py /app/
COPY Simulations/app.py /app/
COPY Simulations/requirements_ui.txt /app/
COPY Simulations/*.csv /app/ 2>/dev/null || true

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_ui.txt

# Expose Streamlit port
EXPOSE 8501

# Create directory for results
RUN mkdir -p /app/results

# Configure Streamlit
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\nport = 8501\nenableXsrfProtection = false\n" > ~/.streamlit/config.toml

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
