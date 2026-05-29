FROM python:3.11

# Set working directory inside container
WORKDIR /app

# Install uv (fast package manager)

RUN pip install --no-cache-dir uv

# Copy dependency files first (for caching layers)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy full project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]