FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system --no-cache openenv-core fastapi uvicorn pydantic openai huggingface_hub requests

# Copy project files
COPY . .

# Expose port
EXPOSE 7860

# Run the server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
