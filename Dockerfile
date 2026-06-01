FROM python:3.11-slim

WORKDIR /app

# Create a secure non-root user
RUN useradd -u 10001 appuser && chown -R appuser:appuser /app

COPY requirements.txt .

# Install CPU-only PyTorch to keep the image size at 299MB instead of 4GB+
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Switch to the non-root user for security
USER 10001

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
