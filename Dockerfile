# Docker configuration for API deployment


# Start from a base Python image
FROM python:3.11-slim

# Set working directory inside container

WORKDIR /app

# Copy requirements first (for better caching)
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "models/train_model.py"]
CMD ["python", "src/app.py"]
