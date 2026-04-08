FROM python:3.11-slim

WORKDIR /app

# Copy all files to /app
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the main.py server
CMD ["python", "main.py"]
