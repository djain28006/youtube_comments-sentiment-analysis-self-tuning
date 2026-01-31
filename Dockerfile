# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the code
COPY . /code

# Create a writable cache directory for Transformers
# (Hugging Face requires this specific permission setup)
RUN mkdir -p /code/cache && chmod -R 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]