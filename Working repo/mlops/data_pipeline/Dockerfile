# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline code to the working directory
COPY pipeline.py .

# Set the entrypoint command
CMD ["dagster", "pipeline", "execute", "--pipeline", "data_pipeline"]

# Expose any necessary ports
# EXPOSE <port>

# Define any environment variables, if needed
# ENV KEY=value
