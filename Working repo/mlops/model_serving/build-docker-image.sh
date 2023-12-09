#!/bin/bash

# Build the Docker image for the model serving component
docker build -t model-serving:latest .

# Tag the Docker image with a specific version
docker tag model-serving:latest model-serving:1.0

# Push the Docker image to a container registry
docker push model-serving:latest
docker push model-serving:1.0
