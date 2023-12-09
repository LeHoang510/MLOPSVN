#!/bin/bash

# Deploy the Docker image for the model serving component
docker run -d -p 8080:8080 --name model-serving model-serving:latest
