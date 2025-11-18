#!/bin/bash
# Docker Build and Deployment Script for rtpipeline
# Usage: ./deploy_docker.sh [options]

set -e

# Default values
IMAGE_NAME="rtpipeline"
TAG="latest"
DOCKER_REPO=""
PUSH=false

# Help message
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --tag TAG        Image tag (default: latest)"
    echo "  -r, --repo REPO      Docker repository (e.g., username/rtpipeline)"
    echo "  -p, --push           Push to repository after build"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -t|--tag)
            TAG="$2"
            shift
            shift
            ;; 
        -r|--repo)
            DOCKER_REPO="$2"
            shift
            shift
            ;; 
        -p|--push)
            PUSH=true
            shift
            ;; 
        -h|--help)
            usage
            ;; 
        *)
            echo "Unknown option: $1"
            usage
            ;; 
    esac
done

# Determine full image name
if [ -n "$DOCKER_REPO" ]; then
    FULL_IMAGE_NAME="${DOCKER_REPO}:${TAG}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
fi

echo "=================================================="
echo "🏗️  Building Docker Image: ${FULL_IMAGE_NAME}"
echo "=================================================="

# Check docker availability
if ! command -v docker &> /dev/null; then
    echo "❌ Error: docker command not found."
    exit 1
fi

# Build arguments
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

# Run build
# Uses BuildKit for better caching and performance
DOCKER_BUILDKIT=1 docker build \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    -t "${FULL_IMAGE_NAME}" \
    -f Dockerfile .

echo "✅ Build successful!"

# Push if requested
if [ "$PUSH" = true ]; then
    if [ -z "$DOCKER_REPO" ]; then
        echo "❌ Error: Repository must be specified for push (use -r or --repo)."
        exit 1
    fi

    echo "=================================================="
    echo "🚀 Pushing to Docker Hub: ${FULL_IMAGE_NAME}"
    echo "=================================================="
    
    docker push "${FULL_IMAGE_NAME}"
    
    echo "✅ Push successful!"
fi

echo "=================================================="
echo "🎉 Done!"
echo "Run locally with:"
echo "  docker-compose up"
echo "=================================================="
