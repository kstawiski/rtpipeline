#!/bin/bash
# Build and push rtpipeline Docker image
# Usage: ./build.sh [options]
# Options:
#   --push              Push to Docker Hub after build
#   --no-cache          Build without cache
#   --tag <tag>         Specify tag (default: latest)
#   --registry <reg>    Docker registry (default: docker.io)
#   --username <user>   Docker Hub username (required for push)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PUSH=false
NO_CACHE=""
TAG="latest"
REGISTRY="docker.io"
USERNAME="kstawiski"
IMAGE_NAME="rtpipeline"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --username)
            USERNAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./build.sh [options]"
            echo "Options:"
            echo "  --push              Push to Docker Hub after build"
            echo "  --no-cache          Build without cache"
            echo "  --tag <tag>         Specify tag (default: latest)"
            echo "  --registry <reg>    Docker registry (default: docker.io)"
            echo "  --username <user>   Docker Hub username (default: kstawiski)"
            echo ""
            echo "Examples:"
            echo "  ./build.sh                                    # Build locally as kstawiski/rtpipeline:latest"
            echo "  ./build.sh --push                             # Build and push to kstawiski/rtpipeline"
            echo "  ./build.sh --push --tag v1.0.0                # Build and push with specific tag"
            echo "  ./build.sh --tag v1.0.0 --no-cache            # Build with specific tag, no cache"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Note: Username defaults to kstawiski but can be overridden

# Set full image name
if [ -n "$USERNAME" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${USERNAME}/${IMAGE_NAME}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}"
fi

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Building rtpipeline Docker image${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Image name: ${FULL_IMAGE_NAME}:${TAG}"
echo "Build date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Build the image
echo -e "${YELLOW}Building image...${NC}"
docker build \
    ${NO_CACHE} \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    -t "${FULL_IMAGE_NAME}:${TAG}" \
    -t "${FULL_IMAGE_NAME}:latest" \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build completed successfully${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Show image info
echo ""
echo -e "${YELLOW}Image information:${NC}"
docker images "${FULL_IMAGE_NAME}" | grep -E "(REPOSITORY|${IMAGE_NAME})"

# Push to registry if requested
if [ "$PUSH" = true ]; then
    echo ""
    echo -e "${YELLOW}Pushing to ${REGISTRY}...${NC}"

    # Login to Docker Hub
    echo -e "${YELLOW}Please login to Docker Hub:${NC}"
    docker login ${REGISTRY}

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Docker login failed${NC}"
        exit 1
    fi

    # Push both tags
    echo -e "${YELLOW}Pushing ${FULL_IMAGE_NAME}:${TAG}...${NC}"
    docker push "${FULL_IMAGE_NAME}:${TAG}"

    if [ "$TAG" != "latest" ]; then
        echo -e "${YELLOW}Pushing ${FULL_IMAGE_NAME}:latest...${NC}"
        docker push "${FULL_IMAGE_NAME}:latest"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Push completed successfully${NC}"
        echo ""
        echo "Image available at: ${FULL_IMAGE_NAME}:${TAG}"
    else
        echo -e "${RED}✗ Push failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Build process completed${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "To run the container:"
echo "  docker run -it --rm ${FULL_IMAGE_NAME}:${TAG}"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up -d"
echo ""
echo "For CPU-only (no GPU):"
echo "  docker-compose --profile cpu-only up -d"
echo ""
echo "To convert to Singularity:"
echo "  singularity build rtpipeline.sif docker-daemon://${FULL_IMAGE_NAME}:${TAG}"
echo ""
echo "To pull from Docker Hub:"
echo "  docker pull ${FULL_IMAGE_NAME}:${TAG}"
echo ""
