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
SKIP_WEIGHTS=false
SKIP_CUSTOM_WEIGHTS=false

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
        --skip-weights)
            SKIP_WEIGHTS=true
            shift
            ;;
        --skip-custom-weights)
            SKIP_CUSTOM_WEIGHTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./build.sh [options]"
            echo "Options:"
            echo "  --push              Push to Docker Hub after build"
            echo "  --no-cache          Build without cache"
            echo "  --tag <tag>         Specify tag (default: latest)"
            echo "  --registry <reg>    Docker registry (default: docker.io)"
            echo "  --username <user>   Docker Hub username (default: kstawiski)"
            echo "  --skip-weights      Skip TotalSegmentator weights download"
            echo "  --skip-custom-weights  Skip custom model weights download"
            echo ""
            echo "Model Weights:"
            echo "  The build will automatically download model weights if they don't exist:"
            echo "  - TotalSegmentator weights (requires TotalSegmentator in rtpipeline env)"
            echo "  - Custom model weights (cardiac_STOPSTORM from Zenodo)"
            echo "  Use --skip-weights and/or --skip-custom-weights to skip downloads."
            echo "  Skipped weights will be downloaded on first container run."
            echo ""
            echo "Examples:"
            echo "  ./build.sh                                    # Build with all weights"
            echo "  ./build.sh --push                             # Build and push to kstawiski/rtpipeline"
            echo "  ./build.sh --push --tag v1.0.0                # Build and push with specific tag"
            echo "  ./build.sh --tag v1.0.0 --no-cache            # Build with specific tag, no cache"
            echo "  ./build.sh --skip-weights --skip-custom-weights  # Build without any weights"
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

# Function to download TotalSegmentator weights
download_totalseg_weights() {
    local WEIGHTS_DIR="totalseg_weights"

    # Check if weights directory exists and has content
    if [ -d "$WEIGHTS_DIR" ] && [ -n "$(ls -A $WEIGHTS_DIR 2>/dev/null)" ]; then
        echo -e "${GREEN}✓ TotalSegmentator weights already present in ${WEIGHTS_DIR}/${NC}"
        return 0
    fi

    # Check if weights exist in the default location first
    local DEFAULT_WEIGHTS="${HOME}/.totalsegmentator"
    if [ -d "$DEFAULT_WEIGHTS/nnunet/results" ] && [ -n "$(ls -A $DEFAULT_WEIGHTS/nnunet/results 2>/dev/null)" ]; then
        echo -e "${YELLOW}Found existing TotalSegmentator weights in ${DEFAULT_WEIGHTS}${NC}"
        echo -e "${YELLOW}Copying to ${WEIGHTS_DIR}/...${NC}"
        mkdir -p "$WEIGHTS_DIR"
        cp -r "$DEFAULT_WEIGHTS"/* "$WEIGHTS_DIR/"
        echo -e "${GREEN}✓ TotalSegmentator weights copied successfully${NC}"
        return 0
    fi

    echo -e "${YELLOW}TotalSegmentator weights not found. Attempting to download...${NC}"
    echo "This may take a while (several GB of model files)..."

    # Create temporary directory for download
    local TEMP_WEIGHTS_DIR="${HOME}/.totalsegmentator_temp_$$"
    mkdir -p "$TEMP_WEIGHTS_DIR"

    # Download weights using TotalSegmentator Python API
    # We use the rtpipeline conda environment if available, otherwise system Python
    local PYTHON_CMD=""
    if command -v conda &> /dev/null && conda env list | grep -q "^rtpipeline "; then
        PYTHON_CMD="conda run -n rtpipeline python"
    elif [ -f "/opt/conda/envs/rtpipeline/bin/python" ]; then
        PYTHON_CMD="/opt/conda/envs/rtpipeline/bin/python"
    else
        PYTHON_CMD="python"
    fi

    echo -e "${YELLOW}Using Python: ${PYTHON_CMD}${NC}"

    # Download the weights using TotalSegmentator's download functionality
    $PYTHON_CMD << 'PYTHON_SCRIPT'
import os
import sys

# Set custom weights path to avoid conflicts
temp_dir = os.environ.get('TOTALSEG_TEMP_DIR', os.path.expanduser('~/.totalsegmentator_temp_' + str(os.getpid())))
os.environ['TOTALSEG_WEIGHTS_PATH'] = temp_dir
os.makedirs(temp_dir, exist_ok=True)

try:
    from totalsegmentator.download_pretrained_weights import download_pretrained_weights

    # Download the main task weights (total, total_mr)
    tasks = ['total', 'total_mr', 'tissue_types', 'body']

    for task in tasks:
        print(f"Downloading weights for task: {task}")
        try:
            download_pretrained_weights(task)
            print(f"  ✓ Downloaded {task}")
        except Exception as e:
            print(f"  ! Warning: Could not download {task}: {e}")

    print(f"\nWeights downloaded to: {temp_dir}")
    sys.exit(0)

except ImportError as e:
    print(f"Error: TotalSegmentator not installed or not accessible: {e}")
    print("Please install TotalSegmentator or activate the rtpipeline conda environment")
    sys.exit(1)
except Exception as e:
    print(f"Error downloading weights: {e}")
    sys.exit(1)
PYTHON_SCRIPT

    local DOWNLOAD_STATUS=$?

    if [ $DOWNLOAD_STATUS -ne 0 ]; then
        echo -e "${RED}✗ Failed to download TotalSegmentator weights${NC}"
        echo "You can manually download weights by running:"
        echo "  conda activate rtpipeline"
        echo "  python -c 'from totalsegmentator.download_pretrained_weights import download_pretrained_weights; download_pretrained_weights(\"total\")'"
        echo "Then copy ~/.totalsegmentator to ./totalseg_weights/"
        rm -rf "$TEMP_WEIGHTS_DIR"
        return 1
    fi

    # Move downloaded weights to the project directory
    echo -e "${YELLOW}Moving weights to ${WEIGHTS_DIR}/...${NC}"
    mkdir -p "$WEIGHTS_DIR"

    # Copy from the temp download location
    if [ -d "$TEMP_WEIGHTS_DIR" ] && [ -n "$(ls -A $TEMP_WEIGHTS_DIR 2>/dev/null)" ]; then
        cp -r "$TEMP_WEIGHTS_DIR"/* "$WEIGHTS_DIR/"
        rm -rf "$TEMP_WEIGHTS_DIR"
        echo -e "${GREEN}✓ TotalSegmentator weights downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Could not find downloaded weights${NC}"
        return 1
    fi

    return 0
}

# Function to download custom model weights
download_custom_model_weights() {
    local CUSTOM_MODELS_DIR="custom_models"

    echo -e "${YELLOW}Checking custom model weights...${NC}"

    # Check if wget or curl is available
    local DOWNLOAD_CMD=""
    if command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget"
    elif command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl"
    else
        echo -e "${RED}✗ Neither wget nor curl is available${NC}"
        echo "Please install wget or curl to download custom model weights"
        return 1
    fi

    # Function to download a single file
    download_file() {
        local url="$1"
        local output="$2"

        if [ -f "$output" ]; then
            echo -e "  ${GREEN}✓${NC} $(basename $output) already exists"
            return 0
        fi

        echo -e "  ${YELLOW}Downloading $(basename $output)...${NC}"
        local output_dir=$(dirname "$output")
        mkdir -p "$output_dir"

        if [ "$DOWNLOAD_CMD" = "wget" ]; then
            if wget -q --show-progress -O "$output" "$url"; then
                echo -e "  ${GREEN}✓${NC} Downloaded $(basename $output)"
                return 0
            fi
        else
            if curl -L -# -o "$output" "$url"; then
                echo -e "  ${GREEN}✓${NC} Downloaded $(basename $output)"
                return 0
            fi
        fi

        echo -e "  ${RED}✗${NC} Failed to download $(basename $output)"
        rm -f "$output"
        return 1
    }

    local DOWNLOAD_FAILED=0

    # ==========================================
    # cardiac_STOPSTORM models (Zenodo)
    # ==========================================
    echo ""
    echo -e "${YELLOW}1. cardiac_STOPSTORM model weights${NC}"
    echo "   Source: https://zenodo.org/records/14033123"

    local CARDIAC_DIR="$CUSTOM_MODELS_DIR/cardiac_STOPSTORM"
    local ZENODO_BASE="https://zenodo.org/records/14033123/files"

    # Model 603 - Small structures (valves, coronary arteries)
    download_file "${ZENODO_BASE}/model603.zip?download=1" "$CARDIAC_DIR/model603.zip" || DOWNLOAD_FAILED=1

    # Model 604 - Medium structures (if needed)
    download_file "${ZENODO_BASE}/model604.zip?download=1" "$CARDIAC_DIR/model604.zip" || DOWNLOAD_FAILED=1

    # Model 605 - Large structures (chambers, great vessels)
    download_file "${ZENODO_BASE}/model605.zip?download=1" "$CARDIAC_DIR/model605.zip" || DOWNLOAD_FAILED=1

    # ==========================================
    # HN_lymph_nodes models (placeholder)
    # ==========================================
    # Note: HN_lymph_nodes weights need to be obtained separately
    # Uncomment and update URLs when weights become available
    # echo ""
    # echo -e "${YELLOW}2. HN_lymph_nodes model weights${NC}"
    # local HN_DIR="$CUSTOM_MODELS_DIR/HN_lymph_nodes"
    # download_file "https://example.com/hn_weights.tar.gz" "$HN_DIR/weights.tar.gz" || DOWNLOAD_FAILED=1

    echo ""
    if [ $DOWNLOAD_FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ Custom model weights downloaded successfully${NC}"
        return 0
    else
        echo -e "${YELLOW}! Some custom model weights failed to download${NC}"
        echo "  Models with missing weights will be skipped during pipeline execution"
        return 1
    fi
}

# Download TotalSegmentator weights if needed
if [ "$SKIP_WEIGHTS" = true ]; then
    echo -e "${YELLOW}Skipping TotalSegmentator weights download (--skip-weights)${NC}"
    # Create empty directory to prevent Docker COPY failure
    mkdir -p totalseg_weights
else
    export TOTALSEG_TEMP_DIR="${HOME}/.totalsegmentator_temp_$$"
    download_totalseg_weights
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Continuing build without TotalSegmentator weights${NC}"
        echo -e "${YELLOW}The image will download weights on first run${NC}"
        # Create empty directory to prevent Docker COPY failure
        mkdir -p totalseg_weights
    fi
fi

# Download custom model weights if needed
if [ "$SKIP_CUSTOM_WEIGHTS" = true ]; then
    echo -e "${YELLOW}Skipping custom model weights download (--skip-custom-weights)${NC}"
else
    download_custom_model_weights
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Some custom model weights are missing${NC}"
        echo -e "${YELLOW}Models with missing weights will be skipped${NC}"
    fi
fi

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
    PUSH_TAG_STATUS=$?

    if [ "$TAG" != "latest" ]; then
        echo -e "${YELLOW}Pushing ${FULL_IMAGE_NAME}:latest...${NC}"
        docker push "${FULL_IMAGE_NAME}:latest"
        PUSH_LATEST_STATUS=$?
    else
        PUSH_LATEST_STATUS=0
    fi

    if [ $PUSH_TAG_STATUS -eq 0 ] && [ $PUSH_LATEST_STATUS -eq 0 ]; then
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
echo "  docker run -it --rm --gpus all \\"
echo "    -v \$(pwd)/Input:/data/input:ro \\"
echo "    -v \$(pwd)/Output:/data/output:rw \\"
echo "    ${FULL_IMAGE_NAME}:${TAG}"
echo ""
echo "Inside container, use container config:"
echo "  snakemake --cores all --use-conda --configfile /app/config.container.yaml"
echo ""
echo "Or use docker-compose (GPU default):"
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
