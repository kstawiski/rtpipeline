#!/bin/bash
# RTpipeline Complete Setup Script
# This script provides multiple deployment options for rtpipeline

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "       RTpipeline Complete Setup Script"
echo "=================================================="
echo
echo "This script will help you set up rtpipeline with all features."
echo "Choose your preferred deployment method:"
echo
echo "1) Conda Environment (Recommended for development)"
echo "2) Docker Setup (Recommended for production)"
echo "3) Virtual Environment (pip-based)"
echo "4) Validate existing installation"
echo "5) Show deployment information"
echo

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo
        echo "Setting up Conda environment..."
        if ! command -v conda &> /dev/null; then
            echo "❌ Conda not found. Please install Miniconda or Anaconda first."
            echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
        
        chmod +x setup_environment.sh
        ./setup_environment.sh
        
        echo
        echo "✅ Conda setup complete!"
        echo "To use rtpipeline:"
        echo "   conda activate rtpipeline-full"
        echo "   rtpipeline doctor"
        ;;
        
    2)
        echo
        echo "Setting up Docker environment..."
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker not found. Please install Docker first."
            echo "   Visit: https://docs.docker.com/get-docker/"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
            echo "❌ Docker Compose not found. Please install Docker Compose."
            exit 1
        fi
        
        echo "Building Docker image (this may take several minutes)..."
        docker build -t rtpipeline .
        
        echo "Starting Docker containers..."
        docker-compose up -d rtpipeline
        
        echo
        echo "✅ Docker setup complete!"
        echo "To use rtpipeline:"
        echo "   docker-compose exec rtpipeline bash"
        echo "   rtpipeline doctor"
        echo
        echo "To start Jupyter Lab:"
        echo "   docker-compose --profile jupyter up -d jupyter"
        echo "   Access at: http://localhost:8889"
        ;;
        
    3)
        echo
        echo "Setting up virtual environment..."
        
        # Check Python version
        python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' || echo "unknown")
        echo "Detected Python version: $python_version"
        
        if [[ "$python_version" != "3.11" ]]; then
            echo "⚠️  Warning: Python 3.11 is recommended for full compatibility."
            echo "   Current version may have issues with some dependencies."
            read -p "Continue anyway? (y/N): " continue_choice
            if [[ "$continue_choice" != [Yy] ]]; then
                echo "Setup cancelled."
                exit 1
            fi
        fi
        
        # Create virtual environment
        ENV_NAME="rtpipeline-venv"
        python3 -m venv "$ENV_NAME"
        source "$ENV_NAME/bin/activate"
        
        echo "Installing dependencies..."
        pip install --upgrade pip wheel setuptools
        
        # Install with constraints
        pip install "numpy>=1.20,<2.0"
        pip install -e .
        
        echo "Installing optional dependencies..."
        pip install "TotalSegmentator==2.4.0" || echo "⚠️  TotalSegmentator installation failed"
        pip install "pyradiomics>=3.0" || echo "⚠️  pyradiomics installation failed"
        
        echo
        echo "✅ Virtual environment setup complete!"
        echo "To activate:"
        echo "   source $ENV_NAME/bin/activate"
        echo "   rtpipeline doctor"
        ;;
        
    4)
        echo
        echo "Validating existing installation..."
        if [ -f "validate_environment.py" ]; then
            python validate_environment.py
        else
            echo "❌ validate_environment.py not found"
            exit 1
        fi
        ;;
        
    5)
        echo
        echo "Opening deployment guide..."
        if [ -f "DEPLOYMENT.md" ]; then
            less DEPLOYMENT.md || cat DEPLOYMENT.md
        else
            echo "❌ DEPLOYMENT.md not found"
            exit 1
        fi
        ;;
        
    *)
        echo "Invalid choice. Please run the script again and choose 1-5."
        exit 1
        ;;
esac

echo
echo "=================================================="
echo "Setup complete! Next steps:"
echo
echo "1. Activate your environment (conda/docker/venv)"
echo "2. Run: rtpipeline doctor"
echo "3. Test with your data: rtpipeline process /path/to/data"
echo
echo "For detailed usage instructions, see:"
echo "  - DEPLOYMENT.md (deployment options)"
echo "  - TROUBLESHOOTING.md (common issues)"
echo "  - docs/ (full documentation)"
echo
echo "Need help? Check the troubleshooting guide or validate"
echo "your installation with: python validate_environment.py"
echo "=================================================="