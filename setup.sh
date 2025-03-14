#!/bin/bash

# PaperWoCode Setup Script
# This script sets up the conda environment and runs the project

set -e  # Exit immediately if a command exits with a non-zero status

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install conda first."
    echo "Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions."
    exit 1
fi

# Check if environment already exists and remove it if it does
if conda env list | grep -q "paperwocode"; then
    echo "Conda environment 'paperwocode' already exists. Removing it to ensure a clean setup..."
    conda env remove -n paperwocode
fi

# Create the conda environment
echo "Creating conda environment from environment.yml..."
if ! conda env create -f environment.yml; then
    echo "Error: Failed to create conda environment from environment.yml."
    echo "Trying alternative setup method..."
    
    # Create a basic environment and install dependencies manually
    conda create -n paperwocode python=3.10 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate paperwocode
    
    # Install dependencies
    echo "Installing dependencies..."
    conda install -y numpy pandas matplotlib scikit-learn pytorch torchvision torchaudio -c pytorch
    pip install 'markitdown[all]~=0.1.0a1' anthropic openai PyPDF2 tqdm pyyaml tiktoken
else
    # Activate the environment
    echo "Activating conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate paperwocode
fi

# Install the package in development mode
echo "Installing PaperWoCode package in development mode..."
pip install -e .

# Verify the environment is activated
if [[ "$(conda info --envs | grep '*' | awk '{print $1}')" != "paperwocode" ]]; then
    echo "Error: Failed to activate conda environment 'paperwocode'."
    echo "Please try to activate it manually with: conda activate paperwocode"
    exit 1
fi

# Check if API keys are set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ANTHROPIC_API_KEY is not set."
    read -p "Enter your Anthropic API key: " anthropic_key
    export ANTHROPIC_API_KEY=$anthropic_key
    
    # Add to shell config file based on shell type
    if [ -n "$ZSH_VERSION" ]; then
        echo "export ANTHROPIC_API_KEY=$anthropic_key" >> ~/.zshrc
        echo "ANTHROPIC_API_KEY has been set and added to your ~/.zshrc file."
    elif [ -n "$BASH_VERSION" ]; then
        echo "export ANTHROPIC_API_KEY=$anthropic_key" >> ~/.bashrc
        echo "ANTHROPIC_API_KEY has been set and added to your ~/.bashrc file."
    else
        echo "ANTHROPIC_API_KEY has been set for this session only."
        echo "Please add 'export ANTHROPIC_API_KEY=$anthropic_key' to your shell configuration file."
    fi
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is not set."
    read -p "Enter your OpenAI API key: " openai_key
    export OPENAI_API_KEY=$openai_key
    
    # Add to shell config file based on shell type
    if [ -n "$ZSH_VERSION" ]; then
        echo "export OPENAI_API_KEY=$openai_key" >> ~/.zshrc
        echo "OPENAI_API_KEY has been set and added to your ~/.zshrc file."
    elif [ -n "$BASH_VERSION" ]; then
        echo "export OPENAI_API_KEY=$openai_key" >> ~/.bashrc
        echo "OPENAI_API_KEY has been set and added to your ~/.bashrc file."
    else
        echo "OPENAI_API_KEY has been set for this session only."
        echo "Please add 'export OPENAI_API_KEY=$openai_key' to your shell configuration file."
    fi
fi

# Create output directory
mkdir -p output/workflows

# Run the project
echo ""
echo "Setup complete! You can now run the project with:"
echo "conda activate paperwocode"
echo "python main.py [arxiv_url]"
echo ""
echo "For example:"
echo "python main.py https://arxiv.org/abs/2411.16905"
