#!/bin/bash
# Script to set up and run the shmem example using uv

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is required but not found. Please install Python 3.10."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    python3.10 -m pip install uv
fi

# Check if virtual environment already exists
if [ -d ".venv" ]; then
    echo "Virtual environment .venv already exists. Activating..."
else
    # Create a virtual environment with Python 3.10
    echo "Creating virtual environment with Python 3.10..."
    uv venv --python=python3.10
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Build the C++ library and Python bindings
echo "Building C++ library and Python bindings..."
cd ..
mkdir -p build
cd build
cmake .. -DPython_EXECUTABLE=$(which python3.10)
cmake --build .

# Copy the compiled module to the Python package
echo "Copying compiled module to Python package..."
cp cyshmem*.so ../pyshmem/shmem/

# Install the package
echo "Installing shmem package..."
cd ../pyshmem
uv pip install -e .

echo "Done!" 
