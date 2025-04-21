# Signature Kernel for Conditional Independence Testing

This repository implements a signature kernel-based approach for conditional independence testing in time series data.

## Installation Guide

### Requirements
- Python (3.9 or 3.10 recommended)
- numpy
- matplotlib
- networkx
- scipy
- iisignature

### Basic Installation

1. Clone this repository:
```bash
git clone 
cd signature_kernel_ci
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the basic requirements:
```bash
pip install -r requirements.txt
```

### Installing iisignature

The iisignature package can be challenging to install, especially with newer Python versions (3.11+). Here are several approaches:

#### Method 1: Direct Installation (works with Python 3.9-3.10)
```bash
pip install iisignature
```

#### Method 2: Manual Build (if Method 1 fails)

If you encounter errors like `ModuleNotFoundError: No module named 'numpy'` during the build process, follow these steps:

1. Ensure numpy is installed first:
```bash
pip install numpy
```

2. Download and build iisignature from source:
```bash
# Create a temporary build directory
mkdir -p temp_build
cd temp_build

# Download the source
pip download iisignature
tar -xzf iisignature-0.24.tar.gz
cd iisignature-0.24

# Build and install
python setup.py build_ext --inplace
python setup.py install
```

3. Verify the installation:
```bash
python -c "import iisignature; print(iisignature.__version__)"
```

#### Method 3: Using an Older Python Version

If you're still having issues, create a new environment with Python 3.10:

1. Install Python 3.10 (using homebrew on macOS):
```bash
brew install python@3.10
```

2. Create a virtual environment with Python 3.10:
```bash
/opt/homebrew/bin/python3.10 -m venv venv_py310
source venv_py310/bin/activate
```

3. Install dependencies and iisignature:
```bash
pip install -r requirements.txt
pip install iisignature
```

If direct installation still fails, use the manual build method (Method 2) within this environment.

## Usage

[Usage instructions will go here]

## Project Structure

- `kernels/`: Contains the signature kernel implementation
- `experiments/`: Example experiments using the signature kernel
- `tests/`: Test cases for the implementation
- `utils/`: Utility functions for evaluation and visualization

## License

[License information]