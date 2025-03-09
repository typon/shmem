# Shmem Python Bindings

Python bindings for the shmem library using nanobind. This library provides a shared memory queue implementation for inter-process communication.

## Features

- Uses POSIX shared memory and semaphores for IPC
- Thread and process safe
- Fixed-size message support
- Non-blocking operations available
- NumPy array interface for efficient data handling

## Requirements

- C++17 compiler
- CMake 3.15+
- Python 3.8+
- NumPy

## Building

To build the Python bindings:

```bash
# Create a build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build .

# Copy the Python module to the python directory
cp pyshmem*.so ../python/
```

## Installation

After building, you can install the Python package using pip or uv:

### Using pip

```bash
cd python
pip install -e .
```

### Using uv (faster installation)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you don't have it
pip install uv

# Install the package with uv
cd python
uv pip install -e .

# Or create a virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Usage

### Publisher Example

```python
import numpy as np
from shmem import SMQueue

# Create a shared memory queue
queue_name = "/my_queue"
max_elements = 10
element_size = 1024  # 1KB messages
queue = SMQueue.create(queue_name, max_elements, element_size)

# Create a message as a NumPy array
message = np.zeros(element_size, dtype=np.uint8)
message[:5] = np.frombuffer(b"Hello", dtype=np.uint8)

# Push the message to the queue
queue.push(message)

# Clean up
SMQueue.destroy(queue_name)
```

### Subscriber Example

```python
import numpy as np
from shmem import SMQueue

# Open an existing shared memory queue
queue_name = "/my_queue"
queue = SMQueue.open(queue_name)

# Pop a message (blocking)
message = queue.pop()
print(bytes(message[:5]).decode())  # Prints "Hello"

# Try to pop a message (non-blocking)
message = queue.try_pop()
if message is not None:
    print("Got message")
else:
    print("Queue is empty")
```

## Running the Examples

The repository includes complete publisher and subscriber examples:

```bash
# Start the subscriber in one terminal
python python/sub.py

# Start the publisher in another terminal
python python/pub.py

# Clean up shared memory
python python/pub.py --cleanup
```

### Multiprocessing Example

The repository also includes a self-contained example using multiprocessing:

```bash
# Run the multiprocessing example
python python/example.py
```

This example demonstrates:
- Creating publisher and subscriber processes
- Using NumPy arrays for efficient data transfer
- Proper signal handling and cleanup
- Synchronization between processes

## Development

This project uses modern Python packaging with `pyproject.toml`. The package structure is:

```
python/
├── pyproject.toml    # Package metadata and dependencies
├── requirements.txt  # For compatibility with older tools
├── example.py        # Multiprocessing example
└── shmem/
    └── __init__.py   # Package code
``` 