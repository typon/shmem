# Try to import the C++ extension module
try:
    # First try to import from the current directory (where the .so file should be)
    from . import cyshmem
except ImportError:
    try:
        # If that fails, try to import from the parent directory
        import cyshmem
    except ImportError:
        raise ImportError(
            "Failed to import cyshmem module. Make sure it's compiled and in the Python path."
        )

# Import the SMQueue class from the extension module
SMQueue = cyshmem.SMQueue

__all__ = ["SMQueue"] 