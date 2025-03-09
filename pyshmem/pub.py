#!/usr/bin/env python3
"""
Publisher implementation for the shmem library using NumPy arrays.
"""

import argparse
import sys
import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Any
import atexit

# Import the SMQueue class from our package
from shmem import SMQueue

# Constants
QUEUE_NAME = "/my_queue_example_2"
MESSAGE_SIZE = int(0.5 * 1024 * 1024)  # 10MB message size
MAX_ELEMENTS = 100  # Maximum number of elements in the queue
HEADER_SIZE = 64  # Reduced header size

# Global variables
running = True
queue: Optional[SMQueue] = None


def cleanup() -> None:
    """Clean up shared memory on exit."""
    try:
        print("Cleaning up shared memory...")
        SMQueue.destroy(QUEUE_NAME)
    except Exception as e:
        print(f"Cleanup error: {e}")


def main() -> None:
    """Main function for the publisher."""
    global running, queue

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Shared memory publisher using NumPy arrays"
    )
    parser.add_argument(
        "--cleanup", "-c", action="store_true", help="Clean up shared memory and exit"
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=0.1, 
        help="Delay between messages in seconds (default: 0.1)"
    )
    args = parser.parse_args()

    # Check for cleanup flag
    if args.cleanup:
        cleanup()
        return


    # Register cleanup function to run on exit
    atexit.register(cleanup)

    try:
        # First try to clean up any existing shared memory
        try:
            SMQueue.destroy(QUEUE_NAME)
        except:
            # Ignore errors during initial cleanup
            pass

        # Create queue with fixed-size messages
        queue = SMQueue.create(QUEUE_NAME, MAX_ELEMENTS, MESSAGE_SIZE)

        counter = 0
        print("Publisher started. Press Ctrl+C to stop.")
        print(f"Message size: {MESSAGE_SIZE} bytes, Max elements: {MAX_ELEMENTS}")
        print(f"Delay between messages: {args.delay} seconds")

        # Pre-allocate buffer for message and pre-fill with random data
        message_buffer: NDArray[np.uint8] = np.zeros(MESSAGE_SIZE, dtype=np.uint8)

        # Fill the buffer with random data once (after header)
        message_buffer[HEADER_SIZE:] = np.random.randint(
            0, 256, size=MESSAGE_SIZE - HEADER_SIZE, dtype=np.uint8
        )

        while running:
            # Get current timestamp in milliseconds (not microseconds)
            # This is a simpler approach that might help diagnose the issue
            timestamp = int(time.time() * 1000)  # milliseconds since epoch
            
            # Create message header with timestamp and counter
            header = f"Message #{counter} {timestamp}".encode("utf-8")
            header_len = len(header)
            
            # Copy header to the beginning of the buffer
            message_buffer[:header_len] = np.frombuffer(header, dtype=np.uint8)
            
            # Push the message to the queue
            push_start = time.time()
            no_drop = queue.push(message_buffer)
            push_end = time.time()
            push_time = (push_end - push_start) * 1000  # in milliseconds
            
            if no_drop:
                print(f"Published: Message #{counter} at {timestamp} ms (push took {push_time:.3f} ms)")
            else:
                print(f"Published (with drop): Message #{counter} at {timestamp} ms (push took {push_time:.3f} ms)")
            
            counter += 1
            
            # Use the delay from command line arguments
            time.sleep(args.delay)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
