#!/usr/bin/env python3
"""
Subscriber implementation for the shmem library using NumPy arrays.
"""

import sys
import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Any
import re

# Import the SMQueue class from our package
from shmem import SMQueue

# Constants
QUEUE_NAME = "/my_queue_example_2"
MESSAGE_SIZE = int(10 * 1024 * 1024)  # 10MB message size
HEADER_SIZE = 64  # Reduced header size

# Global variables
running = True
queue: Optional[SMQueue] = None

# Compile regex pattern for header parsing
HEADER_PATTERN = re.compile(r"Message #(\d+) (\d+)")


def parse_header(header_bytes: NDArray[np.uint8]) -> Tuple[int, int]:
    """
    Parse the header and extract message number and timestamp.

    Args:
        header_bytes: NumPy array containing the header bytes

    Returns:
        Tuple of (message_number, timestamp)

    Raises:
        ValueError: If the header cannot be parsed
    """
    # Convert bytes to string, stopping at the first null byte
    header_str = bytes(header_bytes).split(b"\0")[0].decode("utf-8")

    # Use regex to extract message number and timestamp
    match = HEADER_PATTERN.match(header_str)
    if not match:
        raise ValueError(f"Failed to parse header: {header_str}")

    msg_num = int(match.group(1))
    timestamp = int(match.group(2))

    return msg_num, timestamp


def main() -> None:
    """Main function for the subscriber."""
    global running, queue

    try:
        # Open existing queue
        queue = SMQueue.open(QUEUE_NAME)
        print("Subscriber started. Press Ctrl+C to stop.")
        print(
            f"Element size: {queue.element_size()} bytes, Max elements: {queue.max_elements()}"
        )

        # For calculating running average
        running_avg = 0.0
        alpha = 0.1  # Weight for new samples (0.1 = 10% weight to new sample)
        first_message = True
        message_count = 0
        expected_msg_id = 0  # Track the expected message ID
        
        # Add timing for the entire loop
        loop_start_time = time.time()
        messages_processed = 0
        
        print("Starting message processing loop...")

        while running:
            # Try to pop a message (non-blocking)
            # The profiling is now built into the try_pop_np method
            message: Optional[NDArray[np.uint8]] = queue.pop_np()
            
            if message is not None:
                messages_processed += 1
                # Get current time for latency calculation in milliseconds
                receive_time = time.time() * 1000  # milliseconds since epoch

                # Parse header
                try:
                    msg_num, send_timestamp = parse_header(message[:HEADER_SIZE])

                    # Print basic message info
                    print(f"Received: Message #{msg_num}")
                    print(f"  Send timestamp: {send_timestamp} ms, Receive timestamp: {receive_time} ms")
                    print(f"  Time difference: {receive_time - send_timestamp} ms")

                    # Only calculate transfer time if the message ID matches what we expect
                    if msg_num == expected_msg_id:
                        # Ensure the timestamp is valid
                        if send_timestamp <= 0 or send_timestamp > receive_time:
                            print(f"  Invalid timestamp: {send_timestamp}")
                        else:
                            # Calculate transfer time in milliseconds
                            # Both receive_time and send_timestamp are now in milliseconds
                            transfer_time_ms = receive_time - send_timestamp

                            # Sanity check on transfer time
                            if transfer_time_ms < 0 or transfer_time_ms > 10000:
                                print(
                                    f"  Suspicious transfer time: {transfer_time_ms}ms, ignoring"
                                )
                            else:
                                message_count += 1

                                # Update running average using exponential moving average
                                if first_message:
                                    running_avg = transfer_time_ms
                                    first_message = False
                                else:
                                    running_avg = (
                                        1 - alpha
                                    ) * running_avg + alpha * transfer_time_ms

                                # Print timing information
                                print(f"  Transfer time: {transfer_time_ms:.3f} ms")
                                print(
                                    f"  Running average: {running_avg:.3f} ms (over {message_count} messages)"
                                )
                    else:
                        print(f"  Message ID mismatch. Expected: {expected_msg_id}, Got: {msg_num}")
                    
                    # Update expected message ID for next message
                    expected_msg_id = msg_num + 1

                except ValueError as e:
                    print(f"Error parsing message: {e}")
            else:
                # Queue is empty, wait a bit before trying again
                # Reduce the sleep time to minimize polling delay
                time.sleep(0.001)  # 1 millisecond (was 100 microseconds)
            
            # Calculate and print loop statistics every 10 seconds
            current_time = time.time()
            elapsed_time = current_time - loop_start_time
            if elapsed_time >= 10:
                loop_rate = messages_processed / elapsed_time
                print(f"\nLoop statistics:")
                print(f"  Messages processed: {messages_processed}")
                print(f"  Elapsed time: {elapsed_time:.2f} seconds")
                print(f"  Processing rate: {loop_rate:.2f} messages/second")
                print(f"  Average processing time: {1000/loop_rate if loop_rate > 0 else 0:.2f} ms/message\n")
                
                # Reset statistics
                loop_start_time = current_time
                messages_processed = 0

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
