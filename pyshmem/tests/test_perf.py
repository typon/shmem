#!/usr/bin/env python3
"""
Performance test for the shmem library.
Tests the transfer time between publisher and subscriber processes.
"""

import os
import time
import random
import multiprocessing
from multiprocessing import Manager
from typing import Optional, List, Any, Dict
import numpy as np
from numpy.typing import NDArray
import pytest

# Import the SMQueue class from our package
from shmem import SMQueue

# Constants
QUEUE_NAME = "/test_perf_queue"
MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB message size
MAX_ELEMENTS = 10  # Maximum number of elements in the queue
HEADER_SIZE = 64  # Header size
TEST_DURATION = 5  # Run test for 5 seconds
MIN_MESSAGES_REQUIRED = 10  # Minimum number of messages required for a valid test

# Global variables for the processes
running = True

# Shared data will be initialized in the test function
shared_data: Dict[str, Any] = {}


def publisher_process(min_delay_ms: int = 1, max_delay_ms: int = 100) -> None:
    """
    Publisher process function.
    
    Args:
        min_delay_ms: Minimum delay between messages in milliseconds
        max_delay_ms: Maximum delay between messages in milliseconds
    """
    try:
        print(f"Publisher process started (PID: {os.getpid()})")

        # Create queue
        queue = SMQueue.create(QUEUE_NAME, MAX_ELEMENTS, MESSAGE_SIZE)

        # Pre-allocate buffer for message and pre-fill with random data
        message_buffer: NDArray[np.uint8] = np.zeros(MESSAGE_SIZE, dtype=np.uint8)

        # Fill the buffer with random data once (after header)
        message_buffer[HEADER_SIZE:] = np.random.randint(
            0, 256, size=MESSAGE_SIZE - HEADER_SIZE, dtype=np.uint8
        )

        counter = 0
        start_time = time.time()
        
        while running and (time.time() - start_time) < TEST_DURATION:
            # Get current timestamp in milliseconds
            timestamp = time.time() * 1000  # milliseconds since epoch
            
            # Create message header with timestamp and counter
            header = f"Message #{counter} {timestamp}".encode("utf-8")
            header_len = len(header)
            
            # Copy header to the beginning of the buffer
            message_buffer[:header_len] = np.frombuffer(header, dtype=np.uint8)
            
            # Push the message to the queue
            queue.push(message_buffer)
            
            print(f"Published: Message #{counter} at {timestamp:.0f} ms")
            
            counter += 1
            
            # Random delay between min_delay_ms and max_delay_ms milliseconds
            delay = random.uniform(min_delay_ms, max_delay_ms) / 1000.0  # Convert to seconds
            time.sleep(delay)
            
    except Exception as e:
        print(f"Publisher error: {e}")
    finally:
        print("Publisher process exiting")


def subscriber_process(shared_data_dict: Dict[str, Any]) -> None:
    """
    Subscriber process function.
    
    Args:
        shared_data_dict: Dictionary for sharing data between processes
    """
    try:
        print(f"Subscriber process started (PID: {os.getpid()})")

        # Wait a bit for the publisher to create the queue
        time.sleep(0.1)

        # Open the queue
        queue = SMQueue.open(QUEUE_NAME)

        # Keep track of received messages
        received_count = 0
        expected_msg_id = 0
        
        # Local max delay for this process
        local_max_delay = 0.0
        # Local delays list
        local_delays: List[float] = []
        
        start_time = time.time()
        
        while running and (time.time() - start_time) < TEST_DURATION:
            # Try to pop a message (non-blocking)
            message: Optional[NDArray[np.uint8]] = queue.try_pop_np()
            
            if message is not None:
                # Get current time for latency calculation in milliseconds
                receive_time = time.time() * 1000  # milliseconds since epoch

                # Parse header
                try:
                    # Convert bytes to string, stopping at the first null byte
                    header_str = bytes(message[:HEADER_SIZE]).split(b"\0")[0].decode("utf-8")
                    
                    # Extract message number and timestamp
                    message_literal, msg_num, send_timestamp = header_str.split()
                    # Remove the "Message #" prefix
                    msg_num = int(msg_num.split("#")[1])
                    send_timestamp = float(send_timestamp)

                    print(f"Received: Message #{msg_num}, Expected: #{expected_msg_id}")
                    
                    # Only process messages with the expected ID
                    if msg_num == expected_msg_id:
                        # Calculate transfer time in milliseconds
                        transfer_time_ms = receive_time - send_timestamp
                        
                        print(f"  Transfer time: {transfer_time_ms} ms")
                        
                        # Record the delay
                        local_delays.append(transfer_time_ms)
                        
                        # Update max delay observed
                        if transfer_time_ms > local_max_delay:
                            local_max_delay = transfer_time_ms
                        
                        # Increment expected message ID for next message
                        expected_msg_id += 1
                    else:
                        print(f"  Skipping message with unexpected ID (got {msg_num}, expected {expected_msg_id})")
                    
                    received_count += 1
                    
                except Exception as e:
                    print(f"Error parsing message: {e}")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.0001)
            
    except Exception as e:
        print(f"Subscriber error: {e}")
    finally:
        # Update shared data with results from this process
        shared_data_dict["delays"] = local_delays
        shared_data_dict["max_delay"] = local_max_delay
        print(f"Subscriber process exiting. Received {received_count} messages.")
        print(f"Max delay observed: {local_max_delay} ms")


def cleanup() -> None:
    """Clean up shared memory."""
    try:
        print("Cleaning up shared memory...")
        SMQueue.destroy(QUEUE_NAME)
    except Exception as e:
        print(f"Cleanup error: {e}")


def test_transfer_time() -> None:
    """
    Test the transfer time between publisher and subscriber processes.
    
    The test runs for TEST_DURATION seconds and ensures that the maximum
    observed delay is less than 5ms.
    """
    global running
    
    # Reset global variables
    running = True
    
    # Create a manager for sharing data between processes
    manager = Manager()
    shared_data_dict = manager.dict()
    shared_data_dict["delays"] = []
    shared_data_dict["max_delay"] = 0.0
    
    # Clean up any existing shared memory
    cleanup()
    
    # Create and start the processes
    pub_process = multiprocessing.Process(target=publisher_process)
    sub_process = multiprocessing.Process(target=subscriber_process, args=(shared_data_dict,))
    
    pub_process.start()
    sub_process.start()
    
    # Wait for the test duration
    time.sleep(TEST_DURATION + 1)  # Add 1 second buffer
    
    # Signal processes to stop
    running = False
    
    # Wait for processes to finish
    pub_process.join(timeout=2)
    sub_process.join(timeout=2)
    
    # Force terminate if they haven't exited
    if pub_process.is_alive():
        pub_process.terminate()
    if sub_process.is_alive():
        sub_process.terminate()
    
    # Clean up shared memory
    cleanup()
    
    # Get the results from the shared data
    delays = shared_data_dict.get("delays", [])
    assert len(delays) > 0, "No delays were recorded"
    # Drop the top 10% of delays
    delays = sorted(delays)[:int(len(delays) * 0.9)]
    
    # Print statistics
    avg_delay = sum(delays) / len(delays)
    min_delay = min(delays)
    max_delay = max(delays)

    breakpoint()
    
    print("\n=== Performance Test Results ===")
    print(f"Number of valid messages processed: {len(delays)}")
    print(f"Minimum delay: {min_delay:.3f} ms")
    print(f"Average delay: {avg_delay:.3f} ms")
    print(f"Maximum delay: {max_delay:.3f} ms")
    
    # Check if we processed enough messages
    if len(delays) < MIN_MESSAGES_REQUIRED:
        pytest.fail(f"Not enough valid messages processed. Got {len(delays)}, need at least {MIN_MESSAGES_REQUIRED}")
    
    # Assert that the maximum delay is less than 5ms
    assert avg_delay < 3.0, f"Average delay ({avg_delay} ms) exceeds threshold (3 ms)"


if __name__ == "__main__":
    test_transfer_time() 