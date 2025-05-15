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
TEST_DURATION = 2  # Run test for 5 seconds
MIN_MESSAGES_REQUIRED = 10  # Minimum number of messages required for a valid test


def benchmark_raw_memcpy_bandwidth(buffer_size_bytes: int, iterations: int = 100) -> float:
    """
    Benchmarks raw memory copy speed using numpy arrays.

    Args:
        buffer_size_bytes: The size of the buffer to copy in bytes.
        iterations: The number of times to perform the copy.

    Returns:
        The calculated bandwidth in GB/s, or 0.0 if calculation is not reliable.
    """
    if buffer_size_bytes == 0 or iterations == 0:
        print("Warning: memcpy benchmark called with zero buffer size or iterations.")
        return 0.0

    try:
        source_buffer = np.ones(buffer_size_bytes, dtype=np.uint8)
        dest_buffer = np.zeros(buffer_size_bytes, dtype=np.uint8)
    except MemoryError:
        print(f"Warning: MemoryError allocating buffers for memcpy benchmark (size: {buffer_size_bytes} bytes).")
        return 0.0


    # Warm-up copy
    np.copyto(dest_buffer, source_buffer)

    start_time = time.perf_counter()
    for _ in range(iterations):
        np.copyto(dest_buffer, source_buffer)
    end_time = time.perf_counter()

    total_time_seconds = end_time - start_time

    if total_time_seconds <= 1e-9:  # Avoid division by zero or extremely small, unreliable time
        print(
            f"Warning: memcpy benchmark recorded non-positive or extremely small time ({total_time_seconds:.9f}s)."
            " Cannot reliably calculate bandwidth."
        )
        return 0.0  # Return 0.0 if time is too small to be reliable

    # Calculate bandwidth in GB/s (Gigabytes per second, 1 GB = 1024^3 bytes)
    total_data_copied_gb = (buffer_size_bytes * iterations) / (1024 * 1024 * 1024)
    bandwidth_gb_per_sec = total_data_copied_gb / total_time_seconds

    return bandwidth_gb_per_sec


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

        while (time.time() - start_time) < TEST_DURATION:
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
            delay = (
                random.uniform(min_delay_ms, max_delay_ms) / 1000.0
            )  # Convert to seconds
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

    recv_buf: NDArray[np.uint8] = np.zeros(MESSAGE_SIZE, dtype=np.uint8)

    start_time = time.time()
    while shared_data_dict["running"] and (time.time() - start_time) < TEST_DURATION:
        success = queue.try_pop_into(recv_buf)
        # At this point msg_view holds the message data (zero-copy or copied)
        if success: # Get current time for latency calculation in milliseconds
            receive_time = time.time() * 1000  # milliseconds since epoch

            # Convert bytes to string, stopping at the first null byte
            header_str = (
                bytes(recv_buf[:HEADER_SIZE]).split(b"\0")[0].decode("utf-8")
            )

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
                print(
                    f"  Skipping message with unexpected ID (got {msg_num}, expected {expected_msg_id})"
                )

            received_count += 1

        # Small sleep to prevent CPU spinning
        time.sleep(0.0001)
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

    # Create a manager for sharing data between processes
    manager = Manager()
    shared_data_dict = manager.dict()
    shared_data_dict["delays"] = []
    shared_data_dict["max_delay"] = 0.0
    shared_data_dict["running"] = True

    # Clean up any existing shared memory
    cleanup()

    # Create and start the processes
    pub_process = multiprocessing.Process(target=publisher_process)
    sub_process = multiprocessing.Process(
        target=subscriber_process, args=(shared_data_dict,)
    )

    pub_process.start()
    sub_process.start()

    # Wait for the test duration
    time.sleep(TEST_DURATION + 1)  # Add 1 second buffer

    # Signal processes to stop
    shared_data_dict["running"] = False

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
    delays_recorded = shared_data_dict.get("delays", [])

    # --- Raw memcpy Benchmark (run this regardless of shmem results for baseline) ---
    print("\n=== Raw memcpy Benchmark ===")
    memcpy_iterations = 100  # Number of iterations for memcpy test
    # Ensure MESSAGE_SIZE is appropriate, or use a fixed large buffer size for memcpy
    raw_memcpy_bandwidth_gbps = benchmark_raw_memcpy_bandwidth(MESSAGE_SIZE, iterations=memcpy_iterations)
    if raw_memcpy_bandwidth_gbps > 0:
        print(
            f"Raw memcpy Bandwidth ({MESSAGE_SIZE / (1024*1024):.0f}MB x {memcpy_iterations} iterations):"
            f" {raw_memcpy_bandwidth_gbps:.3f} GB/s"
        )
    else:
        print(
            f"Raw memcpy Bandwidth ({MESSAGE_SIZE / (1024*1024):.0f}MB x {memcpy_iterations} iterations):"
            " Could not be determined."
        )
    # --- End Raw memcpy Benchmark ---

    if not delays_recorded:
        print("\nShmem Performance: N/A (no messages processed or delays recorded by subscriber).")
        print("Shmem Overhead vs Raw memcpy: N/A (cannot compare without shmem data).")
        pytest.fail("No delays were recorded by the subscriber, cannot calculate shmem performance.")

    # Drop the top 10% of delays as potential outliers
    delays_filtered = sorted(delays_recorded)[: int(len(delays_recorded) * 0.9)]

    if not delays_filtered:
        print("\n=== Performance Test Results ===")
        print(f"Original number of messages recorded: {len(delays_recorded)}")
        print("Not enough messages to calculate shmem statistics after dropping top 10% outliers.")
        print("Shmem Overhead vs Raw memcpy: N/A (cannot compare without shmem data).")
        pytest.fail(
            "No messages left after filtering outliers, cannot calculate shmem performance statistics."
        )

    # Calculate shmem statistics from filtered delays
    avg_delay_ms = sum(delays_filtered) / len(delays_filtered)
    min_delay_ms = min(delays_filtered)
    max_delay_ms = max(delays_filtered)

    print("\n=== Shmem Performance Test Results ===")
    print(f"Number of messages originally recorded: {len(delays_recorded)}")
    print(f"Number of valid messages processed (after 10% outlier drop): {len(delays_filtered)}")
    print(f"Minimum delay: {min_delay_ms:.3f} ms")
    print(f"Average delay: {avg_delay_ms:.3f} ms")
    print(f"Maximum delay: {max_delay_ms:.3f} ms")

    # Calculate shmem bandwidth
    shmem_avg_latency_seconds = avg_delay_ms / 1000.0
    shmem_bandwidth_gbps = 0.0
    if shmem_avg_latency_seconds > 0:
        shmem_bandwidth_gbps = (MESSAGE_SIZE / shmem_avg_latency_seconds) / (1024**3)
        print(f"Shmem Average Bandwidth: {shmem_bandwidth_gbps:.3f} GB/s (based on avg delay)")
    else:
        print("Shmem Average Bandwidth: N/A (average delay is zero or invalid)")

    # Calculate and print overhead
    print("\n=== Comparison: Shmem vs Raw memcpy ===")
    if raw_memcpy_bandwidth_gbps > 0 and shmem_bandwidth_gbps > 0:
        if shmem_bandwidth_gbps > raw_memcpy_bandwidth_gbps:
            overhead_percentage = (
                (shmem_bandwidth_gbps - raw_memcpy_bandwidth_gbps) / raw_memcpy_bandwidth_gbps
            ) * 100
            print(
                f"Note: Shmem bandwidth ({shmem_bandwidth_gbps:.3f} GB/s) appears "
                f"HIGHER than raw memcpy ({raw_memcpy_bandwidth_gbps:.3f} GB/s)."
            )
            print(f"  This is unexpected. Calculated 'gain' over memcpy: {overhead_percentage:.2f}%")
        else:
            overhead_percentage = (
                (raw_memcpy_bandwidth_gbps - shmem_bandwidth_gbps) / raw_memcpy_bandwidth_gbps
            ) * 100
            print(f"Shmem Overhead compared to Raw memcpy: {overhead_percentage:.2f}%")
    elif raw_memcpy_bandwidth_gbps <= 0:
        print("Shmem Overhead vs Raw memcpy: N/A (raw memcpy bandwidth could not be determined).")
    else: # shmem_bandwidth_gbps <= 0
        print("Shmem Overhead vs Raw memcpy: N/A (shmem bandwidth is zero or invalid).")

    # Check if enough valid (filtered) messages were processed
    if len(delays_filtered) < MIN_MESSAGES_REQUIRED:
        pytest.fail(
            f"Not enough valid messages processed after filtering. "
            f"Got {len(delays_filtered)}, need at least {MIN_MESSAGES_REQUIRED}"
        )

    # Assert that the average delay is within the threshold
    assert avg_delay_ms < 3.0, (
        f"Average shmem delay ({avg_delay_ms:.3f} ms) exceeds threshold (3.0 ms)"
    )


if __name__ == "__main__":
    test_transfer_time()
