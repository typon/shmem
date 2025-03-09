#include <signal.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "shmem.h"

volatile bool running = true;
std::string queue_name = "/my_queue_example_2";
constexpr size_t MESSAGE_SIZE = 10 * 1024 * 1024; // 10MB message size
constexpr size_t HEADER_SIZE = 64;                // Reduced header size

void signal_handler(int signum) { running = false; }

// Fast function to parse the header and extract timestamp
bool parse_header(const char* header, int& msg_num, long long& timestamp) {
    // Skip "Message #"
    const char* p = header;
    while (*p && *p != '#')
        p++;
    if (!*p)
        return false;
    p++; // Skip '#'

    // Parse message number
    char* end;
    msg_num = strtol(p, &end, 10);
    if (p == end)
        return false;

    // Skip to timestamp
    p = end;
    while (*p && (*p == ' ' || *p == '\t'))
        p++;
    if (!*p)
        return false;

    // Parse timestamp
    timestamp = strtoll(p, nullptr, 10);
    return true;
}

int main() {
    // Set up signal handling for clean shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        // Open existing queue
        auto queue = shmem::SMQueue::open(queue_name);
        std::cout << "Subscriber started. Press Ctrl+C to stop." << std::endl;
        std::cout << "Element size: " << queue.element_size() << " bytes, Max elements: " << queue.max_elements()
                  << std::endl;

        // For calculating running average - use exponential moving average for efficiency
        double running_avg = 0.0;
        const double alpha = 0.1; // Weight for new samples (0.1 = 10% weight to new sample)
        bool first_message = true;
        int message_count = 0;

        // Set stdout to be unbuffered for more accurate timing
        std::cout.setf(std::ios::unitbuf);

        // Pre-allocate buffer for message - use aligned allocation for better performance
        alignas(64) std::vector<std::byte> buffer(MESSAGE_SIZE);

        // Pre-allocate header buffer
        alignas(64) char header_buffer[HEADER_SIZE + 1]; // +1 for null terminator

        // Use steady_clock for more accurate timing
        using clock = std::chrono::steady_clock;

        while (running) {
            // Try to pop a message (non-blocking)
            if (queue.try_pop(buffer.data())) {
                // Get current time for latency calculation - immediately after receiving
                auto now = clock::now();
                auto receive_time =
                    std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

                // Extract header from the message (null-terminate it)
                std::memcpy(header_buffer, buffer.data(), HEADER_SIZE);
                header_buffer[HEADER_SIZE] = '\0';

                // Parse header directly without string conversion
                int msg_num;
                long long send_timestamp;
                if (!parse_header(header_buffer, msg_num, send_timestamp)) {
                    std::cerr << "Failed to parse header: " << header_buffer << std::endl;
                    continue;
                }

                // Ensure the timestamp is valid
                if (send_timestamp <= 0 || send_timestamp > receive_time) {
                    std::cerr << "Invalid timestamp: " << send_timestamp << std::endl;
                    continue;
                }

                // Calculate transfer time in milliseconds
                double transfer_time_ms = (receive_time - send_timestamp) / 1000.0;

                // Sanity check on transfer time
                if (transfer_time_ms < 0 || transfer_time_ms > 10000) {
                    std::cerr << "Suspicious transfer time: " << transfer_time_ms << "ms, ignoring" << std::endl;
                    continue;
                }

                message_count++;

                // Update running average using exponential moving average
                if (first_message) {
                    running_avg = transfer_time_ms;
                    first_message = false;
                } else {
                    running_avg = (1 - alpha) * running_avg + alpha * transfer_time_ms;
                }

                // Print message and timing information
                std::cout << "Received: Message #" << msg_num << std::endl;
                std::cout << "  Transfer time: " << std::fixed << std::setprecision(3) << transfer_time_ms << " ms"
                          << std::endl;
                std::cout << "  Running average: " << std::fixed << std::setprecision(3) << running_avg << " ms (over "
                          << message_count << " messages)" << std::endl;
            } else {
                // Queue is empty, wait a bit before trying again - use a shorter sleep time
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        std::cout << "\nShutting down subscriber..." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}