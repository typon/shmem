#include <signal.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "shmem.h"

volatile bool running = true;
std::string queue_name = "/my_queue_example_2";
constexpr size_t MESSAGE_SIZE = 10 * 1024 * 1024; // 10MB message size
constexpr size_t MAX_ELEMENTS = 10;               // Maximum number of elements in the queue
constexpr size_t HEADER_SIZE = 64;                // Reduced header size

void signal_handler(int signum) { running = false; }

void cleanup() {
    try {
        std::cout << "Cleaning up shared memory..." << std::endl;
        shmem::SMQueue::destroy(queue_name);
    } catch (const std::exception& e) {
        std::cerr << "Cleanup error: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Check for cleanup flag
    if (argc > 1 && (strcmp(argv[1], "--cleanup") == 0 || strcmp(argv[1], "-c") == 0)) {
        cleanup();
        return 0;
    }

    // Set up signal handling for clean shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        // First try to clean up any existing shared memory
        try {
            shmem::SMQueue::destroy(queue_name);
        } catch (...) {
            // Ignore errors during initial cleanup
        }

        // Create queue with fixed-size messages
        auto queue = shmem::SMQueue::create(queue_name, MAX_ELEMENTS, MESSAGE_SIZE);

        int counter = 0;
        std::cout << "Publisher started. Press Ctrl+C to stop." << std::endl;
        std::cout << "Message size: " << MESSAGE_SIZE << " bytes, Max elements: " << MAX_ELEMENTS << std::endl;

        // Set stdout to be unbuffered for more accurate timing
        std::cout.setf(std::ios::unitbuf);

        // Setup random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 255);

        // Pre-allocate buffer for message and pre-fill with random data
        std::vector<std::byte> message_buffer(MESSAGE_SIZE);

        // Fill the buffer with random data once (after header)
        for (size_t i = HEADER_SIZE; i < MESSAGE_SIZE; i++) {
            message_buffer[i] = static_cast<std::byte>(distrib(gen));
        }

        // Use high-resolution clock for more accurate timing
        using clock = std::chrono::high_resolution_clock;

        while (running) {
            // Get current timestamp in microseconds - use steady_clock for better precision
            auto now = std::chrono::steady_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

            // Create message header with timestamp and counter - use snprintf for efficiency
            int header_len = snprintf(reinterpret_cast<char*>(message_buffer.data()), HEADER_SIZE, "Message #%d %lld",
                                      counter++, (long long)timestamp);

            // No need to fill the rest with random data - already done once

            // Push the message to the queue
            bool no_drop = queue.push(message_buffer.data());
            if (no_drop) {
                std::cout << "Published: Message #" << counter - 1 << std::endl;
            } else {
                std::cout << "Published (with drop): Message #" << counter - 1 << std::endl;
            }

            // Add a delay between publishes
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        std::cout << "\nShutting down publisher..." << std::endl;

        // Clean up shared memory on exit
        cleanup();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
