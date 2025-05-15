#pragma once

#include <fcntl.h>     // for O_CREAT, O_RDWR
#include <semaphore.h> // for sem_t, sem_open
#include <sys/mman.h>  // for shm_open, mmap
#include <sys/stat.h>  // for fstat
#include <unistd.h>    // for close, ftruncate

#include <cassert>   // for assert
#include <cerrno>    // for errno
#include <cstddef>   // for std::byte
#include <cstring>   // for memcpy, strncpy
#include <limits>    // for std::numeric_limits
#include <stdexcept> // for std::runtime_error
#include <string>    // for std::string

namespace shmem {

/*
 * A C++17 library implementing a shared memory queue
 * using POSIX shared memory and semaphores for inter-process communication.
 *
 * Features:
 * - Uses shm_open/mmap for shared memory
 * - Uses named semaphores for synchronization (macOS compatible)
 * - Thread and process safe
 * - Fixed-size message support
 * - Non-blocking operations available
 */

// Helper functions
namespace detail {
// Close a file descriptor safely
inline void safe_close(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}
} // namespace detail

// Forward declarations
class SMQueue {
  public:
    // Create a new shared memory queue
    // max_elements: Maximum number of elements in the queue
    // element_size: Size of each element in bytes
    static SMQueue create(const std::string& name, std::size_t max_elements, std::size_t element_size);

    // Open an existing shared memory queue
    static SMQueue open(const std::string& name);

    // Destroy a shared memory queue
    static void destroy(const std::string& name);

    // Move constructor and assignment
    SMQueue(SMQueue&& other) noexcept;
    SMQueue& operator=(SMQueue&& other) noexcept;

    // Destructor
    ~SMQueue() noexcept;

    // Push a message to the queue
    // If the queue is full, the oldest message will be dropped to make room
    // Returns true if no messages were dropped, false if some were dropped
    bool push(const std::byte* data);

    // Pop a message from the queue (blocking)
    // Returns true if successful, false if an error occurred
    bool pop(std::byte* buffer);

    // Try to pop a message (non-blocking)
    bool try_pop(std::byte* buffer);

    // Zero-copy borrow of the next message (non-blocking). Returns true on success. The caller receives
    // a pointer to the message data living inside the queue and the element index that must later be
    // released via commit_pop(index).
    bool borrow(std::byte const** data_ptr, std::size_t& index);

    // Release a previously borrowed element (identified by its index) and make the slot reusable.
    void commit_pop(std::size_t index);

    // Close the queue
    void close();

    // Get maximum number of elements
    std::size_t max_elements() const;

    // Get element size in bytes
    std::size_t element_size() const;

    // Get queue name
    const std::string& name() const;

  private:
    // Control block structure
    struct alignas(64) ControlBlock {
        std::size_t max_elements; // Maximum number of elements
        std::size_t element_size; // Size of each element in bytes
        std::size_t head;         // Write position (element index)
        std::size_t tail;         // Read position (element index)
        std::size_t count;        // Number of elements in the queue
        char mutex_name[128];     // Mutex semaphore name
        char items_name[128];     // Items semaphore name
    };

    // Constructor
    SMQueue(const std::string& name, void* addr, std::size_t size);

    // Get control block
    ControlBlock* get_control_block() const;

    // Get data buffer
    std::byte* get_data_buffer() const;

    // Get element at index
    std::byte* get_element(std::size_t index) const;

    // Create semaphore name
    static std::string make_sem_name(const std::string& name, const std::string& suffix);

    // Initialize semaphores
    void init_semaphores(ControlBlock* cb);

    // Open existing semaphores
    void open_semaphores();

    // Member variables
    std::string m_name; // Queue name
    void* m_addr;       // Mapped memory address
    std::size_t m_size; // Memory size
    sem_t* m_mutex;     // Mutex semaphore
    sem_t* m_items;     // Items semaphore
};

} // namespace shmem