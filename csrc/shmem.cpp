#include "shmem.h"

namespace shmem {

// Create a new shared memory queue
SMQueue SMQueue::create(const std::string& name, std::size_t max_elements, std::size_t element_size) {
    // Validate name - no spaces allowed for semaphore compatibility
    if (name.find(' ') != std::string::npos) {
        throw std::runtime_error("Queue name cannot contain spaces: " + name);
    }

    // Check for potential integer overflow
    if (max_elements > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::runtime_error("Queue size too large, would cause integer overflow");
    }

    // Open shared memory
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0660);
    if (fd < 0) {
        throw std::runtime_error("Failed to create shared memory: " + name + " (errno: " + std::to_string(errno) + ")");
    }

    // Calculate total size needed (header + data)
    std::size_t header_size = sizeof(ControlBlock);
    std::size_t data_size = max_elements * element_size;

    // Check for potential integer overflow in total size calculation
    if (header_size > std::numeric_limits<std::size_t>::max() - data_size) {
        detail::safe_close(fd);
        shm_unlink(name.c_str());
        throw std::runtime_error("Queue size too large, would cause integer overflow in total size calculation");
    }

    std::size_t total_size = header_size + data_size;

    // Set size
    if (ftruncate(fd, total_size) < 0) {
        detail::safe_close(fd);
        shm_unlink(name.c_str());
        throw std::runtime_error("Failed to set size of shared memory: " + name);
    }

    // Map memory
    void* addr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        detail::safe_close(fd);
        shm_unlink(name.c_str());
        throw std::runtime_error("Failed to map shared memory: " + name);
    }

    // Initialize control block
    ControlBlock* cb = static_cast<ControlBlock*>(addr);
    cb->max_elements = max_elements;
    cb->element_size = element_size;
    cb->head = 0;
    cb->tail = 0;
    cb->count = 0;

    // Create queue and initialize semaphores
    try {
        SMQueue queue(name, addr, total_size);
        queue.init_semaphores(cb);
        detail::safe_close(fd);
        return queue;
    } catch (const std::exception& e) {
        detail::safe_close(fd);
        if (addr != nullptr and addr != MAP_FAILED)
            munmap(addr, total_size);
        shm_unlink(name.c_str());
        throw; // Re-throw the exception
    }
}

// Open an existing shared memory queue
SMQueue SMQueue::open(const std::string& name) {
    // Validate name - no spaces allowed for semaphore compatibility
    if (name.find(' ') != std::string::npos) {
        throw std::runtime_error("Queue name cannot contain spaces: " + name);
    }

    // Open shared memory
    int fd = shm_open(name.c_str(), O_RDWR, 0660);
    if (fd < 0) {
        throw std::runtime_error("Failed to open shared memory: " + name);
    }

    // Get size
    struct stat st;
    if (fstat(fd, &st) != 0) {
        detail::safe_close(fd);
        throw std::runtime_error("Failed to get shared memory size: " + name);
    }

    // Map memory
    void* addr = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        detail::safe_close(fd);
        throw std::runtime_error("Failed to map shared memory: " + name);
    }

    // Create queue and open semaphores
    try {
        SMQueue queue(name, addr, static_cast<std::size_t>(st.st_size));
        queue.open_semaphores();
        detail::safe_close(fd);
        return queue;
    } catch (const std::exception& e) {
        detail::safe_close(fd);
        if (addr != nullptr and addr != MAP_FAILED)
            munmap(addr, st.st_size);
        throw;
    }
}

// Destroy a shared memory queue
void SMQueue::destroy(const std::string& name) {
    // Validate name - no spaces allowed for semaphore compatibility
    if (name.find(' ') != std::string::npos) {
        throw std::runtime_error("Queue name cannot contain spaces: " + name);
    }

    // Try to open the shared memory to check if it exists
    int fd = shm_open(name.c_str(), O_RDWR, 0660);
    if (fd < 0) {
        // If it doesn't exist, just return
        if (errno == ENOENT) {
            return;
        }
        throw std::runtime_error("Failed to open shared memory for destruction: " + name);
    }

    // Get size to check if it's a valid queue
    struct stat st;
    if (fstat(fd, &st) != 0) {
        detail::safe_close(fd);
        throw std::runtime_error("Failed to get shared memory size for destruction: " + name);
    }

    // Close the file descriptor
    detail::safe_close(fd);

    // Unlink shared memory
    shm_unlink(name.c_str());

    // Remove leading slash if present for semaphore names
    std::string sem_base_name = name;
    if (!sem_base_name.empty() && sem_base_name[0] == '/') {
        sem_base_name = sem_base_name.substr(1);
    }

    // Ensure base name is not too long (leave room for suffix)
    if (sem_base_name.length() > 24) {
        sem_base_name = sem_base_name.substr(0, 24);
    }

    // Unlink semaphores
    std::string mutex_name = make_sem_name(sem_base_name, "_mutex");
    std::string items_name = make_sem_name(sem_base_name, "_items");
    sem_unlink(mutex_name.c_str());
    sem_unlink(items_name.c_str());
}

// Move constructor
SMQueue::SMQueue(SMQueue&& other) noexcept
    : m_name(std::move(other.m_name)), m_addr(other.m_addr), m_size(other.m_size), m_mutex(other.m_mutex),
      m_items(other.m_items) {
    other.m_addr = nullptr;
    other.m_size = 0;
    other.m_mutex = nullptr;
    other.m_items = nullptr;
}

// Move assignment
SMQueue& SMQueue::operator=(SMQueue&& other) noexcept {
    if (this != &other) {
        close();
        m_name = std::move(other.m_name);
        m_addr = other.m_addr;
        m_size = other.m_size;
        m_mutex = other.m_mutex;
        m_items = other.m_items;
        other.m_addr = nullptr;
        other.m_size = 0;
        other.m_mutex = nullptr;
        other.m_items = nullptr;
    }
    return *this;
}

// Destructor
SMQueue::~SMQueue() noexcept { close(); }

// Push a message to the queue
// If the queue is full, the oldest message will be dropped to make room
bool SMQueue::push(const std::byte* data) {
    if (m_addr == nullptr) {
        throw std::runtime_error("SMQueue not initialized");
    }

    auto* cb = get_control_block();

    // Lock mutex
    int result;
    do {
        result = sem_wait(m_mutex);
    } while (result == -1 and errno == EINTR);

    if (result == -1) {
        throw std::runtime_error("Failed to lock mutex");
    }

    // Check if queue is full
    bool dropped_message = false;
    if (cb->count >= cb->max_elements) {
        // Drop the oldest message by advancing the tail
        cb->tail = (cb->tail + 1) % cb->max_elements;
        cb->count--;
        dropped_message = true;

        // Decrement the semaphore count since we're removing a message
        sem_trywait(m_items);
    }

    // Get pointer to the element at head position
    std::byte* dest = get_element(cb->head);

    std::memcpy(dest, data, cb->element_size);

    // Advance the head
    cb->head = (cb->head + 1) % cb->max_elements;
    cb->count++;

    // Signal that a new item is available
    sem_post(m_items);

    // Unlock the mutex
    sem_post(m_mutex);

    // Return true if no messages were dropped, false if one was dropped
    return !dropped_message;
}

// Pop a message from the queue (blocking)
bool SMQueue::pop(std::byte* buffer) {
    if (m_addr == nullptr) {
        return false;
    }

    auto* cb = get_control_block();

    // Wait for an item to be available
    int result;
    do {
        result = sem_wait(m_items);
    } while (result == -1 and errno == EINTR);

    if (result == -1) {
        return false;
    }

    // Lock mutex
    do {
        result = sem_wait(m_mutex);
    } while (result == -1 and errno == EINTR);

    if (result == -1) {
        sem_post(m_items);
        return false;
    }

    // Get pointer to the element at tail position
    const std::byte* src = get_element(cb->tail);

    std::memcpy(buffer, src, cb->element_size);

    // Advance the tail
    cb->tail = (cb->tail + 1) % cb->max_elements;
    cb->count--;

    // Unlock the mutex
    sem_post(m_mutex);

    return true;
}

// Try to pop a message (non-blocking)
bool SMQueue::try_pop(std::byte* buffer) {
    if (m_addr == nullptr) {
        return false;
    }

    auto* cb = get_control_block();

    // Try to get an item (non-blocking)
    if (sem_trywait(m_items) != 0) {
        return false;
    }

    // Lock mutex
    int result;
    do {
        result = sem_wait(m_mutex);
    } while (result == -1 and errno == EINTR);

    if (result == -1) {
        sem_post(m_items);
        return false;
    }

    // Get pointer to the element at tail position
    const std::byte* src = get_element(cb->tail);

    // MacOS memcpy works really well, so we use it
    std::memcpy(buffer, src, cb->element_size);

    // Advance the tail
    cb->tail = (cb->tail + 1) % cb->max_elements;
    cb->count--;

    // Unlock the mutex
    sem_post(m_mutex);
    return true;
}

// Close the queue
void SMQueue::close() {
    if (m_mutex != nullptr) {
        sem_close(m_mutex);
        m_mutex = nullptr;
    }

    if (m_items != nullptr) {
        sem_close(m_items);
        m_items = nullptr;
    }

    if (m_addr != nullptr) {
        munmap(m_addr, m_size);
        m_addr = nullptr;
        m_size = 0;
    }
}

// Get maximum number of elements
std::size_t SMQueue::max_elements() const { return m_addr != nullptr ? get_control_block()->max_elements : 0; }

// Get element size in bytes
std::size_t SMQueue::element_size() const { return m_addr != nullptr ? get_control_block()->element_size : 0; }

// Get queue name
const std::string& SMQueue::name() const { return m_name; }

// Constructor
SMQueue::SMQueue(const std::string& name, void* addr, std::size_t size)
    : m_name(name), m_addr(addr), m_size(size), m_mutex(nullptr), m_items(nullptr) {}

// Get control block
SMQueue::ControlBlock* SMQueue::get_control_block() const { return static_cast<ControlBlock*>(m_addr); }

// Get data buffer
std::byte* SMQueue::get_data_buffer() const {
    if (m_addr == nullptr) {
        throw std::runtime_error("SMQueue not initialized");
    }

    // Cast directly to std::byte* to avoid multiple pointer conversions
    return reinterpret_cast<std::byte*>(static_cast<char*>(m_addr) + sizeof(ControlBlock));
}

// Get element at index
std::byte* SMQueue::get_element(std::size_t index) const {
    if (m_addr == nullptr) {
        throw std::runtime_error("SMQueue not initialized");
    }

    auto* cb = get_control_block();
    // Calculate the offset directly to avoid multiple pointer arithmetic operations
    std::size_t offset = index * cb->element_size;
    return get_data_buffer() + offset;
}

// Create semaphore name
std::string SMQueue::make_sem_name(const std::string& name, const std::string& suffix) {
    // Assert that name doesn't start with a slash (for macOS compatibility)
    assert(name.empty() or name[0] != '/');

    // Create the semaphore name
    std::string result = name + suffix;

    // Assert that the name is not too long (macOS limit is 31 chars)
    assert(result.length() <= 30);

    return result;
}

// Initialize semaphores
void SMQueue::init_semaphores(ControlBlock* cb) {
    // Remove leading slash if present for semaphore names
    std::string sem_base_name = m_name;
    if (!sem_base_name.empty() and sem_base_name[0] == '/') {
        sem_base_name = sem_base_name.substr(1);
    }

    assert(sem_base_name.length() <= 24);

    // Create semaphore names
    std::string mutex_name = make_sem_name(sem_base_name, "_mutex");
    std::string items_name = make_sem_name(sem_base_name, "_items");

    // Check if names are too long for the buffer
    if (mutex_name.length() >= sizeof(cb->mutex_name) or items_name.length() >= sizeof(cb->items_name)) {
        throw std::runtime_error("Semaphore names too long: " + mutex_name + ", " + items_name);
    }

    // Store names in control block
    strncpy(cb->mutex_name, mutex_name.c_str(), sizeof(cb->mutex_name) - 1);
    cb->mutex_name[sizeof(cb->mutex_name) - 1] = '\0';

    strncpy(cb->items_name, items_name.c_str(), sizeof(cb->items_name) - 1);
    cb->items_name[sizeof(cb->items_name) - 1] = '\0';

    // Unlink any existing semaphores
    sem_unlink(mutex_name.c_str());
    sem_unlink(items_name.c_str());

    // Create semaphores
    m_mutex = sem_open(mutex_name.c_str(), O_CREAT, 0666, 1);
    if (m_mutex == SEM_FAILED) {
        throw std::runtime_error("Failed to create mutex semaphore: " + mutex_name);
    }

    m_items = sem_open(items_name.c_str(), O_CREAT, 0666, 0);
    if (m_items == SEM_FAILED) {
        sem_close(m_mutex);
        sem_unlink(mutex_name.c_str());
        throw std::runtime_error("Failed to create items semaphore: " + items_name);
    }
}

// Open existing semaphores
void SMQueue::open_semaphores() {
    ControlBlock* cb = get_control_block();

    // Open semaphores
    m_mutex = sem_open(cb->mutex_name, 0);
    if (m_mutex == SEM_FAILED) {
        throw std::runtime_error("Failed to open mutex semaphore: " + std::string(cb->mutex_name));
    }

    m_items = sem_open(cb->items_name, 0);
    if (m_items == SEM_FAILED) {
        sem_close(m_mutex);
        throw std::runtime_error("Failed to open items semaphore: " + std::string(cb->items_name));
    }
}

} // namespace shmem