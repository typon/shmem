#include <chrono>
#include <cstdio>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "shmem.h"

namespace nb = nanobind;

// Helper function for timing
inline double get_time_ms() {
    using namespace std::chrono;
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
}

NB_MODULE(cyshmem, m) {
    // Module docstring
    m.doc() = "Python bindings for shmem library - a shared memory queue implementation";

    // Define the SMQueue class
    nb::class_<shmem::SMQueue>(m, "SMQueue")
        .def_static("create", &shmem::SMQueue::create, "Create a new shared memory queue", nb::arg("name"),
                    nb::arg("max_elements"), nb::arg("element_size"))
        .def_static("open", &shmem::SMQueue::open, "Open an existing shared memory queue", nb::arg("name"))
        .def_static("destroy", &shmem::SMQueue::destroy, "Destroy a shared memory queue", nb::arg("name"))
        .def("close", &shmem::SMQueue::close, "Close the queue")
        .def("max_elements", &shmem::SMQueue::max_elements, "Get maximum number of elements")
        .def("element_size", &shmem::SMQueue::element_size, "Get element size in bytes")
        .def("name", &shmem::SMQueue::name, "Get queue name")
        // Custom implementation for push that accepts generic arrays
        .def(
            "push",
            [](shmem::SMQueue& self, nb::ndarray<> array) {
                if (array.size() != self.element_size()) {
                    throw std::runtime_error("Array size does not match element size");
                }

                // Directly use the data from the NumPy array
                bool result = self.push(reinterpret_cast<const std::byte*>(array.data()));

                return result;
            },
            "Push a message to the queue as an array", nb::arg("array"))
        // Custom implementation for pop that returns generic arrays or None
        .def(
            "pop_np",
            [](shmem::SMQueue& self) -> std::optional<nb::ndarray<nb::numpy, uint8_t>> {
                size_t size = self.element_size();
                // Allocate memory for the array
                uint8_t* data = new uint8_t[size];

                // Pop directly into the allocated memory
                bool success = self.pop(reinterpret_cast<std::byte*>(data));

                if (!success) {
                    // Clean up allocated memory if pop failed
                    delete[] data;
                    return std::nullopt;
                }

                // Create shape array
                std::vector<size_t> shape = {size};
                // Create a capsule to manage the memory
                nb::capsule deleter(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });

                auto result = nb::ndarray<nb::numpy, uint8_t>(data, shape.size(), shape.data(), deleter, nullptr,
                                                              nb::dtype<uint8_t>(), nb::device::cpu::value);

                return result;
            },
            "Pop a message from the queue (blocking) as an array")
        // Custom implementation for try_pop that returns generic arrays or None
        .def(
            "try_pop_np",
            [](shmem::SMQueue& self) -> std::optional<nb::ndarray<nb::numpy, uint8_t>> {
                size_t size = self.element_size();

                // Allocate memory for the array
                uint8_t* data = new uint8_t[size];

                // Try to pop directly into the allocated memory
                bool success = self.try_pop(reinterpret_cast<std::byte*>(data));

                if (!success) {
                    // Clean up allocated memory if pop failed
                    delete[] data;

                    return std::nullopt;
                }

                // Create shape array
                std::vector<size_t> shape = {size};
                // Create a capsule to manage the memory
                nb::capsule deleter(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });

                // Create the ndarray with the allocated memory and the capsule as owner
                auto result = nb::ndarray<nb::numpy, uint8_t>(data, shape.size(), shape.data(), deleter, nullptr,
                                                              nb::dtype<uint8_t>(), nb::device::cpu::value);

                return result;
            },
            "Try to pop a message (non-blocking) as an array");
}