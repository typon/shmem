cmake_minimum_required(VERSION 3.15)
project(shmem_queue)

# Set C++ standard and maximum optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/csrc)

# Create a library from shmem.cpp
add_library(shmem STATIC csrc/shmem.cpp)

# Add executables with maximum optimization
add_executable(publisher csrc/pub.cpp)
add_executable(subscriber csrc/sub.cpp)

# Link the library to the executables
target_link_libraries(publisher shmem)
target_link_libraries(subscriber shmem)

# Platform specific settings
if(UNIX)
    if(APPLE)
        # macOS needs pthread only
        target_link_libraries(publisher pthread)
        target_link_libraries(subscriber pthread)
        target_link_libraries(shmem pthread)
    else()
        # Linux needs both rt and pthread
        target_link_libraries(publisher rt pthread)
        target_link_libraries(subscriber rt pthread)
        target_link_libraries(shmem rt pthread)
    endif()
endif()

# Add include directories if shmem.h is in a different directory
# Uncomment and modify if needed:
# target_include_directories(publisher PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(subscriber PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Python bindings with nanobind - specify Python 3.10
find_package(Python 3.10 EXACT COMPONENTS Interpreter Development.Module REQUIRED)

# Fetch nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(nanobind)

# Add the Python module
nanobind_add_module(cyshmem STABLE_ABI csrc/bindings/cyshmem.cpp)
target_link_libraries(cyshmem PRIVATE shmem)

# Install the Python module
install(TARGETS cyshmem DESTINATION pyshmem/shmem)

# Enable Link Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if(supported)
    set_property(TARGET publisher PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET subscriber PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET shmem PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET cyshmem PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif() 