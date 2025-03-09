#!/bin/bash

# Format all C++ files in the project
find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i -style=file

echo "All C++ files have been formatted according to .clang-format" 