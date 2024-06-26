cmake_minimum_required(VERSION 3.10)

project(autograd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${CMAKE_SOURCE_DIR}/autograd)
add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
