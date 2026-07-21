cmake_minimum_required (VERSION 3.5)
project (Transformation)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable (Transformation main.cpp)
