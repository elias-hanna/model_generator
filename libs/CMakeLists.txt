# Lets get a recet version of CMake
cmake_minimum_required(VERSION 3.1)

# Project Name
project(libs)

# Flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include required directories (also directories that contain only implemented template headers and no source files)
include_directories(getter
                    )