cmake_minimum_required(VERSION 3.10)
project(DMPLibrary VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Include directories - updated for your actual structure
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/eigen
)

# Collect all source files from src/dmplib/
file(GLOB DMP_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Debug: print what sources were found
message(STATUS "Found DMP sources: ${DMP_SOURCES}")

# Create a library
add_library(dmplib STATIC ${DMP_SOURCES})

# Set include directories for the library
target_include_directories(dmplib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/eigen
)

# Create executable
add_executable(dmp_app main.cpp)

# Link the library to the executable
target_link_libraries(dmp_app dmplib)