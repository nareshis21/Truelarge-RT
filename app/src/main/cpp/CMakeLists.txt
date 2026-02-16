cmake_minimum_required(VERSION 3.22.1)
project(airllm_native_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3 -ffast-math -fno-finite-math-only)

# Directory pointing to llama.cpp source (now in the same directory)
set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# Build llama.cpp as a subdirectory — this creates the 'llama' and 'ggml' targets
# Disable things we don't need to speed up the build
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

# Include directories for our engine code
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${LLAMA_CPP_DIR}/include)
include_directories(${LLAMA_CPP_DIR}/ggml/include)

# Source files for our engine
set(ENGINE_SOURCES
    GGUFHeaderParser.cpp
    LayerLoader.cpp
    LayerScheduler.cpp
    WeightBuffer.cpp
    TrueLargeRuntime.cpp
    native-lib.cpp
)

# Create a library for the engine
add_library(truelarge-rt SHARED ${ENGINE_SOURCES})

# We need to link against log library for Android logging
find_library(log-lib log)

# Link our engine against llama (which includes ggml) and Android log
target_link_libraries(truelarge-rt llama ${log-lib})
