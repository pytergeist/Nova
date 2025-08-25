# macOS universal? You were arm64-only, keep it explicit:
if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# You can add toggles here if you like
option(BUILD_BENCHMARKS "Build Fusion benchmarks" ON)

# Default to Debug if nothing set
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
