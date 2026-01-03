include(FetchContent)

# macOS/Homebrew path
set(CMAKE_PREFIX_PATH "/opt/homebrew" CACHE STRING "Prefix path for dependencies")

# ---------- Python / pybind11 ----------
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Force pybind11 to use FindPython")
find_package(pybind11 REQUIRED)

# ---------- OpenBLAS ----------
find_library(OPENBLAS_LIB
  NAMES openblas
  PATHS /opt/homebrew/opt/openblas/lib /usr/lib /usr/local/lib
)
find_path(OPENBLAS_INCLUDE_DIR
  NAMES cblas.h openblas_config.h
  PATHS /usr/include /usr/local/include /opt/homebrew/opt/openblas/include
)
if(NOT OPENBLAS_LIB OR NOT OPENBLAS_INCLUDE_DIR)
  message(FATAL_ERROR "OpenBLAS not found (lib: ${OPENBLAS_LIB}, include: ${OPENBLAS_INCLUDE_DIR})")
endif()

# ---------- Eigen3 ----------
find_package(Eigen3 REQUIRED)

# ---------- Tests / GoogleTest ----------
include(CTest)
if(BUILD_TESTING)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(STATUS "GTest not found; fetching via FetchContent")
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

# ---------- SLEEF (prefer installed; else fetch) ----------
find_package(SLEEF CONFIG QUIET)
set(SLEEF_INCLUDE_DIRS "")
set(SLEEF_TARGET sleef::sleef)

if(NOT SLEEF_FOUND)
  message(STATUS "SLEEF not found; fetching via FetchContent")
  FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG        3.9.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
  )
  set(BUILD_SHARED_LIBS  ON  CACHE BOOL "" FORCE)
  set(SLEEF_BUILD_DFT    OFF CACHE BOOL "" FORCE)
  set(SLEEF_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
  set(SLEEF_BUILD_QUAD   OFF CACHE BOOL "" FORCE)
  set(SLEEF_DISABLE_GFNI ON  CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(sleef)

  set(SLEEF_INCLUDE_DIRS
    ${sleef_SOURCE_DIR}/include
    ${sleef_BINARY_DIR}/include
  )
  if(NOT TARGET sleef::sleef)
    add_library(sleef::sleef ALIAS sleef)
  endif()
  set(SLEEF_TARGET sleef)

  include_directories(BEFORE SYSTEM ${SLEEF_INCLUDE_DIRS})
else()
  message(STATUS "Using installed SLEEF package.")
endif()
