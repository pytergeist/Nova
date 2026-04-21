# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-src")
  file(MAKE_DIRECTORY "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-src")
endif()
file(MAKE_DIRECTORY
  "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-build"
  "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-subbuild/nanobench-populate-prefix"
  "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-subbuild/nanobench-populate-prefix/tmp"
  "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
  "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-subbuild/nanobench-populate-prefix/src"
  "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/tompope/Documents/tom_personal_devs/Nova/cmake-build-debug/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
