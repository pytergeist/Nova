docker run --rm -it --platform=linux/amd64 \
  -v "$PWD":/workspace -w /workspace ubuntu:22.04 \
  bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update && apt-get install -y wget gpg
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
  > /etc/apt/sources.list.d/kitware.list

apt-get update && apt-get install -y \
  cmake build-essential ninja-build git python3-dev python3-pip \
  pybind11-dev libsleef-dev libopenblas-dev

# pick a Python; in this image python3 is fine
PY=$(command -v python3)

rm -rf build

# Configure with CMake (not make!)
cmake -S . -B build/dev -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY"

# Build
cmake --build build/dev -j"$(nproc)"

'
