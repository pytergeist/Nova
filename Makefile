SHELL = /bin/bash

# ---------------- C/C++ build config ----------------
CMAKE      ?= cmake
CTEST      ?= ctest
PRESET     ?= dev
BUILD_DIR  ?= build

CFMT       := $(shell command -v clang-format 2>/dev/null)

# Find C/C++ sources while excluding the build dir
FIND_SOURCES := find . -type f \( \
	-name "*.c" -o -name "*.cc" -o -name "*.cpp" -o \
	-name "*.h" -o -name "*.hh" -o -name "*.hpp" \
\) -not -path "./$(BUILD_DIR)/*"

# ---------------- Python linting config -------------
# TARGET is your Python source root; keep using whatever you set externally
# e.g., make style TARGET=src tests
TARGET ?= .

# ---------------- Phony targets ---------------------
.PHONY: help dev asan rebuild test compdb format format-check \
        clean-build allclean style clean clean-logs

# ---------------- Help ------------------------------
help:
	@echo ""
	@echo "Targets:"
	@echo "  dev             - Fresh configure & build with preset '$(PRESET)'."
	@echo "  asan            - Fresh build with NOVA_ENABLE_ASAN=ON."
	@echo "  rebuild         - Incremental build (no reconfigure)."
	@echo "  test            - Run ctest in $(BUILD_DIR)/."
	@echo "  format          - Apply clang-format to C/C++ sources."
	@echo "  format-check    - Verify formatting (CI-friendly)."
	@echo "  compdb          - Copy compile_commands.json to project root."
	@echo "  clean-build     - Remove $(BUILD_DIR)/."
	@echo "  clean           - Clean Python caches, coverage, etc. (your original)."
	@echo "  allclean        - clean + clean-build."
	@echo "  style           - Run black, isort, flake8 on $(TARGET)."
	@echo ""

# ---------------- CMake builds ----------------------
dev:
	@echo "==> Fresh dev configure & build"
	@rm -rf "$(BUILD_DIR)"
	@$(CMAKE) --preset "$(PRESET)"
	@$(CMAKE) --build --preset "$(PRESET)" -j

cpu-profile:
	@echo "==> Fresh CPU profiling build (O2 + debug symbols + frame pointers)"
	@rm -rf "$(BUILD_DIR)"
	@$(CMAKE) --preset "$(PRESET)" \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DCMAKE_CXX_FLAGS="-O2 -g -fno-omit-frame-pointer"
	@$(CMAKE) --build --preset "$(PRESET)" -j


asan:
	@echo "==> Fresh dev (ASAN) configure & build"
	@rm -rf "$(BUILD_DIR)"
	@$(CMAKE) --preset "$(PRESET)" -D NOVA_ENABLE_ASAN=ON
	@$(CMAKE) --build --preset "$(PRESET)" -j

rebuild:
	@echo "==> Incremental build (no reconfigure)"
	@$(CMAKE) --build "$(BUILD_DIR)" -j

test:
	@echo "==> Running tests in $(BUILD_DIR)/"
	@$(CTEST) --test-dir "$(BUILD_DIR)" -j

compdb:
	@echo "==> Copying compile_commands.json to project root"
	@cp -f "$(BUILD_DIR)/compile_commands.json" ./ 2>/dev/null || \
	  (echo "Missing $(BUILD_DIR)/compile_commands.json (configure with CMake first)"; exit 1)

clean-build:
	@echo "==> Removing $(BUILD_DIR)/"
	@rm -rf "$(BUILD_DIR)"

allclean: clean clean-build

# ---------------- Formatting ------------------------
format:
ifndef CFMT
	$(error "clang-format not found in PATH. Please install clang-format.")
endif
	@echo "==> Running clang-format (in-place)"
	@$(FIND_SOURCES) -print0 | xargs -0 -I{} "$(CFMT)" -i {}

format-check:
ifndef CFMT
	$(error "clang-format not found in PATH. Please install clang-format.")
endif
	@echo "==> Checking clang-format (no changes)"
	@$(FIND_SOURCES) -print0 | xargs -0 -I{} "$(CFMT)" --dry-run --Werror {}

# ---------------- Your original Python tooling ------
style:
	black $(TARGET)
	isort $(TARGET)
	flake8 $(TARGET)

clean:
	# Safer zero-terminated deletes
	find . -type f -name "*.DS_Store" -print0 | xargs -0 -r rm -f
	find . -type d -name "__pycache__" -print0 | xargs -0 -r rm -rf
	find . -type f -name "*.py[co]" -print0 | xargs -0 -r rm -f
	find . -type d -name ".pytest_cache" -print0 | xargs -0 -r rm -rf
	find . -type d -name ".ipynb_checkpoints" -print0 | xargs -0 -r rm -rf
	find . -type d -name ".trash" -print0 | xargs -0 -r rm -rf
	rm -rf .coverage*
	@echo "Successfully cleaned caches, checkpoints, and trash"

clean-logs:
	@if [ -f nova/logging/logs/error.log ] || [ -f nova/logging/logs/std.log ] || [ -f nova/logging/logs/debug.log ]; then \
		rm -f nova/logging/logs/error.log nova/logging/logs/std.log nova/logging/logs/debug.log; \
		echo "Successfully removed system log files"; \
	else \
		echo "No log files found"; \
	fi
