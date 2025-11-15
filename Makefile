SHELL = /bin/bash

# ---------------- C/C++ build config ----------------
CMAKE      ?= cmake
CTEST      ?= ctest
PRESET     ?= dev

BUILD_ROOT ?= build
BUILD_DIR  ?= $(BUILD_ROOT)/$(PRESET)

# Where we expect the compile DB
COMPILE_DB_BUILD = $(BUILD_DIR)/compile_commands.json
COMPILE_DB_ROOT  = ./compile_commands.json

# Always ask CMake to export a compile DB when configuring
CMAKE_CONFIGURE_FLAGS ?= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ---------------- Python linting config -------------
TARGET ?= .

# ---------------- clang-format ----------------------
CFMT := $(shell command -v clang-format 2>/dev/null)
FIND_SOURCES := find . -type f \( \
	-name "*.c" -o -name "*.cc" -o -name "*.cpp" -o \
	-name "*.h" -o -name "*.hh" -o -name "*.hpp" \
\) -not -path "./$(BUILD_ROOT)/*"

# ---------------- clang-tidy config -----------------
CLANG_TIDY     ?= /opt/homebrew/opt/llvm/bin/clang-tidy
RUN_CLANG_TIDY ?= /opt/homebrew/opt/llvm/bin/run-clang-tidy
HEADER_FILTER  ?= '^(./)?fusion/src/.*'
TIDY_JOBS      ?= 8
TIDY_ARGS      ?=

# For header-only runs (fallback when not using the DB)
SDK_PATH     := $(shell xcrun --show-sdk-path 2>/dev/null)
SYSROOT_FLAG := $(if $(SDK_PATH),-isysroot $(SDK_PATH),)
CXXSTD       ?= -std=c++20
HDR_IFLAGS   ?= -I. -Ifusion/src $(SYSROOT_FLAG)

# Helper: choose DB dir automatically: prefer build/<preset>, else root
define _pick_db_dir
sh -c ' \
  if [ -f "$(COMPILE_DB_BUILD)" ]; then echo "$(BUILD_DIR)"; \
  elif [ -f "$(COMPILE_DB_ROOT)" ]; then echo "."; \
  else echo ""; fi'
endef

# ---------------- Phony targets ---------------------
.PHONY: help dev cpu-profile asan release rebuild test \
        compdb compdb-link \
        tidy tidy-fix tidy-file tidy-header tidy-changed \
        format format-check style \
        clean clean-build allclean clean-logs

# ---------------- Help ------------------------------
help:
	@echo ""
	@echo "Targets:"
	@echo "  dev           - Fresh configure & build with preset '$(PRESET)' (exports compile DB)."
	@echo "  cpu-profile   - RelWithDebInfo + frame pointers (also exports DB if generated)."
	@echo "  asan          - Fresh build with NOVA_ENABLE_ASAN=ON."
	@echo "  release       - Optimized release build."
	@echo "  rebuild       - Incremental build (no reconfigure)."
	@echo "  test          - Run ctest in $(BUILD_DIR)/."
	@echo "  tidy          - run-clang-tidy across the DB."
	@echo "  tidy-fix      - run-clang-tidy with -fix and format."
	@echo "  tidy-file     - clang-tidy one TU (FILE=path/to/file.cpp)."
	@echo "  tidy-header   - best-effort on a header (FILE=path/to/file.h)."
	@echo "  tidy-changed  - clang-tidy changed files vs origin/main."
	@echo "  compdb        - copy compile DB from current build dir to root (if present)."
	@echo "  compdb-link   - symlink compile DB from current build dir to root (if present)."
	@echo "  format        - clang-format all sources."
	@echo "  format-check  - verify formatting."
	@echo "  clean-build   - rm -rf $(BUILD_ROOT)/."
	@echo "  clean         - remove python caches, coverage, etc."
	@echo "  allclean      - clean + clean-build."
	@echo ""

# ---------------- CMake builds ----------------------
dev:
	@echo "==> Fresh dev configure & build"
	@rm -rf "$(BUILD_DIR)"
	@$(CMAKE) --preset "$(PRESET)" $(CMAKE_CONFIGURE_FLAGS)
	@$(CMAKE) --build --preset "$(PRESET)" -j
	@if [ -f "$(COMPILE_DB_BUILD)" ]; then \
		ln -sf "$(COMPILE_DB_BUILD)" "$(COMPILE_DB_ROOT)"; \
		echo "Symlinked $(COMPILE_DB_BUILD) -> $(COMPILE_DB_ROOT)"; \
	else \
		echo "Note: No $(COMPILE_DB_BUILD). If you already have $(COMPILE_DB_ROOT), tidy will use that."; \
	fi


cli:
	@echo "==> Fresh dev configure & build"
	@rm -rf "$(BUILD_DIR)"
	@$(CMAKE) --preset "$(PRESET)" $(CMAKE_CONFIGURE_FLAGS) \
		-D NOVA_BUILD_CLI=ON
	@$(CMAKE) --build --preset "$(PRESET)" -j
	@if [ -f "$(COMPILE_DB_BUILD)" ]; then \
		ln -sf "$(COMPILE_DB_BUILD)" "$(COMPILE_DB_ROOT)"; \
		echo "Symlinked $(COMPILE_DB_BUILD) -> $(COMPILE_DB_ROOT)"; \
	else \
		echo "Note: No $(COMPILE_DB_BUILD). If you already have $(COMPILE_DB_ROOT), tidy will use that."; \
	fi


cpu-profile:
	@echo "==> Fresh CPU profiling build (O2 + debug symbols + frame pointers)"
	@rm -rf "$(BUILD_DIR)"
	@$(CMAKE) --preset "$(PRESET)" $(CMAKE_CONFIGURE_FLAGS) \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DCMAKE_CXX_FLAGS="-O2 -g -fno-omit-frame-pointer"
	@$(CMAKE) --build --preset "$(PRESET)" -j
	@if [ -f "$(COMPILE_DB_BUILD)" ]; then \
		ln -sf "$(COMPILE_DB_BUILD)" "$(COMPILE_DB_ROOT)"; \
		echo "Symlinked $(COMPILE_DB_BUILD) -> $(COMPILE_DB_ROOT)"; \
	fi

asan:
	@echo "==> Fresh dev (ASAN) configure & build"
	@rm -rf "$(BUILD_DIR)"
	@$(CMAKE) --preset "$(PRESET)" $(CMAKE_CONFIGURE_FLAGS) -D NOVA_ENABLE_ASAN=ON -D NOVA_BUILD_CLI=ON
	@$(CMAKE) --build --preset "$(PRESET)" -j
	@if [ -f "$(COMPILE_DB_BUILD)" ]; then \
		ln -sf "$(COMPILE_DB_BUILD)" "$(COMPILE_DB_ROOT)"; \
		echo "Symlinked $(COMPILE_DB_BUILD) -> $(COMPILE_DB_ROOT)"; \
	fi

release:
	@echo "==> Fresh release configure & build"
	@rm -rf "$(BUILD_ROOT)/release"
	@$(CMAKE) --preset "release" $(CMAKE_CONFIGURE_FLAGS) \
		-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
		-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -ffast-math -fno-math-errno -fno-trapping-math -fstrict-aliasing -funroll-loops"
	@$(CMAKE) --build --preset "release" -j
	@if [ -f "$(BUILD_ROOT)/release/compile_commands.json" ]; then \
		ln -sf "$(BUILD_ROOT)/release/compile_commands.json" "$(COMPILE_DB_ROOT)"; \
		echo "Symlinked $(BUILD_ROOT)/release/compile_commands.json -> $(COMPILE_DB_ROOT)"; \
	fi

rebuild:
	@echo "==> Incremental build (no reconfigure)"
	@$(CMAKE) --build "$(BUILD_DIR)" -j

test:
	@echo "==> Running tests in $(BUILD_DIR)/"
	@$(CTEST) --test-dir "$(BUILD_DIR)" -j

# ---------------- Compile DB helpers ----------------
compdb:
	@echo "==> Copying compile_commands.json to project root (if build DB exists)"
	@if [ -f "$(COMPILE_DB_BUILD)" ]; then \
		cp -f "$(COMPILE_DB_BUILD)" ./; \
		echo "Copied from $(COMPILE_DB_BUILD)"; \
	else \
		echo "No build DB at $(COMPILE_DB_BUILD). Skipping copy."; \
	fi

compdb-link:
	@echo "==> Symlinking compile_commands.json to project root (if build DB exists)"
	@if [ -f "$(COMPILE_DB_BUILD)" ]; then \
		ln -sf "$(COMPILE_DB_BUILD)" ./compile_commands.json; \
		echo "Symlinked from $(COMPILE_DB_BUILD)"; \
	else \
		echo "No build DB at $(COMPILE_DB_BUILD). Skipping symlink."; \
	fi

# ---------------- clang-tidy ------------------------
tidy:
	@echo "==> run-clang-tidy (all)"
	@DB_DIR=$$($(call _pick_db_dir)); \
	if [ -z "$$DB_DIR" ]; then echo "No compile_commands.json found (looked in $(COMPILE_DB_BUILD) then $(COMPILE_DB_ROOT))"; exit 1; fi; \
	$(RUN_CLANG_TIDY) -p "$$DB_DIR" -header-filter=$(HEADER_FILTER) -j $(TIDY_JOBS) -- $(TIDY_ARGS)

tidy-fix:
	@echo "==> run-clang-tidy (fix)"
	@DB_DIR=$$($(call _pick_db_dir)); \
	if [ -z "$$DB_DIR" ]; then echo "No compile_commands.json found"; exit 1; fi; \
	$(RUN_CLANG_TIDY) -p "$$DB_DIR" -header-filter=$(HEADER_FILTER) -j $(TIDY_JOBS) -fix -format -format-style=file -- $(TIDY_ARGS)

tidy-file:
ifndef FILE
	$(error "Usage: make tidy-file FILE=path/to/file.cpp")
endif
	@echo "==> clang-tidy $(FILE)"
	@DB_DIR=$$($(call _pick_db_dir)); \
	if [ -z "$$DB_DIR" ]; then echo "No compile_commands.json found"; exit 1; fi; \
	$(CLANG_TIDY) -p "$$DB_DIR" $(FILE) -- $(TIDY_ARGS)

tidy-header:
ifndef FILE
	$(error "Usage: make tidy-header FILE=path/to/header.h")
endif
	@echo "==> clang-tidy (header-only) $(FILE)"
	@$(CLANG_TIDY) $(FILE) -- -xc++ -std=c++20 -I. -Ifusion/src $(SYSROOT_FLAG)

tidy-changed:
	@echo "==> run-clang-tidy (changed files)"
	@DB_DIR=$$($(call _pick_db_dir)); \
	if [ -z "$$DB_DIR" ]; then echo "No compile_commands.json found"; exit 1; fi; \
	git diff --name-only --diff-filter=ACMR origin/main...HEAD | \
		grep -E '\.(c|cc|cpp|cxx|h|hh|hpp)$$' | \
		xargs -r $(RUN_CLANG_TIDY) -p "$$DB_DIR" -header-filter=$(HEADER_FILTER) -j $(TIDY_JOBS) -- $(TIDY_ARGS) --

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

# ---------------- Python tooling --------------------
style:
	black $(TARGET)
	isort $(TARGET)
	flake8 $(TARGET)

clean:
	# Safer zero-terminated deletes
	find . -type f -name "*.DS_Store" -print0 | xargs -0 -r rm -f
	find . -type d -name "__pycache__" -print0 | xargs -0 -r rm -rf
	find . -type f -name "*.py[co]" -print0 | xargs -0 -r rm -f
	find . -type d -name ".pytest_cache" -print0 -r -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -print0 -r -exec rm -rf {} +
	find . -type d -name ".trash" -print0 -r -exec rm -rf {} +
	rm -rf .coverage*
	@echo "Successfully cleaned caches, checkpoints, and trash"

clean-logs:
	@if [ -f nova/logging/logs/error.log ] || [ -f nova/logging/logs/std.log ] || [ -f nova/logging/logs/debug.log ]; then \
		rm -f nova/logging/logs/error.log nova/logging/logs/std.log nova/logging/logs/debug.log; \
		echo "Successfully removed system log files"; \
	else \
		echo "No log files found"; \
	fi

clean-build:
	@echo "==> Removing $(BUILD_ROOT)/"
	@rm -rf "$(BUILD_ROOT)"

allclean: clean clean-build
