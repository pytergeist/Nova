SHELL = /bin/bash

.PHONY: style

style:
	black $(TARGET)
	isort $(TARGET)
	flake8 $(TARGET)


clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -rf .coverage*
	ECHO "Successfully cleaned caches, checkpoints, and trash"

clean-logs:
	@if [ -f nova/logging/logs/error.log ] || [ -f nova/logging/logs/std.log ] || [ -f nova/logging/logs/debug.log ]; then \
		rm -f nova/logging/logs/error.log nova/logging/logs/std.log nova/logging/logs/debug.log; \
		echo "Successfully removed system log files"; \
	else \
		echo "No log files found"; \
	fi
