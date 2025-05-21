import logging
import os

logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

debug_log_path = os.path.join(logs_dir, "debug.log")
info_log_path = os.path.join(logs_dir, "std.log")
error_log_path = os.path.join(logs_dir, "error.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

debug_handler = logging.FileHandler(debug_log_path)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(debug_handler)

info_handler = logging.FileHandler(info_log_path)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(info_handler)

error_handler = logging.FileHandler(error_log_path)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(error_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(console_handler)
