import logging
import os

def get_tensors_logger():
    logger = logging.getLogger("tensors")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    level_name = os.getenv("TENSORS_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logger.setLevel(level)
    return logger
