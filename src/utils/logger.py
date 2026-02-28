"""
Comprehensive structured logging for py_ntcpx_v1.0.0
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "py_ntcpx",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up structured logger for py_ntcpx pipeline
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : Path, optional
        Path to log file. If None, logs only to console.
    level : int
        Logging level (default: INFO)
    format_string : str, optional
        Custom format string. If None, uses default structured format.
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format: timestamp, level, module, message
    if format_string is None:
        format_string = (
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "py_ntcpx") -> logging.Logger:
    """
    Get existing logger or create new one with default settings
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up default
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


# Context manager for logging to file
class LoggingContext:
    """Context manager for temporary logging configuration"""
    
    def __init__(self, log_file: Path, level: int = logging.INFO):
        self.log_file = Path(log_file)
        self.level = level
        self.original_handlers = None
        self.logger = None
    
    def __enter__(self):
        self.logger = get_logger()
        self.original_handlers = self.logger.handlers.copy()
        
        # Add file handler
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.level)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove file handler
        if self.logger:
            self.logger.handlers = self.original_handlers
        return False
