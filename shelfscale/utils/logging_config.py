"""
Comprehensive logging configuration for ShelfScale

Provides structured logging with different levels, file rotation,
and performance monitoring capabilities.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
import json


class ShelfScaleLogger:
    """
    Enhanced logger for ShelfScale with structured logging and performance tracking
    """
    
    def __init__(self, name: str = "shelfscale", 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_file_logging: Whether to log to files
            enable_console_logging: Whether to log to console
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory
        if enable_file_logging:
            os.makedirs(log_dir, exist_ok=True)
            
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Performance tracking
        self.performance_metrics = {}
        
    def _setup_handlers(self):
        """Setup logging handlers"""
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = self._get_console_formatter()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
        # File handlers
        if self.enable_file_logging:
            # General log file
            general_log_path = os.path.join(self.log_dir, f"{self.name}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                general_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_formatter = self._get_file_formatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Error log file
            error_log_path = os.path.join(self.log_dir, f"{self.name}_errors.log")
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            self.logger.addHandler(error_handler)
            
            # Performance log file
            perf_log_path = os.path.join(self.log_dir, f"{self.name}_performance.log")
            self.perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            self.perf_handler.setLevel(logging.INFO)
            perf_formatter = self._get_performance_formatter()
            self.perf_handler.setFormatter(perf_formatter)
            
    def _get_console_formatter(self):
        """Get console formatter with colors"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def _get_file_formatter(self):
        """Get file formatter with detailed information"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def _get_performance_formatter(self):
        """Get performance formatter for JSON structured logs"""
        return logging.Formatter('%(message)s')
        
    def get_logger(self):
        """Get the configured logger"""
        return self.logger
        
    def log_performance(self, operation: str, duration: float, 
                       details: Optional[dict] = None):
        """
        Log performance metrics in structured format
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            details: Optional additional details
        """
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'details': details or {}
        }
        
        # Store in memory for reporting
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(perf_data)
        
        # Log to file
        if self.enable_file_logging:
            perf_logger = logging.getLogger(f"{self.name}.performance")
            perf_logger.addHandler(self.perf_handler)
            perf_logger.info(json.dumps(perf_data))
            
    def get_performance_summary(self) -> dict:
        """Get summary of performance metrics"""
        summary = {}
        
        for operation, metrics in self.performance_metrics.items():
            durations = [m['duration_seconds'] for m in metrics]
            
            summary[operation] = {
                'count': len(durations),
                'total_duration': sum(durations),
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0
            }
            
        return summary
        
    def log_exception(self, exception: Exception, context: str = None):
        """
        Log exception with full traceback and context
        
        Args:
            exception: The exception to log
            context: Optional context information
        """
        import traceback
        
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Exception occurred: {json.dumps(error_data, indent=2)}")


# Performance monitoring decorator
def monitor_performance(operation_name: str = None):
    """
    Decorator to monitor function performance
    
    Args:
        operation_name: Optional name for the operation (defaults to function name)
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Get logger (assumes global logger is available)
                logger = logging.getLogger("shelfscale")
                if hasattr(logger, 'log_performance'):
                    logger.log_performance(op_name, duration, {
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'success': True
                    })
                else:
                    logger.info(f"Operation {op_name} completed in {duration:.3f}s")
                    
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger = logging.getLogger("shelfscale")
                if hasattr(logger, 'log_performance'):
                    logger.log_performance(op_name, duration, {
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'success': False,
                        'error': str(e)
                    })
                    logger.log_exception(e, f"During operation: {op_name}")
                else:
                    logger.error(f"Operation {op_name} failed after {duration:.3f}s: {e}")
                    
                raise
                
        return wrapper
    return decorator


# Global logger instance
_global_logger = None

def setup_logging(log_dir: str = "logs", 
                 log_level: str = "INFO",
                 enable_file_logging: bool = True) -> ShelfScaleLogger:
    """
    Setup global logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        enable_file_logging: Whether to enable file logging
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    _global_logger = ShelfScaleLogger(
        name="shelfscale",
        log_dir=log_dir,
        log_level=log_level,
        enable_file_logging=enable_file_logging
    )
    
    return _global_logger

def get_logger(name: str = "shelfscale") -> logging.Logger:
    """Get logger instance"""
    if _global_logger is None:
        setup_logging()
    return logging.getLogger(name)


# Error handling utilities
class ShelfScaleError(Exception):
    """Base exception for ShelfScale"""
    pass

class DataProcessingError(ShelfScaleError):
    """Error during data processing"""
    pass

class MatchingError(ShelfScaleError):
    """Error during matching process"""
    pass

class ModelError(ShelfScaleError):
    """Error related to ML models"""
    pass

class ValidationError(ShelfScaleError):
    """Error during data validation"""
    pass


def safe_execute(func, *args, default_return=None, log_errors=True, **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = get_logger()
            logger.error(f"Error executing {func.__name__}: {e}")
        return default_return