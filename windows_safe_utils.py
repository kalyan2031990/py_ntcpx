#!/usr/bin/env python3
"""
Windows-Safe Utilities for py_ntcpx v1.0
=========================================

Provides safe console output and encoding utilities for Windows compatibility.
All Unicode characters are replaced with ASCII-safe equivalents.

Software: py_ntcpx v1.0
"""

import sys
import os


def configure_utf8_encoding():
    """Configure UTF-8 encoding for stdout/stderr (Windows-safe)"""
    try:
        if sys.platform == 'win32':
            # Windows: reconfigure to UTF-8 with error handling
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        else:
            # Unix/Linux/macOS: set encoding if possible
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # Fallback: continue without reconfiguration
        pass


def safe_encode_unicode(text):
    """
    Replace Unicode characters with ASCII-safe equivalents
    
    Args:
        text: String that may contain Unicode characters
        
    Returns:
        ASCII-safe string
    """
    if not isinstance(text, str):
        return str(text)
    
    replacements = {
        '✓': '[OK]',
        '✗': '[FAIL]',
        '✔': '[PASS]',
        '❌': '[ERROR]',
        '→': '->',
        '↳': '->',
        'Δ': 'DELTA',
        '±': '+/-',
        'μ': 'mu',
        'σ': 'sigma',
        'Σ': 'SUM',
        '≥': '>=',
        '≤': '<=',
        '½': '1/2',
        '¼': '1/4',
        '¾': '3/4',
        'α': 'alpha',
        'β': 'beta',
        'θ': 'theta',
        'γ': 'gamma',
        'π': 'pi'
    }
    
    result = text
    for unicode_char, ascii_replacement in replacements.items():
        result = result.replace(unicode_char, ascii_replacement)
    
    return result


def safe_print(*args, **kwargs):
    """
    Safe print function that handles Unicode encoding errors
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print()
    """
    try:
        # Convert all arguments to safe ASCII
        safe_args = [safe_encode_unicode(str(arg)) for arg in args]
        print(*safe_args, **kwargs)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Fallback: encode with error replacement
        try:
            safe_args = [str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args]
            print(*safe_args, **kwargs)
        except Exception:
            # Last resort: print as bytes
            print(repr(args), **kwargs)


def safe_log(log_func, message, *args, **kwargs):
    """
    Safe logging function that handles Unicode encoding errors
    
    Args:
        log_func: Logging function (logger.info, logger.error, etc.)
        message: Log message (may contain format placeholders)
        *args: Format arguments
        **kwargs: Additional keyword arguments
    """
    try:
        # Encode message and format args
        safe_message = safe_encode_unicode(message)
        if args:
            safe_args = [safe_encode_unicode(str(arg)) for arg in args]
            log_func(safe_message % tuple(safe_args), **kwargs)
        else:
            log_func(safe_message, **kwargs)
    except (UnicodeEncodeError, UnicodeDecodeError, TypeError):
        # Fallback: encode with error replacement
        try:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            log_func(safe_message, **kwargs)
        except Exception:
            # Last resort: log as repr
            log_func(repr(message), **kwargs)

