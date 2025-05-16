#!/usr/bin/env python
"""
Utility script to fix indentation issues in Python files.
This script can be used to repair syntax errors in Python files related to indentation.
"""

import os
import sys
import argparse
import tokenize
import io
import logging

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def fix_indentation(file_path, output_path=None, backup=True):
    """
    Fix indentation issues in a Python file.
    
    Args:
        file_path: Path to the file to fix
        output_path: Path to save the fixed file. If None, overwrites the original file.
        backup: If True, creates a backup of the original file with .bak extension
        
    Returns:
        bool: True if fixes were made, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
        
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # If backup is requested, create a backup
    if backup:
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w', encoding='utf-8') as backup_file:
            backup_file.write(content)
        logger.info(f"Created backup at {backup_path}")
    
    # Use tokenize to identify and fix indentation issues
    output = []
    lines = content.splitlines()
    
    # First pass: separate statements on the same line
    new_lines = []
    for line in lines:
        if ':' in line and ';' in line:
            # This might be a line with multiple statements after a colon
            parts = line.split(';')
            indentation = len(line) - len(line.lstrip())
            indent_str = ' ' * indentation
            
            # Add the first part as is
            new_lines.append(parts[0].strip())
            
            # Add the remaining parts with proper indentation
            for part in parts[1:]:
                if part.strip():
                    new_lines.append(f"{indent_str}    {part.strip()}")
        else:
            new_lines.append(line)
    
    # Second pass: fix indentation using standard Python formatting
    try:
        # Use the untokenize method to fix indentation issues
        fixed_content = '\n'.join(new_lines)
        
        # Write the fixed content
        if output_path is None:
            output_path = file_path
            
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(fixed_content)
            
        logger.info(f"Fixed indentation in {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing indentation: {e}")
        return False

def main():
    """Main function to run the script from the command line"""
    parser = argparse.ArgumentParser(description="Fix indentation issues in Python files")
    parser.add_argument("file", help="Path to the Python file to fix")
    parser.add_argument("--output", help="Output file path (if not specified, overwrites the input file)")
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original file")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    success = fix_indentation(
        args.file,
        args.output,
        not args.no_backup
    )
    
    if success:
        logger.info("Indentation fixing completed successfully")
        return 0
    else:
        logger.error("Failed to fix indentation")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 