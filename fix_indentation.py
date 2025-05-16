#!/usr/bin/env python
"""
Script to fix indentation issues in main.py
"""

import re
import os

def fix_indentation_issues(file_path):
    """
    Fixes indentation and syntax issues in the main.py file
    
    Args:
        file_path: Path to the main.py file
    """
    print(f"Fixing indentation issues in {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix McCance Widdowson function indentation
    pattern = r'def load_mccance_widdowson_data\(file_path\):(.*?)return pd\.DataFrame\(\)\s*\n'
    
    def fix_mw_function(match):
        code_block = match.group(1)
        
        # Fix code and output before print statement
        code_block = re.sub(r'(\s*)print\(f"  Loaded {len\(df\)} food items"\)\s*?\n(\s*)print\(f"  Available',
                           r'\1print(f"  Loaded {len(df)} food items")\n\1print(f"  Available', 
                           code_block)
        
        # Fix code and return indentation
        code_block = re.sub(r'(\s*)return df\s*?\n(\s*)except Exception as e:',
                           r'\1    return df\n\1except Exception as e:', 
                           code_block)
        
        return f'def load_mccance_widdowson_data(file_path):{code_block}    return pd.DataFrame()\n'
    
    # Apply the function fix
    content = re.sub(pattern, fix_mw_function, content, flags=re.DOTALL)
    
    # Fix the food portion PDF extraction section
    content = content.replace(
        'try:\n        food_portion_path', 
        'try:\n            food_portion_path'
    )
    
    # Fix the fruit veg PDF extraction section
    content = content.replace(
        'try:\n        fruit_veg_path', 
        'try:\n            fruit_veg_path'
    )
    
    # Fix matching section indentation
    content = content.replace(
        'portion_data = standardize_columns(data_sources.get("portion_data", pd.DataFrame()).copy())\n        print',
        'portion_data = standardize_columns(data_sources.get("portion_data", pd.DataFrame()).copy())\n    print'
    )
    
    content = content.replace(
        'fruit_veg_data = standardize_columns(data_sources.get("fruit_veg_data", pd.DataFrame()).copy())\n        print',
        'fruit_veg_data = standardize_columns(data_sources.get("fruit_veg_data", pd.DataFrame()).copy())\n    print'
    )
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed indentation issues in {file_path}")


if __name__ == "__main__":
    # Get the path to main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(script_dir, "shelfscale", "main.py")
    
    # Fix indentation issues
    fix_indentation_issues(main_py_path) 