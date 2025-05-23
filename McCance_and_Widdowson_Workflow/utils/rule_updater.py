"""
Automated Rule Updater
Automatically updates rules based on LLM reasoning analysis
"""

import os
import re
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RuleUpdater:
    """
    Automatically updates rules based on LLM analysis insights
    """
    
    def __init__(self, rules_file: str = None):
        self.rules_file = rules_file or "shelfscale/data_processing/enhanced_rules_curator.py"
        self.rules_path = Path(self.rules_file)
        
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
    
    def update_rules_from_analysis(self, analysis_results: Dict[str, Any], 
                                 backup: bool = True) -> Dict[str, Any]:
        """
        Update rules based on LLM analysis results
        
        Args:
            analysis_results: Results from LLM reasoning analysis
            backup: Whether to create a backup before updating
            
        Returns:
            Summary of updates applied
        """
        
        logger.info("🔧 Starting automated rule updates...")
        
        # Create backup if requested
        backup_info = {}
        if backup:
            backup_info = self._create_backup()
        
        # Extract improvement suggestions
        improvements = analysis_results.get("suggested_improvements", {})
        patterns = analysis_results.get("patterns", {})
        
        # Track updates
        updates_summary = {
            "timestamp": datetime.now().isoformat(),
            "backup_info": backup_info,
            "updates_applied": {
                "new_exclude_patterns": 0,
                "new_include_patterns": 0,
                "category_improvements": 0,
                "confidence_adjustments": 0
            },
            "details": {}
        }
        
        try:
            # Read current rules file
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply updates
            content = self._update_exclude_patterns(content, improvements, updates_summary)
            content = self._update_include_patterns(content, improvements, updates_summary)
            content = self._update_category_patterns(content, patterns, updates_summary)
            content = self._add_pattern_comments(content, improvements)
            
            # Write updated content
            with open(self.rules_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Rules updated successfully")
            logger.info(f"  📝 Exclude patterns added: {updates_summary['updates_applied']['new_exclude_patterns']}")
            logger.info(f"  📝 Include patterns added: {updates_summary['updates_applied']['new_include_patterns']}")
            
            updates_summary["success"] = True
            
        except Exception as e:
            logger.error(f"❌ Failed to update rules: {str(e)}")
            updates_summary["success"] = False
            updates_summary["error"] = str(e)
            
            # Restore backup if update failed
            if backup and backup_info.get("backup_file"):
                self._restore_backup(backup_info["backup_file"])
        
        return updates_summary
    
    def _create_backup(self) -> Dict[str, Any]:
        """Create a backup of the current rules file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.rules_path.parent / f"enhanced_rules_curator_backup_{timestamp}.py"
        
        shutil.copy2(self.rules_path, backup_file)
        
        backup_info = {
            "backup_created": True,
            "backup_file": str(backup_file),
            "timestamp": timestamp,
            "original_size": os.path.getsize(self.rules_path)
        }
        
        logger.info(f"💾 Backup created: {backup_file}")
        return backup_info
    
    def _restore_backup(self, backup_file: str):
        """Restore from backup file"""
        
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, self.rules_path)
            logger.info(f"🔄 Restored from backup: {backup_file}")
        else:
            logger.error(f"❌ Backup file not found: {backup_file}")
    
    def _update_exclude_patterns(self, content: str, improvements: Dict[str, Any], 
                               updates_summary: Dict[str, Any]) -> str:
        """Update exclude patterns based on analysis"""
        
        new_exclude_patterns = improvements.get("new_exclude_patterns", [])
        if not new_exclude_patterns:
            return content
        
        # Find the database_preparations section
        database_prep_pattern = r"('database_preparations':\s*\[)(.*?)(\]\s*\})"
        match = re.search(database_prep_pattern, content, re.DOTALL)
        
        if match:
            # Get existing patterns
            existing_patterns = match.group(2)
            
            # Add new patterns that aren't already present
            new_patterns_added = []
            for pattern in new_exclude_patterns[:5]:  # Limit to top 5 patterns
                if pattern and len(pattern) > 2:  # Skip very short patterns
                    regex_pattern = f"r'\\b{re.escape(pattern)}\\b'"
                    if regex_pattern not in existing_patterns:
                        new_patterns_added.append(f"                r'\\b{re.escape(pattern)}\\b',")
            
            if new_patterns_added:
                # Add new patterns to the end of the list
                new_content = match.group(1) + match.group(2).rstrip() + "\n" + "\n".join(new_patterns_added) + "\n" + match.group(3)
                content = content.replace(match.group(0), new_content)
                
                updates_summary["updates_applied"]["new_exclude_patterns"] = len(new_patterns_added)
                updates_summary["details"]["new_exclude_patterns"] = [p.strip("r',") for p in new_patterns_added]
                
                logger.info(f"  ➕ Added {len(new_patterns_added)} new exclude patterns")
        
        return content
    
    def _update_include_patterns(self, content: str, improvements: Dict[str, Any], 
                               updates_summary: Dict[str, Any]) -> str:
        """Update include patterns based on analysis"""
        
        new_include_patterns = improvements.get("new_include_patterns", [])
        if not new_include_patterns:
            return content
        
        # Find the cooking_methods section (which was added for over-excluded items)
        cooking_methods_pattern = r"('cooking_methods':\s*\[)(.*?)(\]\s*\})"
        match = re.search(cooking_methods_pattern, content, re.DOTALL)
        
        if match:
            # Get existing patterns
            existing_patterns = match.group(2)
            
            # Add new patterns that aren't already present
            new_patterns_added = []
            for pattern in new_include_patterns[:3]:  # Limit to top 3 patterns
                if pattern and len(pattern) > 2:  # Skip very short patterns
                    regex_pattern = f"r'\\b{re.escape(pattern)}\\b'"
                    if regex_pattern not in existing_patterns:
                        new_patterns_added.append(f"                r'\\b{re.escape(pattern)}\\b',")
            
            if new_patterns_added:
                # Add new patterns to the end of the list
                new_content = match.group(1) + match.group(2).rstrip() + "\n" + "\n".join(new_patterns_added) + "\n" + match.group(3)
                content = content.replace(match.group(0), new_content)
                
                updates_summary["updates_applied"]["new_include_patterns"] = len(new_patterns_added)
                updates_summary["details"]["new_include_patterns"] = [p.strip("r',") for p in new_patterns_added]
                
                logger.info(f"  ➕ Added {len(new_patterns_added)} new include patterns")
        
        return content
    
    def _update_category_patterns(self, content: str, patterns: Dict[str, Any], 
                                updates_summary: Dict[str, Any]) -> str:
        """Update category patterns based on LLM usage"""
        
        llm_categories = patterns.get("llm_categories", {})
        if not llm_categories:
            return content
        
        # Find categories that LLM uses frequently but aren't in our rules
        frequent_categories = {k: v for k, v in llm_categories.items() if v >= 20}  # 20+ uses
        
        improvements_made = 0
        for category, count in frequent_categories.items():
            if category.lower() in ['fresh produce', 'prepared meals', 'fresh vegetables']:
                # These categories are already added in our previous update
                continue
            
            # Add simple pattern matching for this category if not present
            category_pattern = f"'{category}': ["
            if category_pattern not in content:
                # This would require more sophisticated insertion logic
                # For now, we'll just log what could be improved
                logger.info(f"  💡 Could add category '{category}' (used {count} times by LLM)")
                improvements_made += 1
        
        updates_summary["updates_applied"]["category_improvements"] = improvements_made
        return content
    
    def _add_pattern_comments(self, content: str, improvements: Dict[str, Any]) -> str:
        """Add comments explaining the new patterns"""
        
        # Add timestamp comment at the top of the _load_retail_rules method
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comment = f'        # Updated on {timestamp} based on LLM analysis\n'
        
        # Find the _load_retail_rules method
        method_pattern = r'(def _load_retail_rules\(self\):\s*""".*?""")'
        match = re.search(method_pattern, content, re.DOTALL)
        
        if match:
            new_content = match.group(0) + '\n' + comment
            content = content.replace(match.group(0), new_content)
        
        return content


def update_rules_from_file(analysis_file: str, rules_file: str = None, 
                          backup: bool = True) -> Dict[str, Any]:
    """
    Update rules from an analysis results file
    
    Args:
        analysis_file: Path to LLM reasoning analysis JSON file
        rules_file: Path to rules file (optional)
        backup: Whether to create backup
        
    Returns:
        Update summary
    """
    
    # Load analysis results
    with open(analysis_file, 'r') as f:
        analysis_results = json.load(f)
    
    # Create updater and apply updates
    updater = RuleUpdater(rules_file)
    return updater.update_rules_from_analysis(analysis_results, backup)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update rules based on LLM analysis")
    parser.add_argument("analysis_file", help="Path to LLM reasoning analysis JSON file")
    parser.add_argument("--rules-file", help="Path to rules file to update")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    
    args = parser.parse_args()
    
    try:
        results = update_rules_from_file(
            args.analysis_file,
            args.rules_file,
            backup=not args.no_backup
        )
        
        if results["success"]:
            print("✅ Rules updated successfully!")
            print(f"  📝 Exclude patterns: {results['updates_applied']['new_exclude_patterns']}")
            print(f"  📝 Include patterns: {results['updates_applied']['new_include_patterns']}")
        else:
            print("❌ Rule update failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"💥 Update failed: {str(e)}")
        raise 