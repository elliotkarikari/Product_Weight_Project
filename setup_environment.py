#!/usr/bin/env python3
"""
Environment Setup Script for Product Weight Project
Helps users create .env files and configure the project
"""

import os
import shutil
from pathlib import Path
import sys

def create_env_file():
    """Create .env file from template"""
    project_root = Path(__file__).parent
    template_file = project_root / "config_template.env"
    env_file = project_root / ".env"
    
    if env_file.exists():
        response = input(f".env file already exists. Overwrite? (y/N): ").strip().lower()
        if not response.startswith('y'):
            print("Keeping existing .env file.")
            return False
    
    if not template_file.exists():
        print(f"❌ Template file {template_file} not found!")
        return False
    
    # Copy template to .env
    shutil.copy2(template_file, env_file)
    print(f"✅ Created .env file at {env_file}")
    return True

def setup_openai_key():
    """Help user set up OpenAI API key"""
    print("\n🔑 OPENAI API KEY SETUP")
    print("=" * 40)
    print("To use the LLM curation features, you need an OpenAI API key.")
    print("Get one at: https://platform.openai.com/api-keys")
    print()
    
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            # Read existing .env file
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace the API key line
            lines = content.split('\n')
            updated_lines = []
            key_found = False
            
            for line in lines:
                if line.startswith('OPENAI_API_KEY='):
                    updated_lines.append(f'OPENAI_API_KEY={api_key}')
                    key_found = True
                else:
                    updated_lines.append(line)
            
            if not key_found:
                updated_lines.append(f'OPENAI_API_KEY={api_key}')
            
            # Write back to .env file
            with open(env_file, 'w') as f:
                f.write('\n'.join(updated_lines))
            
            print(f"✅ OpenAI API key saved to .env file")
        else:
            print("❌ .env file not found. Please create it first.")
    else:
        print("⏭️  Skipping OpenAI API key setup")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 CREATING DIRECTORIES")
    print("=" * 40)
    
    project_root = Path(__file__).parent
    directories = [
        "output/ai_curation",
        "output/ai_learning",
        "output/reduction_steps",
        "models",
        "Data/Processed",
        "Data/Raw Data"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 CHECKING DEPENDENCIES")
    print("=" * 40)
    
    required_packages = {
        'pandas': 'pandas>=1.3.0',
        'numpy': 'numpy>=1.21.0',
        'aiohttp': 'aiohttp>=3.8.0',
        'openpyxl': 'openpyxl>=3.0.0',
        'scikit-learn': 'scikit-learn>=1.0.0',
    }
    
    optional_packages = {
        'dotenv': 'python-dotenv>=0.19.0'
    }
    
    missing_required = []
    missing_optional = []
    
    for package, requirement in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_required.append(requirement)
    
    for package, requirement in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"⚠️  {package} - missing (optional)")
            missing_optional.append(requirement)
    
    if missing_required or missing_optional:
        print(f"\n📋 INSTALLATION COMMANDS")
        print("=" * 40)
        
        if missing_required:
            print("Required packages:")
            print(f"pip install {' '.join(missing_required)}")
        
        if missing_optional:
            print("Optional packages (recommended):")
            print(f"pip install {' '.join(missing_optional)}")
        
        print("\nOr install all at once:")
        print("pip install -r requirements_llm.txt")
    
    return len(missing_required) == 0

def print_next_steps():
    """Print next steps for the user"""
    print("\n🚀 NEXT STEPS")
    print("=" * 40)
    print("1. Edit your .env file to add your API keys:")
    print("   - Set OPENAI_API_KEY for LLM features")
    print("   - Adjust other settings as needed")
    print()
    print("2. Install any missing dependencies:")
    print("   pip install -r requirements_llm.txt")
    print()
    print("3. Test the LLM curator:")
    print("   python shelfscale/data_processing/llm_retail_curator.py")
    print()
    print("4. Review the configuration in your .env file:")
    print("   - LLM_DEMO_MODE=true (limits to 50 items for testing)")
    print("   - MAX_ESTIMATED_COST=5.00 (safety limit)")
    print("   - Adjust batch sizes and delays as needed")

def main():
    """Main setup function"""
    print("🔧 PRODUCT WEIGHT PROJECT ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Step 1: Create .env file
    print("\n1. Setting up .env file...")
    env_created = create_env_file()
    
    # Step 2: Setup OpenAI key
    if env_created or Path(__file__).parent.joinpath(".env").exists():
        setup_openai_key()
    
    # Step 3: Create directories
    setup_directories()
    
    # Step 4: Check dependencies
    deps_ok = check_dependencies()
    
    # Step 5: Print next steps
    print_next_steps()
    
    if deps_ok:
        print("\n✅ Setup complete! Your environment is ready.")
    else:
        print("\n⚠️  Setup complete, but some dependencies are missing.")
        print("Please install the missing packages before running the project.")

if __name__ == "__main__":
    main() 