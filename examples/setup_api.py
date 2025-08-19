#!/usr/bin/env python3
"""
Setup script for ShelfScale LLM-Enhanced Matching System

This script helps configure the OpenAI API key for GPT-4o-mini integration.
"""

import os
import sys
from pathlib import Path

def setup_openai_api():
    """Setup OpenAI API key configuration"""
    print("🔧 ShelfScale LLM Setup - OpenAI GPT-4o-mini Configuration")
    print("=" * 60)
    print()
    
    # Check if API key is already set
    current_key = os.getenv('OPENAI_API_KEY')
    if current_key:
        print(f"✅ OpenAI API key is already configured: {current_key[:8]}...")
        confirm = input("Do you want to update it? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Configuration unchanged.")
            return True
    
    print("To use ShelfScale's LLM-enhanced matching, you need an OpenAI API key.")
    print("GPT-4o-mini is cost-effective at ~$0.15 per 1M input tokens.")
    print()
    print("Get your API key at: https://platform.openai.com/api-keys")
    print()
    
    # Get API key from user
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key should start with 'sk-'")
        confirm = input("Continue anyway? (y/N): ").lower().strip()
        if confirm != 'y':
            return False
    
    # Create .env file
    env_file = Path('.env')
    env_content = f"OPENAI_API_KEY={api_key}\n"
    
    # If .env exists, append or update
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update existing OPENAI_API_KEY line or add new one
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('OPENAI_API_KEY='):
                lines[i] = env_content
                updated = True
                break
        
        if not updated:
            lines.append(env_content)
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
    else:
        with open(env_file, 'w') as f:
            f.write(env_content)
    
    print(f"✅ API key saved to .env file")
    print()
    print("To use the API key, either:")
    print("1. Export it in your shell: export OPENAI_API_KEY='your-key'")
    print("2. Load from .env file (some environments do this automatically)")
    print()
    print("You can now run the demo:")
    print("  python demo_llm_matching.py")
    print()
    
    return True

def test_api_connection():
    """Test the API connection"""
    try:
        import openai
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No API key found in environment")
            return False
        
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10
        )
        
        print("✅ API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install openai>=1.0.0")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_api_connection()
        return
    
    setup_success = setup_openai_api()
    
    if setup_success:
        print("Testing API connection...")
        test_api_connection()

if __name__ == "__main__":
    main()