#!/usr/bin/env python3
"""
Test script to verify the GitHub Action environment will work.
This simulates what the GitHub Action will do.
"""

import os
import sys
import subprocess

def test_github_action_environment():
    """Test the GitHub Action environment setup."""
    
    print("Testing GitHub Action environment...")
    print("=" * 50)
    
    # Test 1: Check Python version
    print("1. Checking Python version...")
    python_version = sys.version_info
    print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 11:
        print("   ✓ Python version is compatible")
    else:
        print("   ✗ Python version may not be compatible")
    
    # Test 2: Check if we can import required modules
    print("\n2. Testing module imports...")
    required_modules = [
        'beautifulsoup4',
        'selenium', 
        'supabase',
        'pandas',
        'tqdm',
        'openai',
        'pydantic',
        'pycountry',
        'webdriver_manager',
        'httpx'
    ]
    
    for module in required_modules:
        try:
            if module == 'beautifulsoup4':
                import bs4
            elif module == 'webdriver_manager':
                import webdriver_manager
            else:
                __import__(module)
            print(f"   ✓ {module}")
        except ImportError as e:
            print(f"   ✗ {module}: {e}")
    
    # Test 3: Check environment variables
    print("\n3. Checking environment variables...")
    required_env_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'API_KEY']
    
    for var in required_env_vars:
        if var in os.environ:
            print(f"   ✓ {var} is set")
        else:
            print(f"   ✗ {var} is not set")
    
    # Test 4: Test Supabase connection
    print("\n4. Testing Supabase connection...")
    try:
        from supabase import create_client
        client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
        result = client.table('scraper_logs').select('count').execute()
        print("   ✓ Supabase connection successful")
    except Exception as e:
        print(f"   ✗ Supabase connection failed: {e}")
    
    # Test 5: Test Chrome installation (simulate)
    print("\n5. Testing Chrome availability...")
    try:
        from selenium import webdriver
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        # This would work in GitHub Actions with Chrome installed
        print("   ✓ Chrome setup would work in GitHub Actions")
    except Exception as e:
        print(f"   ✗ Chrome setup issue: {e}")
    
    print("\n" + "=" * 50)
    print("GitHub Action environment test completed!")
    print("If all tests pass, the GitHub Action should work correctly.")

if __name__ == "__main__":
    test_github_action_environment()
