#!/usr/bin/env python3
"""
Simple test script to verify Anthropic API key is working
"""

import os
import anthropic

def test_api_key():
    """Test if the Anthropic API key is valid"""
    print("Testing Anthropic API Key")
    print("=" * 40)
    
    # Try to get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("❌ No API key found in environment variable ANTHROPIC_API_KEY")
        print("\nTo set your API key, run:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nOr update the config.py file with your API key")
        return False
    
    print(f"✅ Found API key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Anthropic client initialized successfully")
        
        # Test with a simple message
        print("🔄 Testing API with a simple message...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'Hello, API test successful!'"}]
        )
        
        print("✅ API test successful!")
        print(f"Response: {response.content[0].text}")
        return True
        
    except anthropic.AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("\nThis means your API key is invalid or expired.")
        print("Please get a new API key from: https://console.anthropic.com/")
        return False
        
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    success = test_api_key()
    if success:
        print("\n🎉 Your API key is working! You can now use the vision analyzer.")
    else:
        print("\n🔧 Please fix your API key and try again.") 