#!/usr/bin/env python3
"""
Test script for the FastAPI functionality
"""

def test_api_imports():
    """Test that the API can be imported and basic functionality works"""
    print("🌐 Testing FastAPI Components")
    print("=" * 50)
    
    try:
        # Test basic imports
        import sys
        sys.path.insert(0, '.')
        
        print("📦 Testing imports...")
        from shelfscale.api import app
        print("✅ FastAPI app imported successfully")
        
        # Test pydantic models
        from shelfscale.api import WeightExtractRequest, ScoreRequest
        print("✅ Pydantic models imported successfully") 
        
        # Test basic request models
        weight_request = WeightExtractRequest(
            items=["500ml milk", "250g bread"]
        )
        print("✅ WeightExtractRequest model works")
        
        score_request = ScoreRequest(
            items=[{
                "name": "Apple",
                "energy_kcal": 52,
                "fat_g": 0.2,
                "saturated_fat_g": 0.1,
                "sugars_g": 10.4,
                "salt_g": 0.0
            }]
        )
        print("✅ ScoreRequest model works")
        
        print("\n💡 API is ready! To test the full server:")
        print("   1. Install uvicorn: pip install uvicorn")
        print("   2. Run: uvicorn shelfscale.api:app --host 0.0.0.0 --port 8000")
        print("   3. Visit: http://localhost:8000/docs for interactive API docs")
        print("   4. Test endpoints like POST /products/score")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 You may need to install FastAPI and dependencies:")
        print("   pip install fastapi uvicorn pydantic")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_api_imports()