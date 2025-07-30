#!/usr/bin/env python3
"""
Debug script to test validation functions
"""

from simple_rag import ValidationUtils, ValidationError

def test_validation():
    print("Testing ValidationUtils.validate_query...")
    
    # Test None
    try:
        ValidationUtils.validate_query(None)
        print("ERROR: None should have raised ValidationError")
    except ValidationError as e:
        print(f"✓ None correctly raised: {e}")
    
    # Test empty string
    try:
        ValidationUtils.validate_query("")
        print("ERROR: Empty string should have raised ValidationError")
    except ValidationError as e:
        print(f"✓ Empty string correctly raised: {e}")
    
    # Test whitespace only
    try:
        ValidationUtils.validate_query("   ")
        print("ERROR: Whitespace should have raised ValidationError")
    except ValidationError as e:
        print(f"✓ Whitespace correctly raised: {e}")
    
    # Test valid query
    try:
        result = ValidationUtils.validate_query("What is the revenue?")
        print(f"✓ Valid query returned: '{result}'")
    except Exception as e:
        print(f"ERROR: Valid query failed: {e}")

if __name__ == "__main__":
    test_validation()