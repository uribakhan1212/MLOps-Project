#!/usr/bin/env python3
"""
Create basic unit tests for the project
"""
import os

def create_basic_tests():
    """Create basic unit test files"""
    
    # Ensure test directory exists
    os.makedirs('tests/unit', exist_ok=True)
    
    # Create basic model test
    test_model_content = '''import sys
sys.path.insert(0, '.')

def test_model_import():
    """Test that model can be imported"""
    from src.model import DiabetesModel
    assert DiabetesModel is not None

def test_model_creation():
    """Test model creation"""
    from src.model import DiabetesModel
    model = DiabetesModel.create_model(input_dim=21)
    assert model is not None
    assert len(model.layers) > 0

def test_model_compilation():
    """Test model compilation"""
    from src.model import DiabetesModel
    model = DiabetesModel.create_model(input_dim=21)
    # Model should be compiled after creation
    assert model.optimizer is not None
'''
    
    with open('tests/unit/test_model.py', 'w') as f:
        f.write(test_model_content)
    
    print("✅ Basic unit tests created")

if __name__ == '__main__':
    create_basic_tests()