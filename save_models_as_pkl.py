#!/usr/bin/env python3
"""
Script to save models as .pkl files after training.
This ensures Streamlit can load the models properly.
"""

import sys
from pathlib import Path
import joblib

def find_project_root(start_path=None):
    """Find the MainProject directory by walking up from start_path"""
    if start_path is None:
        start_path = Path(__file__).parent.resolve()
    else:
        start_path = Path(start_path).resolve()
    
    current = start_path
    while current != current.parent:
        # Check if this directory is named "MainProject"
        if current.name == "MainProject":
            return current
        current = current.parent
    
    # If not found, return the original start_path as fallback
    return start_path

# Get the MainProject root directory
PROJECT_DIR = find_project_root()
OUTPUT_DIR = PROJECT_DIR / "ml_outputs"

def save_models_as_pkl():
    """Save models as .pkl files for Streamlit compatibility."""
    try:
        # Try to import models from the .py files
        sys.path.insert(0, str(OUTPUT_DIR))
        
        # Import models
        try:
            from rf_reg import load_model as load_rf_reg
            from xgb_reg import load_model as load_xgb_reg
            from rf_clf import load_model as load_rf_clf
            from scaler import load_model as load_scaler
            
            print("Loading models from .py files...")
            rf_reg = load_rf_reg()
            xgb_reg = load_xgb_reg()
            rf_clf = load_rf_clf()
            scaler = load_scaler()
            
            print("Saving models as .pkl files...")
            joblib.dump(rf_reg, OUTPUT_DIR / 'rf_reg.pkl')
            joblib.dump(xgb_reg, OUTPUT_DIR / 'xgb_reg.pkl')
            joblib.dump(rf_clf, OUTPUT_DIR / 'rf_clf.pkl')
            joblib.dump(scaler, OUTPUT_DIR / 'scaler.pkl')
            
            print("✅ Models saved as .pkl files successfully!")
            return True
            
        except ImportError as e:
            print(f"⚠️  Could not import models from .py files: {e}")
            print("Models may need to be trained first.")
            return False
            
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

if __name__ == "__main__":
    success = save_models_as_pkl()
    sys.exit(0 if success else 1)

