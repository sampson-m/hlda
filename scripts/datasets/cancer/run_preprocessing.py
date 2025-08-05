#!/usr/bin/env python3
"""
Helper script to run preprocessing for both cancer labeling schemes.
"""

import subprocess
import sys
from pathlib import Path

def run_preprocessing():
    """Run preprocessing for both labeling schemes."""
    
    # Get the preprocessing script path
    preprocess_script = Path(__file__).parent / "preprocess_cancer.py"
    
    print("Running cancer dataset preprocessing for both labeling schemes...")
    print("="*60)
    
    # Run combined scheme
    print("\n1. Running COMBINED labeling scheme...")
    print("-" * 40)
    try:
        result = subprocess.run([
            sys.executable, str(preprocess_script),
            "--labeling_scheme", "combined"
        ], check=True, capture_output=True, text=True)
        print("✓ Combined scheme preprocessing completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ Combined scheme preprocessing failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    
    # Run disease-specific scheme
    print("\n2. Running DISEASE-SPECIFIC labeling scheme...")
    print("-" * 40)
    try:
        result = subprocess.run([
            sys.executable, str(preprocess_script),
            "--labeling_scheme", "disease_specific"
        ], check=True, capture_output=True, text=True)
        print("✓ Disease-specific scheme preprocessing completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ Disease-specific scheme preprocessing failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    
    print("\n" + "="*60)
    print("✓ All preprocessing completed successfully!")
    print("You can now run the cancer pipeline with:")
    print("python scripts/cancer/run_cancer_pipeline.py")
    
    return True

if __name__ == "__main__":
    success = run_preprocessing()
    if not success:
        sys.exit(1)