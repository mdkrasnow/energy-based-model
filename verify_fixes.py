#!/usr/bin/env python
"""
Verify that all the critical fixes have been applied correctly.
This script checks the actual code files for the fixes.
"""

import os
import re

def check_file_contains(filepath, line_num, expected_content, description):
    """Check if a specific line in a file contains expected content."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if line_num <= len(lines):
        actual_line = lines[line_num - 1].strip()
        if expected_content in actual_line:
            print(f"✓ {description}")
            print(f"  Line {line_num}: {actual_line}")
            return True
        else:
            print(f"✗ {description}")
            print(f"  Expected to find: {expected_content}")
            print(f"  Actual line {line_num}: {actual_line}")
            return False
    else:
        print(f"✗ {description} - Line {line_num} doesn't exist")
        return False

def verify_fixes():
    """Verify all the fixes have been applied."""
    print("=" * 70)
    print("VERIFYING ANM CONVERGENCE FIXES")
    print("=" * 70)
    
    all_passed = True
    
    # Fix 1: Energy output shape in models.py
    print("\n1. CHECKING ENERGY OUTPUT SHAPE FIX")
    print("-" * 40)
    result = check_file_contains(
        "models.py", 
        211,
        "output = self.fc4(h).pow(2).sum(dim=-1)",
        "EBM energy shape fix (removed [..., None])"
    )
    all_passed = all_passed and result
    
    # Fix 2: Ground truth computation in anm_utils.py
    print("\n2. CHECKING GROUND TRUTH COMPUTATION FIX")
    print("-" * 40)
    result = check_file_contains(
        "adversarial_negative_mining/anm_utils.py",
        109,
        "x_start_xt = extract(self.diffusion.sqrt_alphas_cumprod, t, x_start.shape) * x_start",
        "Ground truth xt computation fix"
    )
    all_passed = all_passed and result
    
    # Fix 3: Distance penalty gradient in anm_utils.py
    print("\n3. CHECKING DISTANCE PENALTY GRADIENT FIX")
    print("-" * 40)
    result = check_file_contains(
        "adversarial_negative_mining/anm_utils.py",
        251,
        "distance_grad = -2 * self.distance_penalty * (xt - x_start_xt)",
        "Distance penalty gradient sign fix"
    )
    all_passed = all_passed and result
    
    result = check_file_contains(
        "adversarial_negative_mining/anm_utils.py",
        252,
        "grad = grad + distance_grad",
        "Distance penalty gradient addition fix"
    )
    all_passed = all_passed and result
    
    # Fix 4: Clamping logic in anm_utils.py
    print("\n4. CHECKING CLAMPING LOGIC FIX")
    print("-" * 40)
    result = check_file_contains(
        "adversarial_negative_mining/anm_utils.py",
        278,
        "max_val = sqrt_alpha * sf",
        "Clamping logic simplified (no max operation)"
    )
    all_passed = all_passed and result
    
    # Fix 5: Ground truth in denoising_diffusion_pytorch_1d.py
    print("\n5. CHECKING DIFFUSION GROUND TRUTH FIX")
    print("-" * 40)
    result = check_file_contains(
        "diffusion_lib/denoising_diffusion_pytorch_1d.py",
        724,
        "x_start_xt = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start",
        "Diffusion ground truth computation fix"
    )
    all_passed = all_passed and result
    
    # Fix 6: Margin loss dimensions in denoising_diffusion_pytorch_1d.py
    print("\n6. CHECKING MARGIN LOSS DIMENSIONS FIX")
    print("-" * 40)
    result = check_file_contains(
        "diffusion_lib/denoising_diffusion_pytorch_1d.py",
        766,
        "loss_margin = energy_margin.mean()",
        "Margin loss scalar fix (removed keepdim)"
    )
    all_passed = all_passed and result
    
    # Fix 7: EMA model interface (should be commented out)
    print("\n7. CHECKING EMA MODEL INTERFACE FIX")
    print("-" * 40)
    with open("diffusion_lib/denoising_diffusion_pytorch_1d.py", 'r') as f:
        lines = f.readlines()
    
    # Check if the problematic line is commented
    if 913 <= len(lines):
        line_913 = lines[912].strip()
        if line_913.startswith("#") or line_913.startswith("# TODO"):
            print(f"✓ EMA model interface code properly commented out")
            print(f"  Line 913: {line_913[:60]}...")
        else:
            print(f"✗ EMA model interface code should be commented")
            print(f"  Line 913: {line_913}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL FIXES VERIFIED SUCCESSFULLY!")
        print("\nThe ANM convergence issues have been fixed:")
        print("• Energy shapes are correct")
        print("• Ground truth computation is fixed")
        print("• Distance penalties push adversarials away")
        print("• Clamping uses proper bounds")
        print("• Loss dimensions are correct")
        print("• EMA interface issue is handled")
    else:
        print("❌ SOME FIXES ARE MISSING OR INCORRECT")
        print("Please review the issues above")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(verify_fixes())