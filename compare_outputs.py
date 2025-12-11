#!/usr/bin/env python3
"""
Compare vision encoder outputs between llama.cpp and PyTorch

This script automatically finds and compares matching tensor files.

Usage:
    python compare_outputs.py [--atol TOLERANCE]

The script looks for:
    - debug_cpp_*.npy (from llama.cpp with DEBUG_DUMP_TENSOR)
    - debug_activation_*.npy (from debug_gemma3n.py)
    - debug_vision_output.npy (final PyTorch output)
"""

import numpy as np
import glob
import argparse
import sys
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colorize(text, color):
    return f"{color}{text}{Colors.ENDC}"

def print_tensor_info(tensor, label):
    """Print detailed information about a tensor"""
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Range: [{tensor.min():+.6f}, {tensor.max():+.6f}]")
    print(f"  Mean:  {tensor.mean():+.6f}")
    print(f"  Std:   {tensor.std():.6f}")

    # Check for problematic values
    if np.any(np.isnan(tensor)):
        print(colorize(f"  ⚠️  Contains NaN values: {np.isnan(tensor).sum()}", Colors.WARNING))
    if np.any(np.isinf(tensor)):
        print(colorize(f"  ⚠️  Contains Inf values: {np.isinf(tensor).sum()}", Colors.WARNING))
    if np.allclose(tensor, 0):
        print(colorize(f"  ⚠️  All values are zero!", Colors.WARNING))

def compare_tensors(cpp_tensor, py_tensor, name, atol=1e-4, rtol=1e-5):
    """Compare two tensors and return detailed statistics"""

    print(f"\n{'='*80}")
    print(colorize(f"Comparing: {name}", Colors.BOLD))
    print('='*80)

    # Shape comparison
    print(f"\n📐 Shape Check:")
    if cpp_tensor.shape != py_tensor.shape:
        print(colorize(f"  ❌ MISMATCH", Colors.FAIL))
        print(f"     C++ shape:    {cpp_tensor.shape}")
        print(f"     Python shape: {py_tensor.shape}")

        # Try to suggest reshape if possible
        if np.prod(cpp_tensor.shape) == np.prod(py_tensor.shape):
            print(colorize(f"  💡 Same total elements ({np.prod(cpp_tensor.shape)}) - might just need reshape", Colors.WARNING))

        return False
    else:
        print(colorize(f"  ✅ MATCH: {cpp_tensor.shape}", Colors.OKGREEN))

    # Value statistics
    print(f"\n📊 C++ Tensor:")
    print_tensor_info(cpp_tensor, "C++")

    print(f"\n📊 Python Tensor:")
    print_tensor_info(py_tensor, "Python")

    # Numerical comparison
    print(f"\n🔍 Numerical Comparison:")

    # Handle NaN/Inf
    if np.any(np.isnan(cpp_tensor)) or np.any(np.isnan(py_tensor)):
        print(colorize("  ❌ FAIL: Contains NaN values", Colors.FAIL))
        return False

    if np.any(np.isinf(cpp_tensor)) or np.any(np.isinf(py_tensor)):
        print(colorize("  ❌ FAIL: Contains Inf values", Colors.FAIL))
        return False

    # Calculate differences
    abs_diff = np.abs(cpp_tensor - py_tensor)
    rel_diff = abs_diff / (np.abs(py_tensor) + 1e-8)

    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    max_rel_diff = rel_diff.max()
    mean_rel_diff = rel_diff.mean()

    print(f"  Absolute difference:")
    print(f"    Max:  {max_abs_diff:.6e}")
    print(f"    Mean: {mean_abs_diff:.6e}")
    print(f"  Relative difference:")
    print(f"    Max:  {max_rel_diff:.6e}")
    print(f"    Mean: {mean_rel_diff:.6e}")

    # Tolerance check
    passes_atol = max_abs_diff < atol
    passes_rtol = max_rel_diff < rtol
    passes_both = np.allclose(cpp_tensor, py_tensor, atol=atol, rtol=rtol)

    print(f"\n  Tolerance check (atol={atol}, rtol={rtol}):")
    print(f"    Absolute: {colorize('✅ PASS' if passes_atol else '❌ FAIL', Colors.OKGREEN if passes_atol else Colors.FAIL)}")
    print(f"    Relative: {colorize('✅ PASS' if passes_rtol else '❌ FAIL', Colors.OKGREEN if passes_rtol else Colors.FAIL)}")
    print(f"    Overall:  {colorize('✅ PASS' if passes_both else '❌ FAIL', Colors.OKGREEN if passes_both else Colors.FAIL)}")

    # Distribution of differences
    if not passes_both:
        print(f"\n  📈 Difference distribution:")
        thresholds = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        total = abs_diff.size

        for thresh in thresholds:
            count = (abs_diff < thresh).sum()
            pct = 100.0 * count / total
            bar = '█' * int(pct / 2)
            print(f"    < {thresh:.0e}: {count:10d} ({pct:5.1f}%) {bar}")

        # Show worst offenders
        print(f"\n  🔥 Worst differences (top 5):")
        flat_diff = abs_diff.flatten()
        worst_indices_flat = np.argpartition(flat_diff, -5)[-5:]
        worst_indices_flat = worst_indices_flat[np.argsort(flat_diff[worst_indices_flat])][::-1]

        for i, idx_flat in enumerate(worst_indices_flat, 1):
            idx = np.unravel_index(idx_flat, abs_diff.shape)
            cpp_val = cpp_tensor[idx]
            py_val = py_tensor[idx]
            diff_val = abs_diff[idx]
            print(f"    {i}. Index {idx}: C++={cpp_val:+.6f}, Py={py_val:+.6f}, Diff={diff_val:.6e}")

    return passes_both

def find_matching_files():
    """Find and match corresponding C++ and Python debug files"""
    cpp_files = sorted(glob.glob('debug_cpp_*.npy'))
    py_files = sorted(glob.glob('debug_activation_*.npy'))

    print(f"\n📁 Found files:")
    print(f"  C++ files: {len(cpp_files)}")
    print(f"  Python files: {len(py_files)}")

    if cpp_files:
        print(f"\n  C++ outputs:")
        for f in cpp_files:
            size = Path(f).stat().st_size
            print(f"    - {f} ({size:,} bytes)")

    if py_files:
        print(f"\n  Python outputs:")
        for f in py_files:
            size = Path(f).stat().st_size
            print(f"    - {f} ({size:,} bytes)")

    return cpp_files, py_files

def main():
    parser = argparse.ArgumentParser(description='Compare llama.cpp and PyTorch vision encoder outputs')
    parser.add_argument('--atol', type=float, default=1e-4,
                        help='Absolute tolerance for comparison (default: 1e-4)')
    parser.add_argument('--rtol', type=float, default=1e-5,
                        help='Relative tolerance for comparison (default: 1e-5)')
    parser.add_argument('--cpp-file', type=str,
                        help='Specific C++ file to compare')
    parser.add_argument('--py-file', type=str,
                        help='Specific Python file to compare')

    args = parser.parse_args()

    print(colorize("\n" + "="*80, Colors.HEADER))
    print(colorize("GEMMA3N VISION ENCODER OUTPUT COMPARISON", Colors.HEADER))
    print(colorize("="*80 + "\n", Colors.HEADER))

    # Manual comparison mode
    if args.cpp_file and args.py_file:
        if not Path(args.cpp_file).exists():
            print(colorize(f"❌ C++ file not found: {args.cpp_file}", Colors.FAIL))
            return 1
        if not Path(args.py_file).exists():
            print(colorize(f"❌ Python file not found: {args.py_file}", Colors.FAIL))
            return 1

        cpp_tensor = np.load(args.cpp_file)
        py_tensor = np.load(args.py_file)

        result = compare_tensors(cpp_tensor, py_tensor,
                                f"{args.cpp_file} vs {args.py_file}",
                                atol=args.atol, rtol=args.rtol)

        return 0 if result else 1

    # Auto-discovery mode
    cpp_files, py_files = find_matching_files()

    if not cpp_files:
        print(colorize("\n❌ No C++ debug files found (debug_cpp_*.npy)", Colors.FAIL))
        print("\nTo generate C++ outputs, add to your clip.cpp:")
        print(colorize("""
    #include "debug_tensor_dump.h"

    // In build_mobilenetv5():
    DEBUG_DUMP_TENSOR(ctx0, tensor, "descriptive_name");
        """, Colors.OKCYAN))
        return 1

    if not py_files and not Path('debug_vision_output.npy').exists():
        print(colorize("\n❌ No Python debug files found", Colors.FAIL))
        print("\nTo generate Python outputs, run:")
        print(colorize("    python debug_gemma3n.py image.jpg /path/to/model", Colors.OKCYAN))
        return 1

    # Compare final outputs
    final_cpp = None
    final_py = None

    # Try to find final output files
    if 'debug_cpp_05_final_output.npy' in cpp_files:
        final_cpp = 'debug_cpp_05_final_output.npy'
    elif cpp_files:
        final_cpp = cpp_files[-1]  # Use last file as fallback

    if Path('debug_vision_output.npy').exists():
        final_py = 'debug_vision_output.npy'

    results = []

    if final_cpp and final_py:
        print(colorize("\n" + "="*80, Colors.HEADER))
        print(colorize("FINAL OUTPUT COMPARISON", Colors.HEADER))
        print(colorize("="*80, Colors.HEADER))

        cpp_tensor = np.load(final_cpp)
        py_tensor = np.load(final_py)

        result = compare_tensors(cpp_tensor, py_tensor, "Final Vision Output",
                                atol=args.atol, rtol=args.rtol)
        results.append(("Final Output", result))

    # Summary
    print(f"\n{'='*80}")
    print(colorize("SUMMARY", Colors.BOLD))
    print('='*80)

    if results:
        passed = sum(1 for _, r in results if r)
        total = len(results)

        print(f"\nTests passed: {passed}/{total}")

        for name, result in results:
            status = colorize("✅ PASS", Colors.OKGREEN) if result else colorize("❌ FAIL", Colors.FAIL)
            print(f"  {name:30s} {status}")

        if passed == total:
            print(colorize("\n🎉 All tests passed!", Colors.OKGREEN))
            print("Your C++ implementation matches PyTorch!")
            return 0
        else:
            print(colorize(f"\n⚠️  {total - passed} test(s) failed", Colors.WARNING))
            print("\nDebugging suggestions:")
            print("  1. Check the first checkpoint where values diverge")
            print("  2. Add DEBUG_DUMP_TENSOR at intermediate layers")
            print("  3. Verify tensor shapes and operations at that layer")
            print("  4. See GEMMA3N_DEBUG_GUIDE.md for detailed debugging steps")
            return 1
    else:
        print(colorize("\n⚠️  No comparisons performed", Colors.WARNING))
        print("\nPlease ensure you have both C++ and Python debug outputs")
        return 1

if __name__ == "__main__":
    sys.exit(main())
