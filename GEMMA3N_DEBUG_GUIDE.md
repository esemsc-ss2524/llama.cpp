# Gemma3n Vision Encoder Debugging Guide

Your Gemma3n implementation is running (256 tokens output, ~473s processing) but producing incorrect results. This guide will help you systematically compare your C++ implementation with the PyTorch reference.

## Quick Answer to Your Questions

### 1. Do I need a compatible GGUF for the text model?
**No.** The multimodal projector (mmproj GGUF) handles the conversion from vision embeddings to text model space. Your text model GGUF remains unchanged. The key code in `src/models/gemma3n-iswa.cpp:17-21` correctly skips scaling for image embeddings:

```cpp
// important: do not normalize weights for raw embeddings input
if (ubatch.token) {
    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));  // Only for text tokens
}
```

### 2. Are you faithfully replicating the vision tower?
**Mostly, but needs verification.** Your debug output shows correct structure:
- ✅ Input: [768, 768, 3, 1]
- ✅ After stem: [384, 384, 64, 1]
- ✅ Fusion points at blocks 44 and 83
- ✅ Final output: 256 tokens (16×16 from MSFA)

However, **incorrect results** suggest issues in:
- Image preprocessing (normalization values)
- MobileNetV5 block implementations (especially MQA attention)
- Tensor permutation/layout operations
- Activation functions or normalization layers

### 3. Best way to compare with transformers?
**Output intermediate tensors at key checkpoints** and compare numerically.

---

## Debugging Strategy

### Phase 1: Verify Preprocessing

#### Check 1.1: Normalization Parameters

```bash
# Check your mmproj GGUF metadata
python -c "
import gguf
reader = gguf.GGUFReader('your_mmproj.gguf')
for field in reader.fields.values():
    if 'mean' in field.name or 'std' in field.name:
        print(f'{field.name}: {field.parts[field.data[0]]}')
"
```

**Expected values** (ImageNet standard):
- `image_mean`: [0.485, 0.456, 0.406]
- `image_std`: [0.229, 0.224, 0.225]

If different, your preprocessing doesn't match PyTorch.

#### Check 1.2: Image Preprocessing

```python
# Run the debug script
python debug_gemma3n.py test_image.jpg /path/to/gemma3n-model

# This generates: debug_preprocessed_image.npy
```

Then add to your C++ code (in `clip.cpp` after normalization):
```cpp
#include "debug_tensor_dump.h"

// In build_mobilenetv5() after build_inp_raw():
ggml_tensor * inp = build_inp_raw();
DEBUG_DUMP_TENSOR(ctx0, inp, "input_raw");
```

Compare:
```python
import numpy as np
cpp_input = np.load('debug_cpp_input_raw.npy')
py_input = np.load('debug_preprocessed_image.npy')

print(f"C++:    shape={cpp_input.shape}, range=[{cpp_input.min():.4f}, {cpp_input.max():.4f}]")
print(f"Python: shape={py_input.shape}, range=[{py_input.min():.4f}, {py_input.max():.4f}]")
print(f"Max difference: {np.abs(cpp_input - py_input).max():.6f}")
```

**If difference > 1e-5**: Your normalization is wrong.

---

### Phase 2: Compare Vision Tower Outputs

#### Add Debug Points in clip.cpp

```cpp
#include "debug_tensor_dump.h"

ggml_cgraph * build_mobilenetv5() {
    fprintf(stderr, "\n--- START build_mobilenetv5 ---\n");

    // 1. Input
    ggml_tensor * inp = build_inp_raw();
    DEBUG_DUMP_TENSOR(ctx0, inp, "00_input");
    DEBUG_PRINT_STATS(inp, "Input");

    // 2. After stem
    ggml_tensor * cur = ggml_conv_2d(ctx0, model.mobilenet_stem_conv_w, inp, 2, 2, 1, 1, 1, 1);
    // ... rest of stem ...
    DEBUG_DUMP_TENSOR(ctx0, cur, "01_after_stem");
    DEBUG_PRINT_STATS(cur, "After Stem");

    // 3. After each stage
    for (int i = 0; i < total_blocks; i++) {
        // ... process block ...

        if (is_fusion_point(i)) {
            char name[64];
            snprintf(name, sizeof(name), "02_fusion_block_%d", i);
            DEBUG_DUMP_TENSOR(ctx0, cur, name);
        }
    }

    // 4. After MSFA
    DEBUG_DUMP_TENSOR(ctx0, cur, "03_after_msfa");
    DEBUG_PRINT_STATS(cur, "After MSFA");

    // 5. After projection
    if (model.mm_input_proj_w) {
        cur = ggml_mul_mat(ctx0, model.mm_input_proj_w, cur);
    }
    DEBUG_DUMP_TENSOR(ctx0, cur, "04_after_projection");

    // 6. After final norm
    if (model.mm_soft_emb_norm_w) {
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
    }
    DEBUG_DUMP_TENSOR(ctx0, cur, "05_final_output");
    DEBUG_PRINT_STATS(cur, "Final Output");

    ggml_build_forward_expand(gf, cur);
    return gf;
}
```

#### Compare with PyTorch

```python
import numpy as np
import glob

# Get all debug files
cpp_files = sorted(glob.glob('debug_cpp_*.npy'))
py_files = sorted(glob.glob('debug_activation_*.npy'))

print("Comparing outputs:")
for cpp_file in cpp_files:
    cpp_tensor = np.load(cpp_file)
    print(f"\n{cpp_file}:")
    print(f"  Shape: {cpp_tensor.shape}")
    print(f"  Range: [{cpp_tensor.min():.4f}, {cpp_tensor.max():.4f}]")
    print(f"  Mean: {cpp_tensor.mean():.4f}, Std: {cpp_tensor.std():.4f}")

    # Look for corresponding PyTorch tensor
    # (You'll need to match names manually)
```

---

### Phase 3: Check Specific Implementation Details

#### Critical Areas to Verify:

1. **MQA Attention** (`build_mobilenet_attn`):
   - Query projection and reshaping
   - Key/Value projection
   - Attention score calculation
   - Output projection
   - **Common issue**: Wrong tensor dimensions after reshape

2. **Edge Residual Blocks** (`build_edge_residual`):
   - Skip connection handling
   - Stride-2 downsampling
   - **Common issue**: Incorrect residual addition when shapes differ

3. **MSFA (Multi-Scale Fusion Adapter)**:
   - Feature collection at correct blocks
   - Resizing to target resolution (16×16)
   - Concatenation along channel dimension
   - **Common issue**: Wrong resize interpolation mode

4. **Tensor Permutations**:
   ```cpp
   cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
   cur = ggml_cont(ctx0, cur);  // IMPORTANT: Always cont() after permute!
   ```
   **Common issue**: Forgetting `ggml_cont()` leads to incorrect memory layout

5. **Layer Scaling**:
   - Verify layer_scale_w broadcasting
   - Check if scale is applied before or after residual

---

### Phase 4: Numerical Comparison Checklist

Run this comparison at each checkpoint:

```python
import numpy as np

def compare_tensors(cpp_path, py_path, name, atol=1e-4):
    cpp = np.load(cpp_path)
    py = np.load(py_path)

    print(f"\n=== {name} ===")

    # Shape check
    if cpp.shape != py.shape:
        print(f"❌ Shape mismatch: C++={cpp.shape} vs Py={py.shape}")
        return False

    # Value comparison
    diff = np.abs(cpp - py)
    max_diff = diff.max()
    mean_diff = diff.mean()
    num_large_diff = (diff > atol).sum()

    print(f"Shape: {cpp.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Elements with diff > {atol}: {num_large_diff}/{diff.size}")

    if max_diff < atol:
        print("✅ PASS")
        return True
    else:
        print("❌ FAIL")

        # Show where biggest differences are
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"Largest diff at index {idx}:")
        print(f"  C++: {cpp[idx]:.6f}")
        print(f"  Py:  {py[idx]:.6f}")

        # Histogram of differences
        print(f"Difference distribution:")
        print(f"  < 1e-6: {(diff < 1e-6).sum()}")
        print(f"  < 1e-5: {(diff < 1e-5).sum()}")
        print(f"  < 1e-4: {(diff < 1e-4).sum()}")
        print(f"  >= 1e-4: {(diff >= 1e-4).sum()}")

        return False

# Run comparisons
compare_tensors('debug_cpp_00_input.npy', 'debug_preprocessed_image.npy', 'Input')
compare_tensors('debug_cpp_01_after_stem.npy', 'debug_activation_stem.npy', 'After Stem')
# ... etc
```

---

## Common Issues and Fixes

### Issue 1: All outputs are zeros or NaN
**Cause**: Likely a normalization issue or incorrect weight loading.

**Fix**:
```cpp
// Add after loading weights:
fprintf(stderr, "Checking weights:\n");
DEBUG_PRINT_STATS(model.mobilenet_stem_conv_w, "Stem conv weight");
DEBUG_PRINT_STATS(model.mm_input_proj_w, "Input projection weight");
```

### Issue 2: Output scale is completely wrong
**Cause**: Missing activation, wrong normalization, or incorrect scaling factor.

**Fix**: Check that:
- RMS norm epsilon matches (usually 1e-6)
- Activation functions are applied (GELU, not ReLU)
- Layer scaling is applied correctly

### Issue 3: First few layers match, then diverge
**Cause**: Usually a tensor layout issue (missing `cont()` after permute).

**Fix**:
```cpp
// WRONG:
cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
cur = ggml_reshape_3d(ctx0, cur, ...);  // Will use wrong strides!

// CORRECT:
cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
cur = ggml_cont(ctx0, cur);  // Make contiguous first!
cur = ggml_reshape_3d(ctx0, cur, ...);
```

### Issue 4: Shapes are correct but values wrong
**Cause**: Likely an operation implementation difference.

**Fix**: Check operation order, especially:
- Attention: QK^T / sqrt(d_k) before softmax
- Residual connections: where exactly they're added
- Normalization: before or after projection

---

## Quick Test Commands

```bash
# 1. Check if image preprocessing matches
python debug_gemma3n.py test_image.jpg /path/to/model

# 2. Rebuild with debug symbols
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j

# 3. Run your server with debug output
./build/bin/llama-server -m model.gguf --mmproj mmproj.gguf 2>&1 | tee debug_output.txt

# 4. Compare outputs
python compare_outputs.py
```

---

## Expected Debug Output

When working correctly, you should see:

```
DEBUG: Input Raw                  | shape=[ 768,  768,    3,    1] | range=[-2.1179, +2.6400] | mean=+0.0823 | std=0.9945
DEBUG: After Stem                 | shape=[ 384,  384,   64,    1] | range=[-3.2156, +4.8921] | mean=+0.1234 | std=1.2341
DEBUG: After MSFA                 | shape=[  16,   16, 1920,    1] | range=[-5.1234, +6.2341] | mean=+0.0123 | std=2.1234
DEBUG: Final Output               | shape=[4096,  256,    1,    1] | range=[-8.1234, +9.2341] | mean=-0.0234 | std=3.4123
```

The key is that these statistics should match PyTorch to within ~1e-4 tolerance.

---

## Still Stuck?

If after following this guide you still have issues:

1. **Save both full outputs** (C++ and PyTorch) and compare element-by-element
2. **Binary search**: Comment out half the blocks, see if output matches
3. **Simplify**: Test with a solid color image (all pixels same value)
4. **Check weight conversion**: Verify your GGUF conversion script preserves all weight values

## Next Steps

Once you identify the divergence point:
1. Add detailed logging at that layer
2. Check the PyTorch source code for that layer
3. Verify tensor shapes before and after each operation
4. Compare intermediate values element-by-element

Remember: **The first checkpoint where values diverge is where your bug is.** Focus debugging efforts there.
