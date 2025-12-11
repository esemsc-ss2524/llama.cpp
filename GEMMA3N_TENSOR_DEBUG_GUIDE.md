# Gemma 3N Vision Encoder Tensor Debugging Guide

This guide explains how to extract and save intermediate tensors from the GGML-based Gemma 3N vision encoder (MobileNetV5) for debugging purposes.

## What Was Added

### 1. NumPy Tensor Saving Function (`clip-impl.h`)

A new helper function `save_tensor_to_npy()` was added to save GGML tensors in NumPy `.npy` format (compatible with your PyTorch debugging code).

**Location:** `tools/mtmd/clip-impl.h` (after line 528)

**Features:**
- Converts F16/F32 tensors to F32 for compatibility
- Writes proper NumPy .npy file format
- Handles shape conversion (GGML: [W,H,C,B] → NumPy: [B,C,H,W])

### 2. Tensor Naming in Graph Building (`clip.cpp`)

All key intermediate tensors are now named using `ggml_set_name()`:

**Stem:**
- `mobilenet_stem_output`

**Blocks (for each block index i):**
- `mobilenet_block_{i}_edge_residual` - Edge residual blocks (stage 0)
- `mobilenet_block_{i}_inverted_residual` - Inverted residual blocks
- `mobilenet_block_{i}_layer_scale` - Layer scale output
- `mobilenet_block_{i}_attn_norm` - Attention normalization
- `mobilenet_block_{i}_attn_key_norm` - Attention key normalization
- `mobilenet_block_{i}_attn_value_norm` - Attention value normalization
- `mobilenet_block_{i}_attn_output` - Attention block output

**MSFA (Multi-Scale Fusion Adapter):**
- `msfa_concat` - After concatenation
- `msfa_ffn_expand` - After FFN expansion
- `msfa_ffn_project` - After FFN projection
- `msfa_downsample` - After downsampling to 16x16
- `msfa_concat_norm` - After final normalization

**Projection:**
- `mm_input_proj` - After multimodal input projection
- `mm_soft_emb_norm` - After soft embedding normalization

### 3. Automatic Tensor Saving (`clip.cpp`)

After graph computation in `clip_image_batch_encode()`, all named tensors matching the patterns above are automatically saved to `.npy` files when debugging is enabled.

**Location:** `tools/mtmd/clip.cpp` (lines 5646-5685)

## How to Use

### Step 1: Enable Tensor Saving

Set the environment variable before running your inference:

```bash
export CLIP_DEBUG_SAVE_TENSORS=1
export CLIP_DEBUG_OUTPUT_DIR="debug_ggml_output"  # Optional, defaults to "debug_ggml_output"
```

### Step 2: Run Your Inference

Run your llama.cpp inference as normal. The intermediate tensors will be automatically saved.

```bash
# Example with llava
./llava-cli -m your_model.gguf --mmproj your_vision.gguf --image test_image.jpg
```

### Step 3: Check Output

You'll find `.npy` files in the output directory:

```bash
ls -lh debug_ggml_output/
# Expected files:
# debug_mobilenet_stem_output.npy
# debug_mobilenet_block_0_edge_residual.npy
# debug_mobilenet_block_1_inverted_residual.npy
# debug_mobilenet_block_2_attn_norm.npy
# debug_mobilenet_block_2_attn_key_norm.npy
# debug_mobilenet_block_2_attn_value_norm.npy
# debug_mobilenet_block_2_layer_scale.npy
# debug_msfa_concat.npy
# debug_msfa_ffn_expand.npy
# debug_msfa_concat_norm.npy
# debug_mm_input_proj.npy
# debug_mm_soft_emb_norm.npy
# ... etc
```

### Step 4: Compare with PyTorch

Load and compare tensors in Python:

```python
import numpy as np

# Load GGML tensor
ggml_tensor = np.load("debug_ggml_output/debug_mobilenet_block_2_attn_key_norm.npy")

# Load PyTorch tensor
pt_tensor = np.load("debug_output/debug_activation_timm_model_blocks_2_9_attn_key_norm.npy")

print(f"GGML shape: {ggml_tensor.shape}")
print(f"PyTorch shape: {pt_tensor.shape}")

# Calculate differences
diff = np.abs(ggml_tensor - pt_tensor)
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
print(f"Relative error: {(diff / (np.abs(pt_tensor) + 1e-8)).mean()}")

# Visual comparison
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ggml_tensor[0, :, :, 0], cmap='viridis')
plt.title("GGML")
plt.subplot(1, 3, 2)
plt.imshow(pt_tensor[0, :, :, 0], cmap='viridis')
plt.title("PyTorch")
plt.subplot(1, 3, 3)
plt.imshow(diff[0, :, :, 0], cmap='hot')
plt.title("Absolute Difference")
plt.tight_layout()
plt.show()
```

## Mapping PyTorch to GGML Tensor Names

Based on your PyTorch debug output, here's how to map names:

| PyTorch Name | GGML Name |
|--------------|-----------|
| `timm_model_blocks_1_0_layer_scale` | `mobilenet_block_?_layer_scale` |
| `timm_model_blocks_2_9_norm` | `mobilenet_block_?_attn_norm` |
| `timm_model_blocks_2_9_attn_key_norm` | `mobilenet_block_?_attn_key_norm` |
| `timm_model_blocks_2_9_attn_value_norm` | `mobilenet_block_?_attn_value_norm` |

**Note:** The block indices may differ between PyTorch and GGML depending on how the model was converted. Check the shape and position in the network to match them correctly.

## Debugging Tips

1. **Check Shapes First:** Always verify that tensor shapes match between PyTorch and GGML
   ```python
   print(f"Shape match: {ggml_tensor.shape == pt_tensor.shape}")
   ```

2. **Transpose if Needed:** GGML uses [W,H,C,B] layout, which is converted to [B,C,H,W] in the .npy files. If shapes don't match, try:
   ```python
   ggml_transposed = np.transpose(ggml_tensor, (0, 2, 3, 1))  # [B,C,H,W] -> [B,H,W,C]
   ```

3. **Check Normalization:** Look for discrepancies in RMS normalization (especially epsilon values)

4. **Attention Patterns:** For attention blocks, check:
   - Key/Value downsampling (stride=2)
   - Multi-query attention broadcasting
   - Softmax output distributions

5. **Layer Scale:** Check if layer_scale weights are applied correctly

## Troubleshooting

### No .npy files generated
- Check that `CLIP_DEBUG_SAVE_TENSORS=1` is set
- Verify the output directory exists and is writable
- Check logs for "Saving intermediate tensors" message

### Shape mismatches
- GGML tensors are saved in [B,C,H,W] order (NumPy convention)
- Original GGML layout is [W,H,C,B] - conversion is automatic
- Check if transpose is needed for comparison

### Large numerical differences
- Check data types (F16 vs F32)
- Verify normalization parameters (eps, mean, std)
- Look for quantization issues if using quantized weights

## Advanced: Adding More Tensors

To save additional intermediate tensors:

1. Add `ggml_set_name(tensor, "your_name")` in the graph building code
2. Update the `should_save` condition in `clip_image_batch_encode()`:
   ```cpp
   bool should_save = (
       name_str.find("mobilenet_") != std::string::npos ||
       name_str.find("msfa_") != std::string::npos ||
       name_str.find("mm_") != std::string::npos ||
       name_str.find("your_pattern_") != std::string::npos ||  // Add this
       name_str == "inp_raw"
   );
   ```

## Files Modified

1. **tools/mtmd/clip-impl.h**
   - Added `save_tensor_to_npy()` function (lines 530-613)

2. **tools/mtmd/clip.cpp**
   - Added headers: `sys/stat.h`, `sys/types.h`
   - Modified `build_edge_residual()` - added block_idx parameter and tensor naming
   - Modified `build_inverted_residual()` - added block_idx parameter and tensor naming
   - Modified `build_mobilenet_attn()` - added block_idx parameter and tensor naming
   - Modified `build_mobilenetv5()` - added tensor naming for stem, MSFA, and projection
   - Modified `clip_image_batch_encode()` - added automatic tensor saving logic (lines 5646-5685)

## Building

Rebuild llama.cpp after making these changes:

```bash
cd /home/user/llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

## Example Output

When running with debugging enabled:

```
DEBUG: Input Raw                 | Type: f32 | Shape: [768, 768, 3, 1]
DEBUG: After Stem                | Type: f32 | Shape: [384, 384, 32, 1]
DEBUG: Collecting fusion feature at block index 19
DEBUG: Fusion Feature            | Type: f32 | Shape: [24, 24, 640, 1]
DEBUG: Collecting fusion feature at block index 29
DEBUG: Fusion Feature            | Type: f32 | Shape: [16, 16, 2048, 1]

=== Saving intermediate tensors to debug_ggml_output ===
Saved tensor 'mobilenet_stem_output' with shape [384, 384, 32, 1] to debug_ggml_output/debug_mobilenet_stem_output.npy
Saved tensor 'mobilenet_block_0_edge_residual' with shape [192, 192, 64, 1] to debug_ggml_output/debug_mobilenet_block_0_edge_residual.npy
...
=== Tensor saving complete ===
```

---

Happy debugging! Compare these tensors with your PyTorch outputs to identify where discrepancies occur in the conversion.
