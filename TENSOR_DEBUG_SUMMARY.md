# Summary: Intermediate Tensor Debugging for Gemma 3N Vision Encoder

## What Was Implemented

A complete debugging system to extract intermediate tensors from your GGML MobileNetV5 (Gemma 3N vision encoder) implementation and save them as NumPy files for comparison with PyTorch outputs.

## Key Features

### 1. **NumPy Export Function** (`clip-impl.h`)
- Converts GGML tensors (F16/F32) to NumPy `.npy` format
- Proper header generation and shape handling
- Compatible with your existing PyTorch debugging workflow

### 2. **Comprehensive Tensor Naming** (`clip.cpp`)
Throughout the graph building process:
- Stem layer output
- All MobileNet blocks (edge residual, inverted residual, attention)
- Attention sub-components (norm, key_norm, value_norm, layer_scale)
- MSFA (Multi-Scale Fusion Adapter) stages
- Final projection layers

### 3. **Automatic Saving** (`clip.cpp`)
After graph computation, named tensors are automatically saved when enabled via environment variable.

## Quick Start

```bash
# 1. Build the project
cd /home/user/llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)

# 2. Enable debugging
export CLIP_DEBUG_SAVE_TENSORS=1
export CLIP_DEBUG_OUTPUT_DIR="debug_ggml_output"

# 3. Run inference
./llava-cli -m your_model.gguf --mmproj your_vision.gguf --image test.jpg

# 4. Compare tensors
python compare_tensors.py
```

## Files Modified

1. **tools/mtmd/clip-impl.h**
   - Added `save_tensor_to_npy()` function (90 lines)

2. **tools/mtmd/clip.cpp**
   - Added system headers for directory creation
   - Modified 4 functions to add tensor naming:
     - `build_edge_residual()`
     - `build_inverted_residual()`
     - `build_mobilenet_attn()`
     - `build_mobilenetv5()`
   - Added automatic saving logic in `clip_image_batch_encode()`

## Comparison Workflow

```python
import numpy as np

# Load GGML output
ggml_out = np.load("debug_ggml_output/debug_mobilenet_block_2_attn_key_norm.npy")

# Load PyTorch output
pt_out = np.load("debug_output/debug_activation_timm_model_blocks_2_9_attn_key_norm.npy")

# Compare
print(f"Max diff: {np.abs(ggml_out - pt_out).max()}")
print(f"Mean diff: {np.abs(ggml_out - pt_out).mean()}")
```

## Expected Output Structure

```
debug_ggml_output/
├── debug_mobilenet_stem_output.npy
├── debug_mobilenet_block_0_edge_residual.npy
├── debug_mobilenet_block_1_inverted_residual.npy
├── debug_mobilenet_block_2_attn_norm.npy
├── debug_mobilenet_block_2_attn_key_norm.npy
├── debug_mobilenet_block_2_attn_value_norm.npy
├── debug_mobilenet_block_2_layer_scale.npy
├── debug_msfa_concat.npy
├── debug_msfa_ffn_expand.npy
├── debug_msfa_ffn_project.npy
├── debug_msfa_downsample.npy
├── debug_msfa_concat_norm.npy
├── debug_mm_input_proj.npy
└── debug_mm_soft_emb_norm.npy
```

## Benefits

1. **Direct Comparison:** Same file format as your PyTorch debug outputs
2. **Comprehensive Coverage:** All major intermediate stages captured
3. **Easy to Enable/Disable:** Simple environment variable control
4. **No Performance Impact:** Only active when explicitly enabled
5. **Automatic:** No manual code changes needed after initial setup

## Next Steps

1. Build and test the code
2. Run inference with debugging enabled
3. Compare outputs with your PyTorch reference
4. Identify any discrepancies at specific layers
5. Fix issues in the corresponding GGML layer implementation

See `GEMMA3N_TENSOR_DEBUG_GUIDE.md` for detailed usage instructions.
