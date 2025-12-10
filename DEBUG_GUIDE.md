# VSCode Debugging Guide for llama.cpp

## Quick Start

### Method 1: VSCode Debugger (Recommended)

1. **Open the project in VSCode**: `code /home/user/llama.cpp`

2. **Set breakpoints** in `tools/mtmd/clip.cpp`:
   - Click in the left margin next to line numbers
   - Good breakpoint locations:
     - `clip.cpp:808` - Start of `build_mobilenetv5()`
     - `clip.cpp:597` - Start of `rms_norm_2d()`
     - `clip.cpp:641` - Inside `build_edge_residual()`
     - `clip.cpp:669` - Inside `build_inverted_residual()`

3. **Start debugging**:
   - Press `F5` or click "Run and Debug" in the sidebar
   - Select **"Debug MobileNetV5 (Direct Breakpoint)"** - This skips minja exceptions!
   - Alternative: "Debug llama-server" or "Debug llama-cli"

### Method 2: Command Line GDB (Faster)

```bash
./debug_vision.sh <model.gguf> <vision.gguf> [optional-image.jpg]
```

This script:
- Automatically skips irrelevant minja exceptions
- Sets breakpoints at key MobileNetV5 functions
- Forces single-threaded execution
- Provides custom commands like `ptensor <var>`

## Single-Threaded Debugging (Easier!)

The debug configurations are already set up to use single-threaded execution:
- `-t 1` flag limits to 1 thread
- `GGML_N_THREADS=1` environment variable
- `-ngl 0` forces CPU execution (no GPU)

This makes debugging much simpler because:
- No thread context switching
- Easier to follow execution flow
- More predictable breakpoint hits

## Key Debugging Commands

- `F5` - Continue execution
- `F10` - Step over (execute current line)
- `F11` - Step into (enter function)
- `Shift+F11` - Step out (exit function)
- `F9` - Toggle breakpoint

## Inspecting Tensors in GDB

When stopped at a breakpoint, you can inspect tensor values:

```gdb
# Print tensor dimensions
p cur->ne[0]  # Width
p cur->ne[1]  # Height
p cur->ne[2]  # Channels
p cur->ne[3]  # Batch

# Print tensor type
p cur->type

# Print first few values (if data is accessible)
p ((float*)cur->data)[0]
p ((float*)cur->data)[1]
```

## Common Debugging Scenarios

### 1. Debug shape mismatch errors

Set breakpoint at `clip.cpp:597` (start of `rms_norm_2d`):
```
(gdb) p inp->ne[0]  # Check input width
(gdb) p inp->ne[1]  # Check input height
(gdb) p inp->ne[2]  # Check input channels
(gdb) n              # Step through and check transformations
```

### 2. Debug convolution operations

Set breakpoint before `ggml_conv_2d` calls:
```
(gdb) p inp->ne[2]            # Input channels
(gdb) p weight->ne[2]         # Weight input channels (should match)
(gdb) p weight->ne[3]         # Weight output channels
```

### 3. Trace block execution

Set breakpoints in `build_mobilenetv5()` at line 838:
```
(gdb) p i                     # Current block index
(gdb) p block.s0_conv_exp_w   # Check if Edge Residual
(gdb) p block.attn_q_w        # Check if Attention
(gdb) p block.dw_start_w      # Check if UIR
```

## Conditional Breakpoints

Right-click on a breakpoint and select "Edit Breakpoint":
- Break only on specific block: `i == 10`
- Break only on specific stage: `stage == 2`
- Break on dimension mismatch: `cur->ne[2] != 64`

## Viewing Variables

The "Variables" panel shows:
- Local variables
- Function parameters
- Tensor dimensions and properties

Hover over variables in the code to see their values.

## Advanced: Watchpoints

Set a watchpoint to break when a variable changes:
```
Debug Console > -exec watch cur->ne[0]
```

## Tips for Effective Debugging

1. **Start with high-level breakpoints** (e.g., start of `build_mobilenetv5()`)
2. **Step through slowly** to understand the flow
3. **Watch tensor shapes** change through transformations
4. **Use the Debug Console** for custom GDB commands
5. **Check the Call Stack** to see how you got to the current point

## Rebuild After Changes

If you modify the code:
```bash
cmake --build build --config Debug --target mtmd -j 4
```

Or use the VSCode task: `Ctrl+Shift+B` → "Compile Debug"

## Performance Note

Debug builds are MUCH slower than release builds. Use them only for debugging,
not for actual inference.

## Handling Minja/Template Exceptions

The minja template library (used for chat templates) may throw exceptions during initialization. These are **usually harmless** and unrelated to the vision encoder.

### How to Skip Them:

**In VSCode:**
- Use the **"Debug MobileNetV5 (Direct Breakpoint)"** configuration
- It automatically skips minja exceptions with: `skip -rfu .*minja.*`

**In GDB manually:**
```gdb
skip -rfu .*minja.*
skip -rfu .*jinja.*
catch throw
commands
  silent
  # Check backtrace and continue if it's minja
  backtrace 1
  if $_regex($_streq, ".*minja.*") == 1
    continue
  end
end
```

**When an exception is actually important:**
- Check the **backtrace** (`bt`)
- Look for functions with **ggml**, **clip**, or **mobilenet** in the name
- If it's all minja/template code, just press `c` to continue

### Using the Debug Script:

The `debug_vision.sh` script handles this automatically:
```bash
./debug_vision.sh model.gguf vision.gguf
# Will skip minja and break at build_mobilenetv5()
```

## Troubleshooting

**Problem**: Exception in minja.hpp before reaching vision code
- **Solution**: This is normal! Use "Debug MobileNetV5 (Direct Breakpoint)" or press `c` to continue

**Problem**: Breakpoints not hitting
- **Solution**: Make sure you built with `-DCMAKE_BUILD_TYPE=Debug`

**Problem**: Can't see variable values
- **Solution**: They might be optimized away. Try stepping to where they're used.

**Problem**: Too many threads make debugging confusing
- **Solution**: The configs already use `-t 1`, but you can also add `OMP_NUM_THREADS=1`

**Problem**: Want to see the actual error message when exception is thrown
- **Solution**: In Debug Console, type: `-exec info exception` then `c` to see what was thrown
