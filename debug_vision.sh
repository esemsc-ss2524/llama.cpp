#!/bin/bash
# Debug script for MobileNetV5 vision encoder
# Usage: ./debug_vision.sh <model.gguf> <vision.gguf>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model.gguf> <vision.gguf> [image.jpg]"
    exit 1
fi

MODEL_PATH="$1"
VISION_PATH="$2"
IMAGE_PATH="${3:-}"

# Force single-threaded execution
export GGML_N_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Build with debug symbols if not already done
if [ ! -f "build/bin/libmtmd.so" ] || [ "$(file build/bin/libmtmd.so | grep -c 'not stripped')" -eq 0 ]; then
    echo "Building with debug symbols..."
    cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF
    cmake --build build --config Debug --target mtmd -j 4
fi

# Create GDB batch file
cat > /tmp/gdb_batch.txt <<'EOF'
# Skip minja/jinja exceptions
skip -rfu .*minja.*
skip -rfu .*jinja.*

# Set breakpoints
break clip_graph::build_mobilenetv5
break ggml_abort

# Catch exceptions but auto-continue minja ones
catch throw
commands
  silent
  # Check if it's from minja
  backtrace 1
  if $_regex($_streq, ".*minja.*") == 1
    continue
  end
  # Not minja, stop and show details
  printf "\n=== Exception Caught ===\n"
  backtrace
  printf "========================\n"
end

# Print tensor helper
define ptensor
  printf "Tensor %s: [%lld, %lld, %lld, %lld] type=%d\n", "$arg0", $arg0->ne[0], $arg0->ne[1], $arg0->ne[2], $arg0->ne[3], $arg0->type
end

set pagination off
set print pretty on
set breakpoint pending on

printf "\n=== Starting Debug Session ===\n"
printf "Breakpoints set at:\n"
printf "  - build_mobilenetv5()\n"
printf "  - ggml_abort()\n"
printf "Commands available:\n"
printf "  - ptensor <var> : Print tensor info\n"
printf "  - c : Continue\n"
printf "  - n : Next line\n"
printf "  - s : Step into\n"
printf "================================\n\n"

run
EOF

# Determine which binary to debug
if [ -n "$IMAGE_PATH" ]; then
    echo "Debugging llama-cli with image: $IMAGE_PATH"
    gdb -x /tmp/gdb_batch.txt \
        --args build/bin/llama-cli \
        -m "$MODEL_PATH" \
        --mmproj "$VISION_PATH" \
        --image "$IMAGE_PATH" \
        -p "Describe this image" \
        -ngl 0 -t 1 -tb 1 -tg 1 \
        --no-mmap
else
    echo "Debugging llama-server (will break at warmup)"
    gdb -x /tmp/gdb_batch.txt \
        --args build/bin/llama-server \
        -m "$MODEL_PATH" \
        --mmproj "$VISION_PATH" \
        -ngl 0 -t 1 -tb 1 -tg 1 \
        --no-mmap
fi

# Cleanup
rm -f /tmp/gdb_batch.txt
