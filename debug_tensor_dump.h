/**
 * Helper utilities for debugging Gemma3n vision encoder
 *
 * Usage in clip.cpp:
 *   #include "debug_tensor_dump.h"
 *
 *   // After building a tensor:
 *   DEBUG_DUMP_TENSOR(ctx0, some_tensor, "my_tensor_name");
 *
 *   // To compare with PyTorch:
 *   python debug_gemma3n.py image.jpg model_path
 *   // Then compare the .npy files
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <cstring>

namespace debug {

// Save a tensor to a numpy .npy file for comparison with PyTorch
inline void save_tensor_to_npy(ggml_tensor * tensor, const char * filename) {
    if (!tensor) {
        fprintf(stderr, "DEBUG: Cannot save NULL tensor to %s\n", filename);
        return;
    }

    // Get tensor data size
    size_t nelements = ggml_nelements(tensor);
    size_t nbytes = ggml_nbytes(tensor);

    // Allocate buffer and copy data from backend
    std::vector<float> buffer(nelements);
    ggml_backend_tensor_get(tensor, buffer.data(), 0, nbytes);

    // Open file
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "DEBUG: Failed to open %s for writing\n", filename);
        return;
    }

    // Write numpy header (simple format for float32, C-contiguous)
    // Magic number
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);

    // Version 1.0
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);

    // Build header dict
    char header[256];
    int ndim = 0;
    int64_t shape[4] = {0};

    // Get dimensions (reverse order for numpy)
    for (int i = 0; i < 4; i++) {
        if (tensor->ne[i] > 1 || (i == 0 && tensor->ne[i] == 1)) {
            shape[ndim++] = tensor->ne[i];
        }
    }

    // Reverse shape for numpy convention
    std::reverse(&shape[0], &shape[ndim]);

    // Format header
    int header_len;
    if (ndim == 1) {
        header_len = snprintf(header, sizeof(header),
            "{'descr': '<f4', 'fortran_order': False, 'shape': (%ld,), }",
            shape[0]);
    } else if (ndim == 2) {
        header_len = snprintf(header, sizeof(header),
            "{'descr': '<f4', 'fortran_order': False, 'shape': (%ld, %ld), }",
            shape[0], shape[1]);
    } else if (ndim == 3) {
        header_len = snprintf(header, sizeof(header),
            "{'descr': '<f4', 'fortran_order': False, 'shape': (%ld, %ld, %ld), }",
            shape[0], shape[1], shape[2]);
    } else if (ndim == 4) {
        header_len = snprintf(header, sizeof(header),
            "{'descr': '<f4', 'fortran_order': False, 'shape': (%ld, %ld, %ld, %ld), }",
            shape[0], shape[1], shape[2], shape[3]);
    }

    // Pad to 64-byte boundary
    int total_header_len = header_len;
    while ((total_header_len + 10) % 64 != 0) {
        header[total_header_len++] = ' ';
    }
    header[total_header_len++] = '\n';

    // Write header length
    uint16_t header_len_bytes = total_header_len;
    fwrite(&header_len_bytes, 2, 1, f);

    // Write header
    fwrite(header, 1, total_header_len, f);

    // Write data
    fwrite(buffer.data(), sizeof(float), nelements, f);

    fclose(f);

    fprintf(stderr, "DEBUG: Saved tensor '%s' to %s (shape: [%ld, %ld, %ld, %ld], %zu elements)\n",
            tensor->name ? tensor->name : "unnamed",
            filename,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            nelements);
}

// Print tensor statistics for debugging
inline void print_tensor_stats(ggml_tensor * tensor, const char * label) {
    if (!tensor) {
        fprintf(stderr, "DEBUG: %s is NULL\n", label);
        return;
    }

    size_t nelements = ggml_nelements(tensor);
    std::vector<float> buffer(nelements);
    ggml_backend_tensor_get(tensor, buffer.data(), 0, ggml_nbytes(tensor));

    // Calculate statistics
    float min_val = buffer[0];
    float max_val = buffer[0];
    double sum = 0.0;

    for (size_t i = 0; i < nelements; i++) {
        float val = buffer[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    float mean = sum / nelements;

    // Calculate std dev
    double var_sum = 0.0;
    for (size_t i = 0; i < nelements; i++) {
        float diff = buffer[i] - mean;
        var_sum += diff * diff;
    }
    float std_dev = sqrt(var_sum / nelements);

    fprintf(stderr, "DEBUG: %-30s | shape=[%4ld, %4ld, %4ld, %4ld] | "
                    "range=[%+.4f, %+.4f] | mean=%+.4f | std=%.4f\n",
            label,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            min_val, max_val, mean, std_dev);
}

// Compare two tensors and report differences
inline void compare_tensors(ggml_tensor * a, ggml_tensor * b, const char * label, float threshold = 1e-5) {
    if (!a || !b) {
        fprintf(stderr, "DEBUG: Cannot compare NULL tensors for %s\n", label);
        return;
    }

    // Check shapes match
    bool shape_match = true;
    for (int i = 0; i < 4; i++) {
        if (a->ne[i] != b->ne[i]) {
            shape_match = false;
            break;
        }
    }

    if (!shape_match) {
        fprintf(stderr, "DEBUG: %s - Shape mismatch! A=[%ld,%ld,%ld,%ld] vs B=[%ld,%ld,%ld,%ld]\n",
                label,
                a->ne[0], a->ne[1], a->ne[2], a->ne[3],
                b->ne[0], b->ne[1], b->ne[2], b->ne[3]);
        return;
    }

    size_t nelements = ggml_nelements(a);
    std::vector<float> buf_a(nelements);
    std::vector<float> buf_b(nelements);

    ggml_backend_tensor_get(a, buf_a.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_get(b, buf_b.data(), 0, ggml_nbytes(b));

    size_t num_diffs = 0;
    float max_diff = 0.0f;
    double sum_abs_diff = 0.0;

    for (size_t i = 0; i < nelements; i++) {
        float diff = fabs(buf_a[i] - buf_b[i]);
        if (diff > threshold) {
            num_diffs++;
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
        sum_abs_diff += diff;
    }

    float mean_abs_diff = sum_abs_diff / nelements;

    fprintf(stderr, "DEBUG: %s comparison - %zu/%zu elements differ (threshold=%.1e) | "
                    "max_diff=%.4e | mean_abs_diff=%.4e\n",
            label, num_diffs, nelements, threshold, max_diff, mean_abs_diff);
}

} // namespace debug

// Convenience macros
#define DEBUG_DUMP_TENSOR(ctx, tensor, name) \
    do { \
        char filename[256]; \
        snprintf(filename, sizeof(filename), "debug_cpp_%s.npy", name); \
        debug::save_tensor_to_npy(tensor, filename); \
    } while(0)

#define DEBUG_PRINT_STATS(tensor, label) \
    debug::print_tensor_stats(tensor, label)

#define DEBUG_COMPARE_TENSORS(a, b, label) \
    debug::compare_tensors(a, b, label)

// Example usage in clip.cpp:
/*
#include "debug_tensor_dump.h"

// In build_mobilenetv5():
ggml_tensor * inp = build_inp_raw();
DEBUG_DUMP_TENSOR(ctx0, inp, "input_raw");
DEBUG_PRINT_STATS(inp, "Input Raw");

// After stem:
DEBUG_DUMP_TENSOR(ctx0, cur, "after_stem");
DEBUG_PRINT_STATS(cur, "After Stem");

// After MSFA:
DEBUG_DUMP_TENSOR(ctx0, cur, "after_msfa");
DEBUG_PRINT_STATS(cur, "After MSFA");

// After final projection:
DEBUG_DUMP_TENSOR(ctx0, cur, "final_output");
DEBUG_PRINT_STATS(cur, "Final Output");
*/
