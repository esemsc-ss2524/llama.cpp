#!/usr/bin/env python3
"""
Debug script to compare Gemma3n vision encoder outputs between transformers and llama.cpp

Usage:
  1. Install: pip install torch transformers pillow numpy
  2. Run: python debug_gemma3n.py <path_to_image> <path_to_model>

This script will:
  - Load the same image that llama.cpp uses
  - Process it through the PyTorch model
  - Output intermediate activations for comparison
  - Save tensors to numpy files for analysis
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
import sys
import os

def normalize_image(img_array, mean, std):
    """Normalize image using mean and std (same as llama.cpp)"""
    img_float = img_array.astype(np.float32) / 255.0
    for c in range(3):
        img_float[:, :, c] = (img_float[:, :, c] - mean[c]) / std[c]
    return img_float

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <image_path> <model_path>")
        print("\nExample:")
        print(f"  {sys.argv[0]} test_image.jpg /path/to/gemma3n-model")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 80)
    print("GEMMA3N VISION ENCODER DEBUG COMPARISON")
    print("=" * 80)

    # Load model and processor
    print(f"\n1. Loading model from: {model_path}")
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model.eval()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        sys.exit(1)

    # Load and preprocess image
    print(f"\n2. Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"   Original size: {image.size}")

        # Process image using the model's processor
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        print(f"   Processed tensor shape: {pixel_values.shape}")
        print(f"   Tensor dtype: {pixel_values.dtype}")
        print(f"   Value range: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")

        # Save preprocessed image for comparison
        np.save("debug_preprocessed_image.npy", pixel_values.numpy())
        print("   ✓ Saved: debug_preprocessed_image.npy")

    except Exception as e:
        print(f"   ✗ Error loading image: {e}")
        sys.exit(1)

    # Get vision encoder
    print("\n3. Extracting vision encoder")
    try:
        # This will vary based on the actual model architecture
        # Common patterns:
        if hasattr(model, 'vision_tower'):
            vision_model = model.vision_tower
        elif hasattr(model, 'vision_model'):
            vision_model = model.vision_model
        elif hasattr(model, 'vision_encoder'):
            vision_model = model.vision_encoder
        else:
            print("   Searching for vision encoder in model...")
            for name, module in model.named_modules():
                if 'vision' in name.lower():
                    print(f"   Found potential vision module: {name}")
            raise AttributeError("Could not find vision encoder. Please check model architecture.")

        print(f"   ✓ Found vision encoder: {type(vision_model).__name__}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("\n   Model structure:")
        for name, module in model.named_children():
            print(f"     - {name}: {type(module).__name__}")
        sys.exit(1)

    # Forward pass with hooks to capture intermediate activations
    print("\n4. Running forward pass with debug hooks")

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().cpu()
        return hook

    # Register hooks on key layers
    # Adjust these based on actual model architecture
    hooks = []
    layer_count = 0

    for name, module in vision_model.named_modules():
        # Hook conv layers, attention layers, and normalization layers
        if any(x in name.lower() for x in ['conv', 'attn', 'norm', 'proj', 'fusion', 'msfa']):
            if len(list(module.children())) == 0:  # Only leaf modules
                hooks.append(module.register_forward_hook(get_activation(name)))
                layer_count += 1

    print(f"   Registered {layer_count} hooks")

    # Run inference
    with torch.no_grad():
        try:
            # Try different input methods
            if hasattr(vision_model, 'encode_images'):
                output = vision_model.encode_images(pixel_values)
            elif hasattr(vision_model, 'forward'):
                output = vision_model(pixel_values)
            else:
                # Full model forward
                output = model(pixel_values=pixel_values)
                if hasattr(output, 'image_embeds'):
                    output = output.image_embeds
                elif hasattr(output, 'vision_outputs'):
                    output = output.vision_outputs

            print(f"   ✓ Forward pass complete")
            print(f"   Output shape: {output.shape if torch.is_tensor(output) else 'complex output'}")

            if torch.is_tensor(output):
                print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
                print(f"   Output mean: {output.mean():.4f}, std: {output.std():.4f}")

                # Save final output
                np.save("debug_vision_output.npy", output.cpu().numpy())
                print("   ✓ Saved: debug_vision_output.npy")

        except Exception as e:
            print(f"   ✗ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save intermediate activations
    print("\n5. Saving intermediate activations")
    saved_count = 0
    for name, tensor in activations.items():
        if torch.is_tensor(tensor):
            safe_name = name.replace('.', '_').replace('/', '_')
            filename = f"debug_activation_{safe_name}.npy"
            np.save(filename, tensor.numpy())
            saved_count += 1

            # Print stats for key layers
            if any(x in name.lower() for x in ['stem', 'fusion', 'proj', 'norm']):
                print(f"   {name:50s} shape={str(tuple(tensor.shape)):20s} "
                      f"range=[{tensor.min():.4f}, {tensor.max():.4f}] "
                      f"mean={tensor.mean():.4f}")

    print(f"\n   ✓ Saved {saved_count} activation tensors")

    # Print normalization parameters if accessible
    print("\n6. Checking normalization parameters")
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        if hasattr(ip, 'image_mean'):
            print(f"   Image mean: {ip.image_mean}")
        if hasattr(ip, 'image_std'):
            print(f"   Image std:  {ip.image_std}")
        if hasattr(ip, 'size'):
            print(f"   Image size: {ip.size}")
        if hasattr(ip, 'do_rescale'):
            print(f"   Do rescale: {ip.do_rescale}")
        if hasattr(ip, 'rescale_factor'):
            print(f"   Rescale factor: {ip.rescale_factor}")

    # Generate comparison instructions
    print("\n" + "=" * 80)
    print("COMPARISON INSTRUCTIONS")
    print("=" * 80)
    print("""
1. Check preprocessing:
   - Compare debug_preprocessed_image.npy with llama.cpp input
   - Verify normalization values match

2. Compare intermediate activations:
   - Add debug prints in clip.cpp at corresponding layers
   - Compare tensor shapes and value ranges
   - Key checkpoints: stem output, fusion features, final projection

3. Compare final output:
   - Shape should be [batch, n_tokens, hidden_dim]
   - For Gemma3n, n_tokens should be 256 (16x16 from MSFA)
   - Values should be in similar range

4. Common issues:
   - Wrong mean/std normalization
   - Incorrect tensor transpose/permute operations
   - Missing activation functions
   - Wrong padding/stride in convolutions
   - Incorrect attention implementation

5. To compare a specific tensor from llama.cpp:
   - Add: ggml_backend_tensor_get(tensor, buffer, 0, size);
   - Save to file
   - Load and compare with corresponding .npy file
""")

    print("\n✓ Debug script complete!")
    print(f"  Generated files: debug_*.npy")
    print(f"  Compare these with llama.cpp intermediate outputs\n")

if __name__ == "__main__":
    main()
